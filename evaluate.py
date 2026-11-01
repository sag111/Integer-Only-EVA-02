import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from datasets import load_from_disk

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from int_pytorch import IntEvaModel

IMG_SIZE_MAP = {
    "Tiny": 336,
    "Small": 336,
    "Base": 448,
    "Large": 448,
}


def calculate_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def evaluate(
    checkpoint_path: str,
    val_dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "auto",
    max_samples: int = None,
):
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file.resolve()}")

    print(f"Loading checkpoint: {ckpt_file.resolve()}")
    checkpoint = torch.load(ckpt_file, map_location=selected_device, weights_only=False)

    model_type = checkpoint["model_type"]
    img_size = IMG_SIZE_MAP[model_type]
    in_scale = float(checkpoint["in_scale"])
    config = checkpoint["config"]
    input_bit = config.get("input_bit", 8)
    q_min = -(2 ** (input_bit - 1))
    q_max = (2 ** (input_bit - 1)) - 1

    model = IntEvaModel(checkpoint["params"], model_type=model_type).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transforms = timm.data.create_transform(
        input_size=(3, img_size, img_size),
        interpolation="bicubic",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        crop_pct=1.0,
        is_training=False,
    )

    def transform_fn(examples):
        pixel_values = [transforms(img.convert("RGB")) for img in examples["image"]]
        label_key = "label" if "label" in examples else "labels"
        return {"pixel_values": pixel_values, "label": examples[label_key]}

    print(f"Loading dataset: {val_dataset_path}")
    val_dataset = load_from_disk(val_dataset_path)

    if max_samples is not None and max_samples < len(val_dataset):
        val_dataset = val_dataset.select(range(max_samples))

    val_dataset.set_transform(transform_fn)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if selected_device == "cuda" else False,
        shuffle=False,
    )

    correct_1, correct_5, total_processed = 0.0, 0.0, 0

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc=f"Evaluating {model_type}", leave=True):
            images_fp32 = batch["pixel_values"].to(selected_device, non_blocking=True)
            labels = batch["label"].to(selected_device, non_blocking=True)

            images_int8 = torch.round(images_fp32 / in_scale).clamp(q_min, q_max).to(torch.int8)

            logits_int16 = model(images_int8)

            k_val = (1, 5) if logits_int16.size(1) >= 5 else (1,)
            acc_res = calculate_accuracy(logits_int16.float(), labels, topk=k_val)

            correct_1 += acc_res[0]
            if len(acc_res) > 1:
                correct_5 += acc_res[1]
            total_processed += labels.size(0)

    top1 = (100.0 * correct_1 / total_processed) if total_processed > 0 else 0.0
    top5 = (100.0 * correct_5 / total_processed) if total_processed > 0 else 0.0

    print(f"Top-1: {top1:.3f}%")
    print(f"Top-5: {top5:.3f}%")
    print(f"Total: {total_processed}")

    return top1, top5


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure PyTorch Integer Model on ImageNet Validation Dataset")
    parser.add_argument(
        "--val_dataset", "-v",
        type=str,
        required=True,
        help="Path to ImageNet validation dataset"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to integer model checkpoint (.pt)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", "-j",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device"
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=None,
        help="Optional limit of samples to evaluate"
    )

    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        val_dataset_path=args.val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
