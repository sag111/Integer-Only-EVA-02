import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import timm

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


def run_inference(checkpoint_path: str, image_path: str, device: str = "auto"):
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file.resolve()}")

    img_file = Path(image_path)
    if not img_file.is_file():
        raise FileNotFoundError(f"Image file not found: {img_file.resolve()}")

    print(f"Loading checkpoint: {ckpt_file.resolve()}")
    checkpoint = torch.load(ckpt_file, map_location=selected_device, weights_only=False)

    model_type = checkpoint["model_type"]
    img_size = IMG_SIZE_MAP[model_type]
    in_scale = checkpoint["in_scale"]
    config = checkpoint["config"]
    input_bit = config.get("input_bit", 8)
    q_min = -(2 ** (input_bit - 1))
    q_max = (2 ** (input_bit - 1)) - 1

    model = IntEvaModel(checkpoint["params"], model_type=model_type).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading image: {img_file.resolve()}")
    img = Image.open(img_file).convert("RGB")

    transforms = timm.data.create_transform(
        input_size=(3, img_size, img_size),
        interpolation="bicubic",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        crop_pct=1.0,
        is_training=False,
    )
    img_tensor = transforms(img).unsqueeze(0)
    img_int8 = torch.round(img_tensor / in_scale).clamp(q_min, q_max).to(torch.int8).to(selected_device)

    with torch.no_grad():
        logits_int16 = model(img_int8)

    logits_float = logits_int16.float()
    top5_values, top5_indices = torch.topk(logits_float, k=5, dim=1)

    print(f"Input: {list(img_int8.shape)}, {img_int8.dtype}")
    print("Top-5:")
    for rank in range(5):
        idx = top5_indices[0][rank].item()
        val = int(top5_values[0][rank].item())
        print(f"Rank {rank + 1}: class {idx}, logit {val}")

    return logits_int16


def main():
    parser = argparse.ArgumentParser(description="Pure PyTorch Integer Inference for EVA-02")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to integer model checkpoint (.pt)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)"
    )

    args = parser.parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        device=args.device,
    )


if __name__ == "__main__":
    main()
