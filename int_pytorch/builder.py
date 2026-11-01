import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

class IntConfig:
    def __init__(self, config_dict: dict):
        self.input_bit = config_dict.get('input_bit', 8)
        self.weight_bit = config_dict.get('weight_bit', 8)
        self.internal_bit = config_dict.get('internal_bit', 16)
        self.accumulator_bit = config_dict.get('accumulator_bit', 32)
        self.input_bit_norm = config_dict.get('input_bit_norm', 16)
        self.bias_bit = config_dict.get('bias_bit', 16)
        self.rope_bit = config_dict.get('rope_bit', 8)

        self.internal_dtype = self._get_torch_dtype(self.internal_bit)
        self.acc_dtype = torch.int64 

    def _get_torch_dtype(self, bits: int) -> torch.dtype:
        if bits <= 8: return torch.int8
        elif bits <= 16: return torch.int16
        elif bits <= 32: return torch.int32
        else: return torch.int64

    @staticmethod
    def get_limits(bits: int):
        min_val = -(2 ** (bits - 1))
        max_val = (2 ** (bits - 1)) - 1
        return min_val, max_val

    def clip_and_cast(self, x: torch.Tensor, bits: int, target_dtype: torch.dtype) -> torch.Tensor:
        q_min, q_max = self.get_limits(bits)
        return torch.clamp(x, min=q_min, max=q_max).to(target_dtype)


class NPZLoader:
    def __init__(self, npz_or_dict):
        if isinstance(npz_or_dict, str):
            self.params = dict(np.load(npz_or_dict))
        elif isinstance(npz_or_dict, dict):
            self.params = npz_or_dict
        else:
            raise ValueError("Expected file path or dictionary")

        if '__meta__' not in self.params:
            raise KeyError("Missing '__meta__' in parameters.")
        
        meta = json.loads(str(self.params['__meta__']))
        self.config = IntConfig(meta['__global_info__']['config'])
        
    def get(self, name: str, device: str = 'cpu') -> torch.Tensor:
        if name not in self.params:
            raise KeyError(f"Parameter {name} not found.")
        
        arr = self.params[name]
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.from_numpy(arr).to(device)

class IntMath:
    @staticmethod
    def round_shift(x_acc: torch.Tensor, shift: Union[int, torch.Tensor]) -> torch.Tensor:
        if isinstance(shift, int):
            if shift <= 0:
                return x_acc
            shift = torch.tensor(shift, dtype=x_acc.dtype, device=x_acc.device)
            
        shift = shift.to(x_acc.dtype)
        safe_shift = torch.maximum(shift, torch.tensor(1, dtype=x_acc.dtype, device=x_acc.device))
        
        q = x_acc >> safe_shift
        mask = (1 << safe_shift) - 1
        remainder = x_acc & mask
        half = 1 << (safe_shift - 1)
        q_is_odd = q & 1
        
        round_up = (remainder > half) | ((remainder == half) & (q_is_odd == 1))
        rounded = q + round_up.to(x_acc.dtype)
        
        if isinstance(shift, torch.Tensor):
            return torch.where(shift > 0, rounded, x_acc)
        return rounded

    @staticmethod
    def shift_exp(x: torch.Tensor, prefix: str, loader: NPZLoader, config: IntConfig) -> torch.Tensor:
        device = x.device
        x_acc = x.to(config.acc_dtype)
        
        term1 = x_acc >> 1
        term2 = x_acc >> 4
        x_acc = x_acc + term1 - term2
        x_acc = config.clip_and_cast(x_acc, config.internal_bit, config.acc_dtype)
        
        x0_int = loader.get(f"{prefix}.x0_int", device).to(config.acc_dtype)
        n = loader.get(f"{prefix}.n", device).to(config.acc_dtype)
        
        try:
            limit_val = loader.get(f"{prefix}.limit_val", device).to(config.acc_dtype)
            x_acc = torch.maximum(x_acc, limit_val)
        except KeyError:
            pass
            
        m_magic = loader.get(f"{prefix}.m_magic", device).to(config.acc_dtype)
        s_magic = loader.get(f"{prefix}.s_magic", device).to(config.acc_dtype)
        
        q_raw = x_acc * m_magic
        q = q_raw >> s_magic
        
        r = x_acc - (x0_int * q)
        r = config.clip_and_cast(r, config.internal_bit, config.acc_dtype)
        
        val = r - (x0_int * 2)
        val = config.clip_and_cast(val, config.internal_bit, config.acc_dtype)
        
        shift_amt = n - q - 1
        zero = torch.tensor(0, dtype=config.acc_dtype, device=device)
        max_shift = torch.tensor(config.accumulator_bit - 1, dtype=config.acc_dtype, device=device)
        
        safe_pos_shift = torch.clamp(shift_amt, min=zero, max=max_shift)
        safe_neg_shift = torch.clamp(-shift_amt, min=zero, max=max_shift)
        
        pos_shift_val = val << safe_pos_shift
        neg_shift_val = val >> safe_neg_shift
        
        exp_acc = torch.where(shift_amt >= 0, pos_shift_val, neg_shift_val)
        
        try:
            norm_shift = loader.get(f"{prefix}.norm_shift", device).to(config.acc_dtype)
            exp_acc = exp_acc >> norm_shift
        except KeyError:
            pass
            
        _, q_max = config.get_limits(config.internal_bit)
        exp_acc = torch.clamp(exp_acc, min=0, max=q_max)
        
        return exp_acc.to(config.internal_dtype)

class IntLinear(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, is_3d: bool = True):
        super().__init__()
        self.config = config
        self.is_3d = is_3d
        
        self.register_buffer('weight', loader.get(f"{prefix}.weight"))
        self.register_buffer('mul_shr', loader.get(f"{prefix}.mul_shr"))
        try:
            self.register_buffer('bias', loader.get(f"{prefix}.bias"))
        except KeyError:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if self.is_3d:
            x = x.view(-1, x.shape[-1])
            
        if x.is_cuda:
            # PyTorch cuBLAS does not support integer GEMM on CUDA (addmm_cuda is not implemented for integer types).
            # We temporarily cast to float for GPU execution and cast back to integer accumulator.
            out_acc = F.linear(x.float(), self.weight.float()).to(self.config.acc_dtype)
        else:
            # On CPU, compute directly with pure integer arithmetic using integer accumulator.
            out_acc = F.linear(x.to(self.config.acc_dtype), self.weight.to(self.config.acc_dtype))
        
        mul_shr_acc = self.mul_shr.to(self.config.acc_dtype)
        out_acc = out_acc >> mul_shr_acc
        out_acc = self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.acc_dtype)
        
        if self.bias is not None:
            out_acc = out_acc + self.bias.to(self.config.acc_dtype)
            out_acc = self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.acc_dtype)
            
        out = out_acc.to(self.config.internal_dtype)
        
        if self.is_3d:
            out = out.view(*orig_shape[:-1], -1)
        return out


class IntConv2d(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, stride: int, padding: int = 0):
        super().__init__()
        self.config = config
        self.stride = stride
        self.padding = padding
        
        self.register_buffer('weight', loader.get(f"{prefix}.weight"))
        self.register_buffer('mul_shr', loader.get(f"{prefix}.mul_shr"))
        try:
            self.register_buffer('bias', loader.get(f"{prefix}.bias"))
        except KeyError:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            # PyTorch cuDNN does not support integer Conv2d on CUDA.
            # We temporarily cast to float for GPU execution and cast back to integer accumulator.
            out_acc = F.conv2d(x.float(), self.weight.float(), stride=self.stride, padding=self.padding).to(self.config.acc_dtype)
        else:
            # On CPU, compute directly with pure integer arithmetic using integer accumulator.
            out_acc = F.conv2d(x.to(self.config.acc_dtype), self.weight.to(self.config.acc_dtype), stride=self.stride, padding=self.padding)
        
        mul_shr_acc = self.mul_shr.to(self.config.acc_dtype)
        out_acc = out_acc >> mul_shr_acc
        out_acc = self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.acc_dtype)
        
        if self.bias is not None:
            bias_acc = self.bias.view(1, -1, 1, 1).to(self.config.acc_dtype)
            out_acc = out_acc + bias_acc
            out_acc = self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.acc_dtype)
            
        return out_acc.to(self.config.internal_dtype)


class IntMatMul(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, transpose_b: bool = True):
        super().__init__()
        self.config = config
        self.transpose_b = transpose_b
        self.register_buffer('mul_shr', loader.get(f"{prefix}.mul_shr"))

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.transpose_b:
            B = B.transpose(-2, -1)
            
        if A.is_cuda:
            # PyTorch cuBLAS does not support integer batched GEMM on CUDA (bmm is not implemented for integer types).
            # We temporarily cast to float for GPU execution and cast back to integer accumulator.
            out_acc = (A.float() @ B.float()).to(self.config.acc_dtype)
        else:
            # On CPU, compute directly with pure integer arithmetic using integer accumulator.
            out_acc = A.to(self.config.acc_dtype) @ B.to(self.config.acc_dtype)
        
        mul_shr_acc = self.mul_shr.to(self.config.acc_dtype)
        out_acc = out_acc >> mul_shr_acc
        
        return self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.internal_dtype)


class IntElementWiseMul(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader):
        super().__init__()
        self.config = config
        self.register_buffer('mul_shr', loader.get(f"{prefix}.mul_shr"))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1_acc = x1.to(self.config.acc_dtype)
        x2_acc = x2.to(self.config.acc_dtype)
        
        out_acc = x1_acc * x2_acc
        
        mul_shr_acc = self.mul_shr.to(self.config.acc_dtype)
        out_acc = out_acc >> mul_shr_acc
        
        return self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.internal_dtype)


class IntAct(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, target_bit: int):
        super().__init__()
        self.config = config
        self.target_bit = target_bit
        
        self.register_buffer('multiplier', loader.get(f"{prefix}.multiplier"))
        self.register_buffer('shift', loader.get(f"{prefix}.shift"))
        
        try:
            self.register_buffer('id_multiplier', loader.get(f"{prefix}.id_multiplier"))
            self.register_buffer('id_shift', loader.get(f"{prefix}.id_shift"))
            self.has_identity = True
        except KeyError:
            self.has_identity = False

    def _rescale(self, data: torch.Tensor, mult: torch.Tensor, shift: torch.Tensor, clip_to_internal: bool) -> torch.Tensor:
        data_acc = data.to(self.config.acc_dtype)
        mult_acc = mult.to(self.config.acc_dtype)
        
        p_acc = data_acc * mult_acc
        y_acc = IntMath.round_shift(p_acc, shift)
        
        if clip_to_internal:
            return self.config.clip_and_cast(y_acc, self.config.internal_bit, self.config.acc_dtype)
        return y_acc

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_identity and identity is not None:
            out_acc = self._rescale(x, self.multiplier, self.shift, clip_to_internal=False)
            id_acc = self._rescale(identity, self.id_multiplier, self.id_shift, clip_to_internal=False)
            
            acc_sum = out_acc + id_acc
            out_acc = self.config.clip_and_cast(acc_sum, self.config.internal_bit, self.config.acc_dtype)
        else:
            out_acc = self._rescale(x, self.multiplier, self.shift, clip_to_internal=True)
            
        target_dtype = self.config._get_torch_dtype(self.target_bit)
        return self.config.clip_and_cast(out_acc, self.target_bit, target_dtype)

class IntLayerNorm(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader):
        super().__init__()
        self.config = config
        
        self.register_buffer('N_const', loader.get(f"{prefix}.N_const"))
        self.register_buffer('half_N', loader.get(f"{prefix}.half_N"))
        self.register_buffer('sq_shift', loader.get(f"{prefix}.sq_shift")[0])
        self.register_buffer('var_shift', loader.get(f"{prefix}.var_shift")[0])
        self.register_buffer('init_k', loader.get(f"{prefix}.init_k"))
        self.register_buffer('large_const', loader.get(f"{prefix}.large_const"))
        self.register_buffer('shift_y', loader.get(f"{prefix}.shift_y"))
        self.register_buffer('bias', loader.get(f"{prefix}.bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = self.config.acc_dtype
        q_min, q_max = self.config.get_limits(self.config.internal_bit)
        acc_min, acc_max = self.config.get_limits(self.config.accumulator_bit)
        
        x_acc = x.to(acc)
        N_const = self.N_const.to(acc)
        half_N = self.half_N.to(acc)
        
        sum_x = x_acc.sum(dim=-1, keepdim=True)
        
        Q = torch.div(sum_x, N_const, rounding_mode='trunc')
        abs_sum = torch.abs(sum_x)
        Q_abs = torch.div(abs_sum, N_const, rounding_mode='trunc')
        R = abs_sum - (Q_abs * N_const)
        
        cond_gt = R > half_N
        cond_eq = R == half_N
        q_odd = (Q & 1) == 1
        
        should_add_1 = cond_gt | (cond_eq & q_odd)
        correction = torch.sign(sum_x) * should_add_1.to(acc)
        
        mean_int = torch.clamp(Q + correction, min=q_min, max=q_max)
        y_int = torch.clamp(x_acc - mean_int, min=q_min, max=q_max)
        
        y_sq_int = torch.clamp(y_int * y_int, min=acc_min, max=acc_max)
        y_sq_int = IntMath.round_shift(y_sq_int, self.sq_shift)
        
        sq_shift_val = self.sq_shift.item()
        var_shift_val = self.var_shift.item()
        max_sq_clip = q_max if sq_shift_val >= (sq_shift_val + var_shift_val) else acc_max
        y_sq_int = torch.clamp(y_sq_int, min=0, max=max_sq_clip)
        
        var_int = torch.clamp(y_sq_int.sum(dim=-1, keepdim=True), min=1, max=acc_max)
        var_int = torch.clamp(IntMath.round_shift(var_int, self.var_shift), min=1, max=q_max)
        
        k = self.init_k.to(acc).expand_as(var_int).clone()
        for _ in range(6):
            q = torch.div(var_int, k, rounding_mode='trunc')
            k = ((k + q) >> 1)
            k = torch.clamp(k, min=1, max=q_max)
            
        std_int = k
        large_const = self.large_const.to(acc)
        factor = torch.div(large_const, std_int, rounding_mode='trunc')
        
        shift_y_acc = self.shift_y.to(acc)
        y_norm = (y_int * factor) >> shift_y_acc
        y_norm = torch.clamp(y_norm, min=q_min, max=q_max)
        
        bias_acc = self.bias.to(acc)
        out_acc = torch.clamp(y_norm + bias_acc, min=q_min, max=q_max)
        
        return out_acc.to(self.config.internal_dtype)


class IntRoPE(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, rotate_half: bool = False):
        super().__init__()
        self.config = config
        self.rotate_half = rotate_half
        self.register_buffer('sin', loader.get(f"{prefix}.sin"))
        self.register_buffer('cos', loader.get(f"{prefix}.cos"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = self.config.acc_dtype
        x_acc = x.to(acc)
        
        if self.rotate_half:
            x1, x2 = x_acc.chunk(2, dim=-1)
            x_rotated = torch.cat([-x2, x1], dim=-1)
        else:
            x_pairs = x_acc.view(*x_acc.shape[:-1], -1, 2)
            x_even, x_odd = x_pairs.chunk(2, dim=-1)
            x_rotated = torch.cat([-x_odd, x_even], dim=-1).view_as(x_acc)
            
        q_min, q_max = self.config.get_limits(self.config.internal_bit)
        
        cos_acc = self.cos.to(acc)
        sin_acc = self.sin.to(acc)
        
        x_term = torch.clamp(x_acc * cos_acc, min=q_min, max=q_max)
        rot_term = torch.clamp(x_rotated * sin_acc, min=q_min, max=q_max)
        
        out_acc = torch.clamp(x_term + rot_term, min=q_min, max=q_max)
        out_acc = out_acc >> (self.config.rope_bit - 1)
        
        return self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.internal_dtype)


class IntSoftmax(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.loader = loader
        self.out_shift = loader.get(f"{prefix}.out_shift")[0].item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = self.config.acc_dtype
        q_min, q_max = self.config.get_limits(self.config.internal_bit)
        in_min, in_max = self.config.get_limits(self.config.input_bit)
        
        x_acc = x.to(acc)
        x_acc = torch.clamp(x_acc, min=in_min, max=in_max)
        
        x_max = x_acc.max(dim=-1, keepdim=True)[0]
        x_norm = torch.clamp(x_acc - x_max, min=q_min, max=q_max)
        
        exp_int = IntMath.shift_exp(x_norm, self.prefix, self.loader, self.config).to(acc)
        
        exp_sum = torch.clamp(exp_int.sum(dim=-1, keepdim=True), min=1, max=q_max)
        q_max_tensor = torch.tensor(q_max, dtype=acc, device=x.device)
        factor = torch.div(q_max_tensor, exp_sum, rounding_mode='trunc')
        
        out_acc = exp_int * factor
        
        if self.out_shift > 0:
            out_acc = IntMath.round_shift(out_acc, self.out_shift)
            out_acc = torch.clamp(out_acc, min=0, max=(q_max >> self.out_shift))
        else:
            out_acc = torch.clamp(out_acc, min=0, max=q_max)
            
        return self.config.clip_and_cast(out_acc, self.config.internal_bit, self.config.internal_dtype)


class IntSiLU(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.loader = loader
        self.register_buffer('sigmoid_shift', loader.get(f"{prefix}.sigmoid_shift")[0])
        self.register_buffer('sigmoid_one', loader.get(f"{prefix}.sigmoid_one")[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = self.config.acc_dtype
        q_min, q_max = self.config.get_limits(self.config.internal_bit)
        in_min, in_max = self.config.get_limits(self.config.input_bit)
        
        x_acc = x.to(acc)
        x_acc = torch.clamp(x_acc, min=in_min, max=in_max)
        
        mask_pos = x_acc >= 0
        x_neg = torch.where(mask_pos, -x_acc, x_acc)
        
        zeros = torch.zeros_like(x_acc)
        
        exp_zero = IntMath.shift_exp(zeros, self.prefix, self.loader, self.config).to(acc)
        exp_neg = IntMath.shift_exp(x_neg, self.prefix, self.loader, self.config).to(acc)
        
        exp_sum = torch.clamp(exp_zero + exp_neg, min=1, max=q_max)
        numerator = torch.where(mask_pos, exp_zero, exp_neg)
        num_shifted = numerator * 2
        
        den_shift = self.sigmoid_shift.item() - 1
        if den_shift > 0:
            den_shifted = IntMath.round_shift(exp_sum, den_shift)
        else:
            den_shifted = exp_sum
            
        den_shifted = torch.clamp(den_shifted, min=1, max=q_max)
        sigmoid_int = torch.div(num_shifted, den_shifted, rounding_mode='trunc')
        sigmoid_one_acc = self.sigmoid_one.to(acc).item()
        sigmoid_int = torch.clamp(sigmoid_int, min=0, max=sigmoid_one_acc)
        
        out_acc = torch.clamp(x_acc * sigmoid_int, min=q_min, max=q_max)
        
        return out_acc.to(self.config.internal_dtype)

class IntEvaAttention(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, arch: dict):
        super().__init__()
        self.arch = arch
        self.heads = arch["heads"]
        self.qkv_fused = arch["qkv_fused"]
        
        if self.qkv_fused:
            self.qkv = IntLinear(f"{prefix}.qkv", config, loader)
            self.qact_qkv = IntAct(f"{prefix}.qact_qkv", config, loader, config.input_bit)
        else:
            self.q_proj = IntLinear(f"{prefix}.q_proj", config, loader)
            self.k_proj = IntLinear(f"{prefix}.k_proj", config, loader)
            self.v_proj = IntLinear(f"{prefix}.v_proj", config, loader)
            self.qact_q = IntAct(f"{prefix}.qact_q", config, loader, config.input_bit)
            self.qact_k = IntAct(f"{prefix}.qact_k", config, loader, config.input_bit)
            self.qact_v = IntAct(f"{prefix}.qact_v", config, loader, config.input_bit)
            
        self.int_rope = IntRoPE(f"{prefix}.int_rope", config, loader)
        self.qact_rope_q = IntAct(f"{prefix}.qact_rope_q", config, loader, config.input_bit)
        self.qact_rope_k = IntAct(f"{prefix}.qact_rope_k", config, loader, config.input_bit)
        
        self.matmul_q_k = IntMatMul(f"{prefix}.matmul_q_k", config, loader, transpose_b=True)
        self.qact_attn = IntAct(f"{prefix}.qact_attn", config, loader, config.input_bit)
        self.int_softmax = IntSoftmax(f"{prefix}.int_softmax", config, loader)
        
        self.matmul_attn_v = IntMatMul(f"{prefix}.matmul_attn_v", config, loader, transpose_b=False)
        self.qact_before_proj = IntAct(f"{prefix}.qact_before_proj", config, loader, config.input_bit)
        self.proj = IntLinear(f"{prefix}.proj", config, loader)
        self.qact_final = IntAct(f"{prefix}.qact_final", config, loader, config.input_bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        if self.qkv_fused:
            qkv = self.qact_qkv(self.qkv(x))
            qkv = qkv.view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.qact_q(self.q_proj(x)).view(B, N, self.heads, -1).permute(0, 2, 1, 3)
            k = self.qact_k(self.k_proj(x)).view(B, N, self.heads, -1).permute(0, 2, 1, 3)
            v = self.qact_v(self.v_proj(x)).view(B, N, self.heads, -1).permute(0, 2, 1, 3)

        q_pref, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
        k_pref, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
        
        q_rot = self.int_rope(q_patch)
        k_rot = self.int_rope(k_patch)
        
        q = torch.cat([q_pref, q_rot], dim=2)
        k = torch.cat([k_pref, k_rot], dim=2)
        
        q = self.qact_rope_q(q)
        k = self.qact_rope_k(k)

        attn = self.matmul_q_k(q, k)
        attn = self.qact_attn(attn)
        
        attn = self.int_softmax(attn)
        
        x_out = self.matmul_attn_v(attn, v)
        x_out = self.qact_before_proj(x_out)
        
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.qact_final(self.proj(x_out))
        return x_out


class IntEvaMLP(nn.Module):
    def __init__(self, prefix: str, config: IntConfig, loader: NPZLoader, arch: dict):
        super().__init__()
        self.swiglu = arch["swiglu"]
        
        if self.swiglu:
            self.fc1_g = IntLinear(f"{prefix}.fc1_g", config, loader)
            self.qact_g = IntAct(f"{prefix}.qact_g", config, loader, config.input_bit)
            self.fc1_x = IntLinear(f"{prefix}.fc1_x", config, loader)
            self.qact_x = IntAct(f"{prefix}.qact_x", config, loader, config.input_bit)
            
            self.act = IntSiLU(f"{prefix}.act", config, loader)
            self.glu_mul = IntElementWiseMul(f"{prefix}.glu_mul", config, loader)
            
            self.qact_before_norm = IntAct(f"{prefix}.qact_before_norm", config, loader, config.input_bit_norm)
            self.norm = IntLayerNorm(f"{prefix}.norm", config, loader)
            self.qact_before_fc2 = IntAct(f"{prefix}.qact_before_fc2", config, loader, config.input_bit)
            
            self.fc2 = IntLinear(f"{prefix}.fc2", config, loader)
            self.qact_final = IntAct(f"{prefix}.qact_final", config, loader, config.input_bit)
        else:
            self.fc1 = IntLinear(f"{prefix}.fc1", config, loader)
            self.qact_before_chunk = IntAct(f"{prefix}.qact_before_chunk", config, loader, config.input_bit)
            
            self.act = IntSiLU(f"{prefix}.act", config, loader)
            self.glu_mul = IntElementWiseMul(f"{prefix}.glu_mul", config, loader)
            self.qact_glu = IntAct(f"{prefix}.qact_glu", config, loader, config.input_bit)
            
            self.fc2 = IntLinear(f"{prefix}.fc2", config, loader)
            self.qact_final = IntAct(f"{prefix}.qact_final", config, loader, config.input_bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            x_g = self.qact_g(self.fc1_g(x))
            x_m = self.qact_x(self.fc1_x(x))
            
            x_g = self.act(x_g)
            x = self.glu_mul(x_g, x_m)
            
            x = self.qact_before_norm(x)
            x = self.norm(x)
            x = self.qact_before_fc2(x)
            
            x = self.qact_final(self.fc2(x))
        else:
            x = self.qact_before_chunk(self.fc1(x))
            x1, x2 = x.chunk(2, dim=-1)
            
            x1 = self.act(x1)
            x = self.glu_mul(x1, x2)
            x = self.qact_glu(x)
            
            x = self.qact_final(self.fc2(x))
        return x


class IntEvaModel(nn.Module):
    def __init__(self, npz_or_dict, model_type: str = "Tiny"):
        super().__init__()
        self.loader = NPZLoader(npz_or_dict)
        self.config = self.loader.config
        
        arch_map = {
            "Tiny":  {"embed_dim": 192,  "heads": 3,  "depth": 12, "qkv_fused": True,  "swiglu": False},
            "Small": {"embed_dim": 384,  "heads": 6,  "depth": 12, "qkv_fused": True,  "swiglu": False},
            "Base":  {"embed_dim": 768,  "heads": 12, "depth": 12, "qkv_fused": False, "swiglu": True},
            "Large": {"embed_dim": 1024, "heads": 16, "depth": 24, "qkv_fused": False, "swiglu": True},
        }
        self.arch = arch_map[model_type]
        
        self.patch_proj = IntConv2d("patch_embed.proj", self.config, self.loader, stride=14, padding=0)
        self.qact_patch = IntAct("patch_embed.qact", self.config, self.loader, self.config.internal_bit)
        
        self.register_buffer('cls_token', self.loader.get("cls_token_integer"))
        self.register_buffer('pos_embed', self.loader.get("pos_embed_integer"))
        
        self.qact1 = IntAct("qact1", self.config, self.loader, self.config.input_bit_norm)
        
        self.blocks = nn.ModuleList()
        for i in range(self.arch["depth"]):
            pref = f"blocks.{i}"
            blk = nn.ModuleDict({
                'norm1': IntLayerNorm(f"{pref}.norm1", self.config, self.loader),
                'qact1': IntAct(f"{pref}.qact1", self.config, self.loader, self.config.input_bit),
                'attn':  IntEvaAttention(f"{pref}.attn", self.config, self.loader, self.arch),
                'qact2': IntAct(f"{pref}.qact2", self.config, self.loader, self.config.input_bit_norm),
                'norm2': IntLayerNorm(f"{pref}.norm2", self.config, self.loader),
                'qact3': IntAct(f"{pref}.qact3", self.config, self.loader, self.config.input_bit),
                'mlp':   IntEvaMLP(f"{pref}.mlp", self.config, self.loader, self.arch),
                'qact4': IntAct(f"{pref}.qact4", self.config, self.loader, self.config.input_bit_norm),
            })
            self.blocks.append(blk)
            
        self.qact_before_fc_norm = IntAct("qact_before_fc_norm", self.config, self.loader, self.config.input_bit_norm)
        self.fc_norm = IntLayerNorm("fc_norm", self.config, self.loader)
        self.qact_before_head = IntAct("qact_before_head", self.config, self.loader, self.config.input_bit)
        
        self.head = IntLinear("head", self.config, self.loader, is_3d=False)
        
        try:
            self.register_buffer('logit_mults', self.loader.get("logit_multipliers"))
        except KeyError:
            self.logit_mults = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.qact_patch(x)
        x = self.qact1(x, identity=self.pos_embed)
        
        seq_len = x.shape[1]
        
        for blk in self.blocks:
            residual = x
            x = blk['qact1'](blk['norm1'](x))
            x = blk['attn'](x)
            x = blk['qact2'](x, identity=residual)
            
            residual = x
            x = blk['qact3'](blk['norm2'](x))
            x = blk['mlp'](x)
            x = blk['qact4'](x, identity=residual)
            
        acc = self.config.acc_dtype
        x_acc = x[:, 1:, :].to(acc)
        x_sum = x_acc.sum(dim=1)
        
        seq_len_patches = seq_len - 1
        len_const = torch.tensor(seq_len_patches, dtype=acc, device=x.device)
        
        Q = torch.div(x_sum, len_const, rounding_mode='trunc')
        abs_sum = torch.abs(x_sum)
        Q_abs = torch.div(abs_sum, len_const, rounding_mode='trunc')
        R = abs_sum - (Q_abs * len_const)
        
        half_len = len_const // 2
        cond_gt = R > half_len
        cond_eq = R == half_len
        q_odd = (Q & 1) == 1
        
        should_add_1 = cond_gt | (cond_eq & q_odd)
        correction = torch.sign(x_sum) * should_add_1.to(acc)
        
        x_pool = self.config.clip_and_cast(Q + correction, self.config.internal_bit, self.config.internal_dtype)
        
        x = self.qact_before_fc_norm(x_pool)
        x = self.fc_norm(x)
        x = self.qact_before_head(x)
        
        logits = self.head(x)
        
        if self.logit_mults is not None:
            logits_acc = logits.to(acc) * self.logit_mults.to(acc)
            align_shift_val = self.config.internal_bit - 1
            logits_acc = IntMath.round_shift(logits_acc, align_shift_val)
            logits = self.config.clip_and_cast(logits_acc, self.config.internal_bit, self.config.internal_dtype)
            
        return logits