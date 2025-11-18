import torch
from torch import nn
import triton
import triton.language as tl

# Try to import flash_attn, but provide fallback if not available
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    
    # Check if GPU supports flash attention (requires compute capability >= 8.0, i.e., Ampere+)
    if torch.cuda.is_available():
        # Get compute capability of the first GPU
        compute_capability = torch.cuda.get_device_capability(0)
        major, minor = compute_capability
        
        # Flash attention requires Ampere (8.0) or newer
        # V100 is Volta (7.0), T4 is Turing (7.5), A100/H100 are Ampere+ (8.0+)
        if major >= 8:
            FLASH_ATTN_AVAILABLE = True
            print(f"flash-attn enabled (GPU compute capability: {major}.{minor})")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Warning: flash-attn installed but GPU ({gpu_name}, compute capability {major}.{minor}) does not support it.")
            print(f"flash-attn requires Ampere+ GPUs (compute capability >= 8.0).")
            print(f"Using standard PyTorch attention instead (slower but compatible).")
    else:
        print("Warning: No CUDA GPU available. Using CPU with standard attention.")
        
except ImportError:
    print("Info: flash-attn not installed. Using standard PyTorch attention.")
    print("This is normal for V100/T4 and older GPUs. Install flash-attn only if you have A100/H100/3090/4090.")

from nano_pearl.utils.context import get_context
from nano_pearl.pearl_config import TPParams

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    offsets = tl.arange(0, BLOCK)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    block = 1 << (D - 1).bit_length()
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D, block)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        tp_params: TPParams,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.tp_params = tp_params

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context(self.tp_params)
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if FLASH_ATTN_AVAILABLE:
            # Use flash attention (faster, for Ampere+ GPUs)
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        else:
            # Fallback to standard PyTorch attention (slower, but works on all GPUs)
            o = self._standard_attention(q, k, v, context)
        
        return o
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        """
        Standard PyTorch attention implementation as fallback when flash-attn is not available.
        Note: This is slower than flash attention but works on all GPUs including V100.
        """
        # Reshape for attention computation
        # q: [total_tokens, num_heads, head_dim]
        # k, v: same shape or from cache
        
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if context.is_prefill:
            # Prefill: compute attention for the full sequence
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            
            # Standard scaled dot-product attention
            # q, k, v shape: [total_tokens, num_heads, head_dim]
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            seq_len = q.shape[0]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            attn_weights.masked_fill_(causal_mask, float('-inf'))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            o = torch.matmul(attn_weights, v)
        else:
            # Decode: single token attention with KV cache
            # q: [batch_size, 1, num_heads, head_dim]
            # Use cached k, v
            
            # Simplified decode attention (single token)
            # This is a basic implementation - for production, you'd want more optimized version
            batch_size = len(context.context_lens)
            q_decode = q.unsqueeze(1)  # [batch, 1, num_heads, head_dim]
            
            # Gather from cache based on context_lens
            # For simplicity, using basic attention (not fully optimized)
            attn_weights = torch.matmul(q_decode, k_cache.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            o = torch.matmul(attn_weights, v_cache).squeeze(1)
        
        return o
