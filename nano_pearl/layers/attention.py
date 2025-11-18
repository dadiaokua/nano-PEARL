import torch
from torch import nn
import triton
import triton.language as tl

# Try to import flash_attn and check GPU compatibility
FLASH_ATTN_AVAILABLE = False

if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    gpu_name = torch.cuda.get_device_name(0)
    
    # Flash attention requires Ampere (8.0) or newer
    # V100 is Volta (7.0), T4 is Turing (7.5), A100/H100 are Ampere+ (8.0+)
    if major >= 8:
        # Try to import flash_attn
        try:
            from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
            FLASH_ATTN_AVAILABLE = True
            print(f"✓ flash-attn enabled (GPU: {gpu_name}, compute capability: {major}.{minor})")
        except ImportError:
            print(f"⚠ flash-attn not installed on compatible GPU ({gpu_name}, {major}.{minor})")
            print(f"  Install with: pip install flash-attn --no-build-isolation")
            print(f"  Using fallback attention (slower)")
            FLASH_ATTN_AVAILABLE = False
    else:
        print("="*80)
        print(f"⚠ GPU does not support flash-attention")
        print(f"GPU: {gpu_name} (compute capability: {major}.{minor})")
        print(f"")
        print(f"flash-attn requires Ampere+ GPUs (>= 8.0):")
        print(f"  ✅ Supported: A100, H100, A10, RTX 3090, RTX 4090")
        print(f"  ❌ Your GPU: {gpu_name} ({major}.{minor})")
        print(f"")
        print(f"⚠ Using fallback attention implementation (3-5x slower)")
        print(f"  For best performance, use Ampere+ GPU")
        print("="*80)
        FLASH_ATTN_AVAILABLE = False

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
            # Use flash attention (faster, Ampere+ GPUs)
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
            # Use fallback attention (slower, compatible with all GPUs)
            if context.is_prefill:
                o = self._fallback_prefill_attention(q, k, v, context, k_cache, v_cache)
            else:
                o = self._fallback_decode_attention(q, context, k_cache, v_cache)
        
        return o
    
    def _fallback_prefill_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                     context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        Fallback implementation for prefill phase using PyTorch operations.
        Handles variable-length sequences and prefix KV cache.
        
        Args:
            q: [total_tokens, num_heads, head_dim]
            k: [total_tokens, num_heads, head_dim]  
            v: [total_tokens, num_heads, head_dim]
            context: Context object with cu_seqlens_q, cu_seqlens_k, block_tables, etc.
            k_cache: KV cache for prefix
            v_cache: KV cache for prefix
        """
        # Use prefix cache if available
        if context.block_tables is not None:
            k, v = k_cache, v_cache
        
        # Get cumulative sequence lengths
        cu_seqlens_q = context.cu_seqlens_q.cpu().tolist()
        cu_seqlens_k = context.cu_seqlens_k.cpu().tolist()
        
        # Process each sequence separately
        outputs = []
        num_seqs = len(cu_seqlens_q) - 1
        
        for i in range(num_seqs):
            start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            
            # Extract this sequence's Q, K, V
            q_seq = q[start_q:end_q]  # [seq_len_q, num_heads, head_dim]
            k_seq = k[start_k:end_k]  # [seq_len_k, num_heads, head_dim]
            v_seq = v[start_k:end_k]  # [seq_len_k, num_heads, head_dim]
            
            seq_len_q = q_seq.shape[0]
            seq_len_k = k_seq.shape[0]
            
            # Reshape for batched matrix multiplication
            # [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
            q_seq = q_seq.transpose(0, 1)  # [num_heads, seq_len_q, head_dim]
            k_seq = k_seq.transpose(0, 1)  # [num_heads, seq_len_k, head_dim]
            v_seq = v_seq.transpose(0, 1)  # [num_heads, seq_len_k, head_dim]
            
            # Compute attention scores: Q @ K^T
            # [num_heads, seq_len_q, head_dim] @ [num_heads, head_dim, seq_len_k]
            # -> [num_heads, seq_len_q, seq_len_k]
            scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            if seq_len_q == seq_len_k:
                # Standard causal mask for generation
                causal_mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
                    diagonal=1
                )
            else:
                # For prefix caching: allow attention to all prefix tokens
                # Only mask future tokens in the non-prefix part
                causal_mask = torch.zeros(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device)
                prefix_len = seq_len_k - seq_len_q
                if prefix_len < seq_len_k:
                    # Causal mask for non-prefix tokens
                    causal_mask[:, prefix_len:] = torch.triu(
                        torch.ones(seq_len_q, seq_len_q, dtype=torch.bool, device=q.device),
                        diagonal=1
                    )
            
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            
            # Compute attention output: attn_weights @ V
            # [num_heads, seq_len_q, seq_len_k] @ [num_heads, seq_len_k, head_dim]
            # -> [num_heads, seq_len_q, head_dim]
            attn_output = torch.matmul(attn_weights, v_seq)
            
            # Reshape back to [seq_len_q, num_heads, head_dim]
            attn_output = attn_output.transpose(0, 1)
            outputs.append(attn_output)
        
        # Concatenate all sequences
        o = torch.cat(outputs, dim=0)
        return o
    
    def _fallback_decode_attention(self, q: torch.Tensor, context, 
                                    k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        Fallback implementation for decode phase using PyTorch operations.
        Handles block-based KV cache access.
        
        Args:
            q: [batch_size, num_heads, head_dim]
            context: Context object with context_lens, block_tables
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        
        context_lens = context.context_lens
        block_tables = context.block_tables
        
        # Get block size from k_cache
        block_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        
        # Handle Grouped Query Attention (GQA)
        num_queries_per_kv = num_heads // num_kv_heads
        
        outputs = []
        
        for i in range(batch_size):
            ctx_len = context_lens[i].item()
            
            # Gather K and V from blocks
            num_blocks_needed = (ctx_len + block_size - 1) // block_size
            
            # Get block indices for this sequence
            block_indices = block_tables[i, :num_blocks_needed]
            
            # Gather K and V blocks
            # k_cache[block_indices]: [num_blocks_needed, block_size, num_kv_heads, head_dim]
            k_seq = k_cache[block_indices].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
            v_seq = v_cache[block_indices].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
            # k_seq, v_seq: [ctx_len, num_kv_heads, head_dim]
            
            # Expand K and V for GQA if needed
            if num_queries_per_kv > 1:
                # Repeat each KV head for multiple query heads
                k_seq = k_seq.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
                k_seq = k_seq.reshape(ctx_len, num_heads, head_dim)
                v_seq = v_seq.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
                v_seq = v_seq.reshape(ctx_len, num_heads, head_dim)
            
            # Get query for this sequence
            q_seq = q[i:i+1]  # [1, num_heads, head_dim]
            
            # Compute attention
            # q: [1, num_heads, head_dim]
            # k: [ctx_len, num_heads, head_dim]
            
            # Transpose for matmul: [num_heads, 1, head_dim] @ [num_heads, head_dim, ctx_len]
            q_seq = q_seq.transpose(0, 1)  # [num_heads, 1, head_dim]
            k_seq = k_seq.transpose(0, 1)  # [num_heads, ctx_len, head_dim]
            v_seq = v_seq.transpose(0, 1)  # [num_heads, ctx_len, head_dim]
            
            # Compute scores
            scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
            # scores: [num_heads, 1, ctx_len]
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            
            # Compute output
            attn_output = torch.matmul(attn_weights, v_seq)
            # attn_output: [num_heads, 1, head_dim]
            
            # Reshape to [1, num_heads, head_dim]
            attn_output = attn_output.transpose(0, 1)
            outputs.append(attn_output)
        
        # Concatenate all outputs
        o = torch.cat(outputs, dim=0)
        return o
