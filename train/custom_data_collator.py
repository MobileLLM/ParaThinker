from llamafactory.data.collator import DataCollatorForSeq2Seq
from dataclasses import dataclass, field
from typing import Any, Dict, Literal
import torch
import random
from typing import Any, Optional, List, Dict, Union
import os
os.environ['CUDA_VISIBLE_DEVIC ES'] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

# custom attention mask for ParaThinker
def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.
    
    Set prompt part as `1` and summary part as `10`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    device = attention_mask_with_indices.device
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Special case 1: All non-zero tokens can attend to tokens with ID 1, which represents the prompt part
    is_special_token_1 = (expanded_mask == 1)
    can_attend_to_1 = padding_mask.transpose(-1, -2) * is_special_token_1
    # Special case 2: Tokens with ID 10 can attend to all non-zero tokens
    is_special_token_10 = (expanded_mask.transpose(-1, -2) == 10)
    token_10_can_attend = is_special_token_10 * padding_mask
    
    # Combine all attention patterns (using logical OR)
    combined_mask = torch.clamp(attention_mask_4d + can_attend_to_1 + token_10_can_attend, 0, 1)
    
    # Use the lower triangular mask to zero out the upper triangular part
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=combined_mask.dtype, device=device))
    combined_mask = combined_mask * tril_mask 
    
    # Convert to the final mask with the desired dtype in one clean step
    # Create a new tensor directly with the target dtype
    if dtype == torch.bool:
        # For boolean mask (True = masked, False = not masked)
        final_attention_mask = ~(combined_mask > 0)
    else:
        # For float masks (0 = keep, large negative = mask)
        min_dtype = torch.finfo(dtype).min
        zeros = torch.zeros(combined_mask.shape, dtype=dtype, device=device)
        ones = torch.ones(combined_mask.shape, dtype=dtype, device=device)
        final_attention_mask = torch.where(combined_mask > 0, zeros, ones * min_dtype)
    
    return final_attention_mask

@dataclass
class ParallelCoTsDataCollator(DataCollatorForSeq2Seq):
    
    block_diag_attn: bool = True
    intra_rope: bool = True # Used for Thought-Specific Positional Embedding
    ablation_study = False
    pe_ablation = False
    special_token_ablation = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.bfloat16
    think_token_ids = [(151665 + off * 2) for off in range(9)] # <think1> ~ <think8> + <summary>
    extra_think_token_ids = [151665 + 6*2, 151665 + 7*2]
    extrapolation_probs: List[float] = field(default_factory=lambda: [0.3, 0.7]) # [apply_extra, don't_apply_extra], used for Extensible Special Tokens Training
    
    def __call__(self, features: List[Dict[str, Any]]):
        # Get the basic batch from parent
        batch = super().__call__(features)
        
        # Determine whether to apply extrapolation feature (30% probability)
        apply_extra = bool(torch.multinomial(torch.tensor(self.extrapolation_probs), 1).item() == 0)
            
        bsz, seq_len = batch["input_ids"].shape
        assert bsz == 1
    
        # Update sequence length
        _, seq_len = batch["input_ids"].shape
        
        if self.ablation_study:       
            batch["position_ids"] = torch.arange(seq_len, device=batch["input_ids"].device).long().expand((bsz, -1))
            return batch
        
        if apply_extra and not self.ablation_study:
            for b in range(bsz):
                assert bsz == 1
                think_start_indices = []
                think_start_tokens = []
                for i in range(seq_len):
                    if batch["input_ids"][b, i].item() in self.think_token_ids:
                        think_start_indices.append(i)
                        think_start_tokens.append(batch["input_ids"][b, i].item())
                if len(think_start_indices) >= 4:
                    random_indices = random.sample(range(1, len(think_start_indices) - 1), 2)
                else:
                    print("WARNING: Do not apply extra, because of a single cot only, Skipping")
                    break
                extra_token_ids = random.sample(self.extra_think_token_ids, 2)
                assert len(extra_token_ids) == 2
                for i, extra_token_id in enumerate(extra_token_ids):
                    idx = think_start_indices[random_indices[i]]
                    assert think_start_tokens[random_indices[i]] in self.think_token_ids[:6]
                    batch["input_ids"][b, idx] = extra_token_id
                    if (think_start_tokens[random_indices[i]] + 1) in batch["input_ids"][b]:
                        think_end_idx = batch["input_ids"][b].tolist().index(think_start_tokens[random_indices[i]] + 1)
                        assert batch["input_ids"][b, think_end_idx] == think_start_tokens[random_indices[i]] + 1
                        batch["input_ids"][b, think_end_idx] = extra_token_id + 1
                        batch["labels"][b, think_end_idx] = extra_token_id + 1
        
        batch["position_ids"] = torch.arange(seq_len, device=batch["input_ids"].device).long().expand((bsz, -1))
        batch["seg_ids"] = torch.zeros((bsz, seq_len), device=batch["input_ids"].device).long()
        
        if self.intra_rope and not self.ablation_study:
            for b in range(bsz):
                think_start_indices = []
                think_start_tokens = []
                for i in range(seq_len):
                    if batch["input_ids"][b, i].item() in self.think_token_ids:
                        think_start_indices.append(i)
                        think_start_tokens.append(batch["input_ids"][b, i].item())
                        
                assert len(think_start_indices) > 1
                
                summary_start_pos = 0
                for idx in range(1, len(think_start_indices)-1):
                    current_think_idx = think_start_indices[idx]
                    next_think_idx = think_start_indices[idx+1]
                    offset = current_think_idx
                    batch["position_ids"][b, current_think_idx:next_think_idx] += (think_start_indices[1] - offset)
                    cot_end_pos = next_think_idx - 1 + think_start_indices[1] - offset
                    if cot_end_pos > summary_start_pos:
                        summary_start_pos = cot_end_pos + 1
                    assert (think_start_tokens[idx] - self.think_token_ids[0]) % 2 == 0
                    batch["seg_ids"][b, current_think_idx:next_think_idx] = (think_start_tokens[idx] - self.think_token_ids[0]) // 2 + 1
                    if idx == (len(think_start_indices)-2):
                        assert batch["input_ids"][b, think_start_indices[idx+1]].item() == 151681
                        offset = next_think_idx
                        batch["position_ids"][b, next_think_idx:] += (summary_start_pos - offset)
                        # Set summary segment id equals to 0
                        batch["seg_ids"][b, next_think_idx:] = 0
        
        if self.pe_ablation:
            del batch["seg_ids"]
        assert "seg_ids" in batch
        
        return batch
    