"""SFT data loader for nmoe training.

Loads instruction-following datasets, applies chat templates, tokenizes,
generates loss masks (0=prompt, 1=response), and returns 3- or 4-tuples
for supervised fine-tuning.

Without packing:
    inputs, targets, loss_mask = loader.next()

With packing (sft_packing_enabled=True):
    inputs, targets, loss_mask, cu_seqlens = loader.next()

Packing uses First-Fit Decreasing (FFD) bin packing to concatenate multiple
SFT examples into a single sequence, with cu_seqlens marking document boundaries
for document-isolated attention via FlashMLA varlen or FlexAttention block_mask.

Supports DeepSeek V3.2 chat format and standard chatml / llama3 formats.

Target: DeepSeek V3.2 REAP-345B (vocab_size=129280)

Usage:
    loader = build_sft_loader(
        cfg=config,
        dp_rank=rank,
        dp_world_size=16,  # DP=16 for EP=8/DP=16
    )
    inputs, targets, loss_mask = loader.next()
"""

from __future__ import annotations

import logging
import os
import random
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Chat template definitions
# ============================================================================

# DeepSeek V3.2 special tokens (from chat_template_deepseekv32_speciale.jinja)
# Uses fullwidth bar U+FF5C: |
DEEPSEEK_V32_TOKENS = {
    "bos": "<|begin_of_sentence|>",       # bos_token_id = 0
    "user": "<\uff5cUser\uff5c>",         # <｜User｜>
    "assistant": "<\uff5cAssistant\uff5c>", # <｜Assistant｜>
    "eos": "<\uff5cend\u2581of\u2581sentence\uff5c>",  # <｜end▁of▁sentence｜>
    "think_open": "<think>",
    "think_close": "</think>",
}


def format_deepseekv32(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Format messages using DeepSeek V3.2 chat template.

    Returns:
        (formatted_text, role_spans): The formatted text and a list of
        (start_char, end_char) tuples marking the assistant response spans.
        These spans are used to generate the loss mask.
    """
    parts = []
    role_spans = []  # (start_char_idx, end_char_idx) for assistant responses

    # Collect system prompt
    system_parts = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
    system_prompt = "\n\n".join(system_parts) if system_parts else ""

    # BOS + system
    parts.append(DEEPSEEK_V32_TOKENS["bos"])
    if system_prompt:
        parts.append(system_prompt)

    last_was_user = False
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            parts.append(DEEPSEEK_V32_TOKENS["user"])
            parts.append(msg["content"])
            last_was_user = True
        elif msg["role"] == "assistant":
            if last_was_user:
                parts.append(DEEPSEEK_V32_TOKENS["assistant"])
                # Add </think> prefix (standard for non-thinking mode)
                parts.append(DEEPSEEK_V32_TOKENS["think_close"])

            # Mark assistant response start
            text_so_far = "".join(parts)
            response_start = len(text_so_far)

            content = msg["content"]
            # Strip think tags if present
            if "</think>" in content:
                content = content.split("</think>", 1)[1]

            parts.append(content)
            parts.append(DEEPSEEK_V32_TOKENS["eos"])

            text_so_far_after = "".join(parts)
            response_end = len(text_so_far_after)
            role_spans.append((response_start, response_end))
            last_was_user = False

    if add_generation_prompt and last_was_user:
        parts.append(DEEPSEEK_V32_TOKENS["assistant"])
        parts.append(DEEPSEEK_V32_TOKENS["think_close"])

    return "".join(parts), role_spans


def format_chatml(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Format messages using ChatML template.

    Returns:
        (formatted_text, role_spans) for assistant responses.
    """
    parts = []
    role_spans = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n")
        if role == "assistant":
            text_so_far = "".join(parts)
            response_start = len(text_so_far)
            parts.append(content)
            parts.append("<|im_end|>\n")
            text_so_far_after = "".join(parts)
            role_spans.append((response_start, len(text_so_far_after)))
        else:
            parts.append(content)
            parts.append("<|im_end|>\n")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts), role_spans


def format_llama3(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Format messages using Llama 3 template.

    Returns:
        (formatted_text, role_spans) for assistant responses.
    """
    parts = []
    role_spans = []

    parts.append("<|begin_of_text|>")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
        if role == "assistant":
            text_so_far = "".join(parts)
            response_start = len(text_so_far)
            parts.append(content)
            parts.append("<|eot_id|>")
            text_so_far_after = "".join(parts)
            role_spans.append((response_start, len(text_so_far_after)))
        else:
            parts.append(content)
            parts.append("<|eot_id|>")

    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(parts), role_spans


CHAT_FORMATTERS = {
    "deepseekv32": format_deepseekv32,
    "chatml": format_chatml,
    "llama3": format_llama3,
}


# ============================================================================
# Tokenization and loss mask
# ============================================================================

def tokenize_with_loss_mask(
    text: str,
    role_spans: List[Tuple[int, int]],
    tokenizer: Any,
    seq_len: int,
    mask_prompt_loss: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Tokenize text and generate loss mask from role spans.

    The loss mask is 1 for tokens in assistant response spans and 0 for
    prompt/system tokens. This ensures the model only learns to predict
    the assistant's responses.

    Args:
        text: Formatted chat text.
        role_spans: List of (start_char, end_char) for assistant responses.
        tokenizer: HuggingFace tokenizer.
        seq_len: Maximum sequence length.
        mask_prompt_loss: If True, mask prompt tokens (loss_mask=0).

    Returns:
        (inputs, targets, loss_mask) tensors of shape [seq_len], or None if
        the example is too short (< 4 tokens in response).
    """
    # Tokenize the full text
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,  # We handle special tokens in the template
        truncation=True,
        max_length=seq_len + 1,  # +1 because we split into input/target
    )

    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]  # List of (char_start, char_end) per token

    if len(token_ids) < 4:
        return None

    # Truncate to seq_len + 1
    token_ids = token_ids[: seq_len + 1]
    offsets = offsets[: seq_len + 1]

    # Build loss mask based on character offsets
    loss_mask_list = []
    for tok_start, tok_end in offsets:
        if not mask_prompt_loss:
            loss_mask_list.append(1)
            continue

        # Check if this token overlaps with any assistant response span
        is_response = False
        for span_start, span_end in role_spans:
            # Token overlaps with response span if they share any characters
            if tok_end > span_start and tok_start < span_end:
                is_response = True
                break
        loss_mask_list.append(1 if is_response else 0)

    # Create tensors
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    mask_tensor = torch.tensor(loss_mask_list, dtype=torch.float32)

    # Split into inputs and targets (shifted by 1)
    inputs = token_tensor[:-1]
    targets = token_tensor[1:]
    loss_mask = mask_tensor[1:]  # Mask applies to prediction targets

    # Drop examples that have too few supervised tokens after truncation.
    # This prevents all-zero masks (no training signal) from entering the
    # train/loss path and matches this function's return contract.
    if mask_prompt_loss and int(loss_mask.sum().item()) < 4:
        return None

    # Pad if needed (right-padding)
    pad_len = seq_len - inputs.shape[0]
    if pad_len > 0:
        inputs = torch.nn.functional.pad(inputs, (0, pad_len), value=0)
        targets = torch.nn.functional.pad(targets, (0, pad_len), value=0)
        loss_mask = torch.nn.functional.pad(loss_mask, (0, pad_len), value=0.0)

    # Ensure no gradient flows from padding
    # Padding tokens get loss_mask=0 automatically (from the pad)

    return inputs, targets, loss_mask


@dataclass
class PackedBin:
    """A packed sequence bin containing multiple SFT examples.

    Stores pre-computed tensors ready for batching: inputs, targets, loss_mask,
    and cu_seqlens marking document boundaries within the packed sequence.
    """
    inputs: torch.Tensor       # [seq_len] long -- concatenated input token IDs
    targets: torch.Tensor      # [seq_len] long -- concatenated target token IDs
    loss_mask: torch.Tensor    # [seq_len] float -- 0 at prompt/boundary, 1 at response
    cu_seqlens: torch.Tensor   # [num_docs + 1] int32 -- cumulative document lengths
    num_real_tokens: int       # Total non-padding tokens in this bin
    num_docs: int              # Number of documents packed into this bin


def pack_examples_ffd(
    examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    seq_len: int,
    pad_token_id: int = 0,
    max_docs_per_bin: int = 16,
) -> List[PackedBin]:
    """Pack tokenized SFT examples into fixed-length bins using First-Fit Decreasing.

    Algorithm:
      1. Sort examples by token length (descending) for optimal packing
      2. For each example, find the first bin with enough remaining capacity
      3. If no bin has capacity, create a new bin
      4. After all examples are assigned, build packed tensors for each bin

    Document boundary handling:
      - Each document's tokens are concatenated directly (no separator tokens needed)
      - cu_seqlens marks where each document starts/ends within the packed sequence
      - loss_mask is set to 0 at the last position of each non-final document to prevent
        the model from learning to predict the first token of the next document given
        the previous document's context (cross-document contamination)

    Args:
        examples: List of (inputs, targets, loss_mask) tuples from tokenize_with_loss_mask.
                  inputs and targets are [length] long tensors (NOT padded to seq_len).
        seq_len: Maximum sequence length (bin capacity).
        pad_token_id: Token ID for padding unused positions.
        max_docs_per_bin: Maximum documents per bin (safety limit).

    Returns:
        List of PackedBin instances, each representing one packed training sequence.
    """
    if not examples:
        return []

    # Compute actual (unpadded) token lengths for each example.
    # Examples from tokenize_with_loss_mask are padded to seq_len, so we need to
    # find the actual length by looking for the last non-zero loss_mask position
    # or by checking where padding starts in the input.
    indexed_examples = []
    for idx, (inp, tgt, mask) in enumerate(examples):
        # Find actual length: last non-pad position + 1.
        # Inputs are right-padded with pad_token_id=0. Find the actual length by
        # looking for the first pad token or using the full length if no padding.
        length = inp.shape[0]
        # Check from the end for padding
        while length > 0 and inp[length - 1].item() == pad_token_id and mask[length - 1].item() == 0.0:
            length -= 1
        if length == 0:
            continue
        indexed_examples.append((idx, length, inp[:length], tgt[:length], mask[:length]))

    if not indexed_examples:
        return []

    # Step 1: Sort by length descending (FFD)
    indexed_examples.sort(key=lambda x: x[1], reverse=True)

    # Step 2-3: First-Fit Decreasing bin packing
    # Each bin tracks: list of (inp_slice, tgt_slice, mask_slice), remaining capacity
    bins: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    bin_remaining: List[int] = []

    for _, length, inp_slice, tgt_slice, mask_slice in indexed_examples:
        if length > seq_len:
            # Truncate to seq_len if somehow longer (should not happen with proper tokenization)
            inp_slice = inp_slice[:seq_len]
            tgt_slice = tgt_slice[:seq_len]
            mask_slice = mask_slice[:seq_len]
            length = seq_len

        placed = False
        for bin_idx in range(len(bins)):
            if bin_remaining[bin_idx] >= length and len(bins[bin_idx]) < max_docs_per_bin:
                bins[bin_idx].append((inp_slice, tgt_slice, mask_slice))
                bin_remaining[bin_idx] -= length
                placed = True
                break

        if not placed:
            bins.append([(inp_slice, tgt_slice, mask_slice)])
            bin_remaining.append(seq_len - length)

    # Step 4: Build packed tensors for each bin
    packed_bins = []
    for bin_docs in bins:
        packed = _build_packed_bin(bin_docs, seq_len, pad_token_id)
        packed_bins.append(packed)

    return packed_bins


def _build_packed_bin(
    docs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    seq_len: int,
    pad_token_id: int,
) -> PackedBin:
    """Build a single PackedBin from a list of document slices.

    Handles:
    - Concatenation of document tokens
    - Construction of cu_seqlens for varlen attention
    - Zeroing loss_mask at document boundaries to prevent cross-document target leakage
    - Right-padding to seq_len

    Args:
        docs: List of (inputs, targets, loss_mask) unpadded tensor slices.
        seq_len: Target sequence length.
        pad_token_id: Padding token ID.

    Returns:
        PackedBin with all tensors sized to seq_len.
    """
    all_inputs = []
    all_targets = []
    all_masks = []
    cu_seqlens = [0]
    offset = 0

    for inp, tgt, mask in docs:
        doc_len = inp.shape[0]
        all_inputs.append(inp)
        all_targets.append(tgt)
        all_masks.append(mask)
        offset += doc_len
        cu_seqlens.append(offset)

    # Concatenate all documents
    inputs_cat = torch.cat(all_inputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)
    num_real_tokens = inputs_cat.shape[0]

    # Zero out loss_mask at document boundaries to prevent cross-document target leakage.
    # At the last position of Doc_i, the target would be the first token of Doc_{i+1},
    # which is an invalid prediction target. Set loss_mask=0 there.
    for boundary_idx in range(len(cu_seqlens) - 2):  # All boundaries except the last
        boundary_pos = cu_seqlens[boundary_idx + 1] - 1  # Last position of this document
        if 0 <= boundary_pos < num_real_tokens:
            masks_cat[boundary_pos] = 0.0

    # Right-pad to seq_len
    pad_len = seq_len - num_real_tokens
    if pad_len > 0:
        inputs_cat = torch.nn.functional.pad(inputs_cat, (0, pad_len), value=pad_token_id)
        targets_cat = torch.nn.functional.pad(targets_cat, (0, pad_len), value=pad_token_id)
        masks_cat = torch.nn.functional.pad(masks_cat, (0, pad_len), value=0.0)
    elif pad_len < 0:
        # Truncate (should not happen with correct packing, but safety)
        inputs_cat = inputs_cat[:seq_len]
        targets_cat = targets_cat[:seq_len]
        masks_cat = masks_cat[:seq_len]
        # Adjust cu_seqlens
        cu_seqlens_adjusted = [0]
        for cs in cu_seqlens[1:]:
            if cs <= seq_len:
                cu_seqlens_adjusted.append(cs)
            else:
                cu_seqlens_adjusted.append(seq_len)
                break
        cu_seqlens = cu_seqlens_adjusted
        num_real_tokens = seq_len

    # Add final boundary at seq_len if the last cu_seqlen doesn't reach it.
    # The padding region gets its own "document" in cu_seqlens so the varlen kernel
    # processes it without attending across the last real document.
    if cu_seqlens[-1] < seq_len:
        cu_seqlens.append(seq_len)

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

    return PackedBin(
        inputs=inputs_cat,
        targets=targets_cat,
        loss_mask=masks_cat,
        cu_seqlens=cu_seqlens_tensor,
        num_real_tokens=num_real_tokens,
        num_docs=len(docs),
    )


def compute_packing_stats(bins: List[PackedBin], seq_len: int) -> Dict[str, float]:
    """Compute packing efficiency statistics.

    Args:
        bins: List of packed bins.
        seq_len: Sequence length.

    Returns:
        Dictionary with packing statistics.
    """
    if not bins:
        return {"efficiency": 0.0, "avg_docs_per_bin": 0.0, "total_bins": 0}

    total_real = sum(b.num_real_tokens for b in bins)
    total_capacity = len(bins) * seq_len
    avg_docs = sum(b.num_docs for b in bins) / len(bins)

    return {
        "efficiency": total_real / total_capacity if total_capacity > 0 else 0.0,
        "avg_docs_per_bin": avg_docs,
        "total_bins": len(bins),
        "total_real_tokens": total_real,
        "total_capacity": total_capacity,
        "waste_fraction": 1.0 - (total_real / total_capacity) if total_capacity > 0 else 1.0,
    }


# ============================================================================
# Dataset adapters
# ============================================================================

def extract_messages_from_example(
    example: Dict[str, Any],
) -> Optional[List[Dict[str, str]]]:
    """Extract chat messages from various dataset formats.

    Supports:
    - "messages" field (standard chat format: [{role, content}, ...])
    - "conversations" field (ShareGPT format: [{from, value}, ...])
    - "instruction" + "output" fields (alpaca format)
    - "prompt" + "completion" fields

    Returns:
        List of {role, content} dicts, or None if format not recognized.
    """
    # Standard messages format (Nemotron-Agentic-v1, OpenAI format)
    if "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            # Ensure proper format
            normalized = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    normalized.append({
                        "role": msg["role"],
                        "content": str(msg["content"]),
                    })
            if normalized:
                return normalized

    # ShareGPT format
    if "conversations" in example:
        convos = example["conversations"]
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = []
        for turn in convos:
            role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
            content = turn.get("value", "")
            messages.append({"role": role, "content": str(content)})
        if messages:
            return messages

    # Alpaca format
    if "instruction" in example and "output" in example:
        messages = []
        if example.get("input"):
            messages.append({
                "role": "user",
                "content": f"{example['instruction']}\n\n{example['input']}",
            })
        else:
            messages.append({"role": "user", "content": str(example["instruction"])})
        messages.append({"role": "assistant", "content": str(example["output"])})
        return messages

    # Prompt-completion format
    if "prompt" in example and "completion" in example:
        return [
            {"role": "user", "content": str(example["prompt"])},
            {"role": "assistant", "content": str(example["completion"])},
        ]

    return None


# ============================================================================
# SFT Loader
# ============================================================================

class SFTLoader:
    """SFT data loader that returns (inputs, targets, loss_mask) 3-tuples,
    or (inputs, targets, loss_mask, cu_seqlens) 4-tuples when packing is enabled.

    Loads data from a HuggingFace dataset, applies chat formatting,
    tokenizes, and generates per-token loss masks.

    When packing is enabled (sft_packing_enabled=True):
    - Multiple SFT examples are packed into each sequence using FFD bin packing
    - cu_seqlens marks document boundaries for document-isolated attention
    - Loss mask is zeroed at document boundaries to prevent cross-document leakage
    - Packed bins are precomputed at init time for deterministic checkpoint/resume

    For EP=8/DP=16: data is sharded across dp_world_size=16 replicas.
    Each DP rank sees 1/16 of the data.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        tokenizer_path: str,
        seq_len: int,
        batch_size: int,
        dp_rank: int,
        dp_world_size: int,
        gradient_accumulation_steps: int = 1,
        prompt_format: str = "deepseekv32",
        mask_prompt_loss: bool = True,
        device: str = "cuda",
        seed: int = 42,
        max_examples: Optional[int] = None,
        dataset_split: str = "train",
        packing_enabled: bool = False,
        packing_max_docs_per_bin: int = 16,
    ):
        """Initialize SFT loader.

        Args:
            dataset_path: HuggingFace dataset path or local directory.
            tokenizer_path: Path to tokenizer (HF model name or local dir).
            seq_len: Maximum sequence length.
            batch_size: Number of sequences per batch (global, divided by dp_world_size).
            dp_rank: This rank's DP index (0 to dp_world_size-1).
            dp_world_size: Data parallelism group size (NOT total world size).
            gradient_accumulation_steps: Number of micro-batches per optimizer step.
            prompt_format: Chat template format ('deepseekv32', 'chatml', 'llama3').
            mask_prompt_loss: If True, set loss_mask=0 for prompt tokens.
            device: Target device for tensors.
            seed: Random seed for shuffling.
            max_examples: Limit number of examples (for debugging).
            dataset_split: Dataset split to load.
            packing_enabled: If True, pack multiple examples per sequence.
            packing_max_docs_per_bin: Maximum documents per packed sequence.
        """
        self.seq_len = seq_len
        self.global_batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # Per-DP-rank micro-batch size: batch_size / dp_world_size / accum_steps
        # The training loop calls loader.next() once per micro-step.
        self.local_batch_size = batch_size // (dp_world_size * gradient_accumulation_steps)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.mask_prompt_loss = mask_prompt_loss
        self.device = device
        self.packing_enabled = packing_enabled
        self.packing_max_docs_per_bin = packing_max_docs_per_bin

        if prompt_format not in CHAT_FORMATTERS:
            raise ValueError(
                f"Unknown prompt_format: {prompt_format}. "
                f"Expected one of: {list(CHAT_FORMATTERS.keys())}"
            )
        self.formatter = CHAT_FORMATTERS[prompt_format]

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        from transformers import AutoTokenizer
        import warnings
        with warnings.catch_warnings():
            # Suppress false positive Mistral regex warning from transformers 5.1.0
            # (off-by-one version gate: > 4.57.3 should be >= 4.57.3, DeepSeek V3.2
            # ships transformers_version=4.57.3 which falls through both checks)
            warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path} (split={dataset_split})")
        self._load_dataset(dataset_path, dataset_split, max_examples)

        # Shuffle and shard for DP
        self._setup_sharding(seed)

        # Internal state
        self._index = 0
        self._epoch = 0
        self._step = 0
        self._total_tokens = 0

        # Packing: precompute packed bins if enabled
        self._packed_bins: Optional[List[PackedBin]] = None
        self._bin_index = 0
        if self.packing_enabled:
            self._build_packed_bins()

    def _load_dataset(
        self,
        dataset_path: str,
        split: str,
        max_examples: Optional[int],
    ) -> None:
        """Load dataset from HuggingFace or local files."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required: pip install datasets"
            )

        # Columns the SFT loader actually needs (chat messages).
        # We request only these to avoid schema-cast errors from complex
        # nested columns (tool schemas, function defs, etc.) that some
        # agentic datasets include.
        _needed_cols = {"messages", "conversations", "instruction", "output",
                        "prompt", "completion"}

        def _apply_local_split_slice(dataset, split_expr: str, split_base: str):
            """Apply train[...]-style slicing for local files loaded via Dataset.from_*."""
            split_expr = str(split_expr)
            if split_expr == split_base:
                return dataset

            match = re.fullmatch(rf"{re.escape(split_base)}\[(.*)\]", split_expr)
            if match is None:
                raise ValueError(
                    f"Unsupported local dataset split expression: {split_expr!r}. "
                    f"Use '{split_base}' or '{split_base}[start:end]' (int or %)."
                )

            body = match.group(1)
            if ":" not in body:
                raise ValueError(
                    f"Unsupported local dataset split expression: {split_expr!r}. "
                    "Only range slicing is supported (start:end)."
                )

            start_tok, end_tok = body.split(":", 1)
            total = len(dataset)

            def _parse_index(tok: str, *, is_start: bool) -> int:
                token = tok.strip()
                if not token:
                    return 0 if is_start else total
                if token.endswith("%"):
                    pct = float(token[:-1])
                    return int((pct / 100.0) * total)
                idx = int(token)
                if idx < 0:
                    idx += total
                return idx

            start = max(0, min(total, _parse_index(start_tok, is_start=True)))
            end = max(0, min(total, _parse_index(end_tok, is_start=False)))
            if end < start:
                end = start
            return dataset.select(range(start, end))

        if os.path.isdir(dataset_path):
            # Local directory: enumerate files explicitly to avoid
            # datasets/fsspec glob semantics drift across versions.
            from datasets import Dataset as HFDataset

            root = Path(dataset_path)
            json_files = sorted(str(p) for p in root.rglob("*.json") if p.is_file())
            jsonl_files = sorted(str(p) for p in root.rglob("*.jsonl") if p.is_file())
            parquet_files = sorted(str(p) for p in root.rglob("*.parquet") if p.is_file())

            all_json_files = json_files + jsonl_files
            if not all_json_files and not parquet_files:
                raise FileNotFoundError(
                    f"No JSON/JSONL/Parquet files found under local dataset path: {dataset_path}"
                )
            if all_json_files and parquet_files:
                raise ValueError(
                    "Local SFT dataset directory mixes JSON/JSONL and Parquet files. "
                    "Use a single format per dataset path."
                )

            records: list[dict[str, Any]] = []
            if all_json_files:
                for file_path in all_json_files:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Detect JSON array vs JSONL by peeking first non-whitespace byte.
                        first_non_ws = ""
                        while True:
                            ch = f.read(1)
                            if ch == "":
                                break
                            if not ch.isspace():
                                first_non_ws = ch
                                break
                        f.seek(0)
                        if first_non_ws == "[":
                            payload = json.load(f)
                            if not isinstance(payload, list):
                                raise ValueError(
                                    f"Expected JSON array in {file_path}, got {type(payload).__name__}"
                                )
                            for row in payload:
                                if isinstance(row, dict):
                                    keep = {k: v for k, v in row.items() if k in _needed_cols}
                                    if keep:
                                        records.append(keep)
                        else:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                row = json.loads(line)
                                if isinstance(row, dict):
                                    keep = {k: v for k, v in row.items() if k in _needed_cols}
                                    if keep:
                                        records.append(keep)
            else:
                import pyarrow.parquet as pq

                for file_path in parquet_files:
                    table = pq.read_table(file_path)
                    for row in table.to_pylist():
                        if isinstance(row, dict):
                            keep = {k: v for k, v in row.items() if k in _needed_cols}
                            if keep:
                                records.append(keep)

            if not records:
                raise ValueError(
                    f"No records loaded from local dataset path: {dataset_path}"
                )

            split_name = str(split).split("[", 1)[0] or "train"
            dataset = HFDataset.from_list(records, split=split_name)
            dataset = _apply_local_split_slice(dataset, str(split), split_name)
        else:
            # HuggingFace Hub dataset — discover available columns first,
            # then load only the ones we actually need to avoid Arrow
            # schema-cast errors on deeply-nested struct columns.
            from huggingface_hub import HfApi
            try:
                info = HfApi().dataset_info(dataset_path)
                if info.card_data and hasattr(info.card_data, "dataset_info"):
                    # Some datasets expose column names in metadata
                    pass
            except Exception:
                pass

            # Try loading with only needed columns (datasets ≥ 3.x)
            _load_kwargs: dict = dict(split=split)
            errors = []
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # Attempt 1: standard load with select columns
                        dataset = load_dataset(dataset_path, **_load_kwargs)
                        break
                    elif attempt == 1:
                        # Attempt 2: streaming → materialise only needed cols
                        logger.warning(
                            "Standard load failed; retrying with streaming "
                            "to work around nested-schema issues"
                        )
                        ds_iter = load_dataset(
                            dataset_path, streaming=True, **_load_kwargs,
                        )
                        rows = []
                        for row in ds_iter:
                            keep = {k: v for k, v in row.items()
                                    if k in _needed_cols}
                            if keep:
                                rows.append(keep)
                        from datasets import Dataset as _Dataset
                        dataset = _Dataset.from_list(rows)
                        break
                    else:
                        # Attempt 3: download parquet files directly and
                        # read only needed columns with pyarrow
                        logger.warning(
                            "Streaming also failed; downloading raw parquet"
                        )
                        import pyarrow.parquet as pq
                        from huggingface_hub import hf_hub_download, list_repo_files
                        parquet_files = [
                            f for f in list_repo_files(dataset_path, repo_type="dataset")
                            if f.endswith(".parquet")
                        ]
                        if not parquet_files:
                            raise RuntimeError(
                                f"No parquet files found in {dataset_path}"
                            )
                        tables = []
                        for pf in parquet_files:
                            local = hf_hub_download(
                                dataset_path, pf, repo_type="dataset",
                            )
                            t = pq.read_table(
                                local,
                                columns=[c for c in _needed_cols
                                         if c in pq.read_schema(local).names],
                            )
                            tables.append(t)
                        import pyarrow as pa
                        from datasets import Dataset as _Dataset
                        merged = pa.concat_tables(tables)
                        dataset = _Dataset(merged)
                        break
                except Exception as e:
                    errors.append(str(e))
                    if attempt == 2:
                        raise RuntimeError(
                            f"Failed to load dataset {dataset_path} after "
                            f"3 attempts:\n" + "\n".join(errors)
                        ) from e

        # Drop columns we don't need (reduces memory and avoids
        # downstream serialization issues with complex Arrow types).
        drop = [c for c in dataset.column_names if c not in _needed_cols]
        if drop:
            dataset = dataset.remove_columns(drop)

        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        self.dataset = dataset
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")

    def _setup_sharding(self, seed: int) -> None:
        """Set up DP-aware data sharding."""
        n = len(self.dataset)
        # Create a shuffled index
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)

        # Shard across DP ranks (interleaved)
        self._indices = indices[self.dp_rank :: self.dp_world_size]
        logger.info(
            f"DP rank {self.dp_rank}/{self.dp_world_size}: "
            f"{len(self._indices)} examples (of {n} total)"
        )

    def _build_packed_bins(self) -> None:
        """Precompute packed bins from all examples for deterministic training.

        Tokenizes all examples assigned to this DP rank, then packs them into
        fixed-length bins using FFD. The packed bins replace the raw example
        indices as the iteration source.

        This is done once at init time (or after checkpoint load triggers re-sharding).
        Runtime cost is amortized over the entire training run.
        """
        t0 = time.time()
        logger.info(
            f"[Packing] Tokenizing {len(self._indices)} examples for DP rank {self.dp_rank}..."
        )

        # Tokenize all examples
        all_examples = []
        skipped = 0
        for idx in range(len(self._indices)):
            result = self._process_example(idx)
            if result is None:
                skipped += 1
                continue
            all_examples.append(result)

        logger.info(
            f"[Packing] Tokenized {len(all_examples)} examples "
            f"(skipped {skipped}) in {time.time() - t0:.1f}s"
        )

        if not all_examples:
            self._packed_bins = []
            return

        # Pack into bins using FFD
        t1 = time.time()
        self._packed_bins = pack_examples_ffd(
            examples=all_examples,
            seq_len=self.seq_len,
            pad_token_id=0,
            max_docs_per_bin=self.packing_max_docs_per_bin,
        )

        # Log packing statistics
        stats = compute_packing_stats(self._packed_bins, self.seq_len)
        pack_time = time.time() - t1
        total_time = time.time() - t0
        logger.info(
            f"[Packing] DP rank {self.dp_rank}: "
            f"{len(all_examples)} examples -> {stats['total_bins']} packed bins | "
            f"efficiency={stats['efficiency']:.1%} | "
            f"avg_docs/bin={stats['avg_docs_per_bin']:.1f} | "
            f"waste={stats['waste_fraction']:.1%} | "
            f"pack_time={pack_time:.1f}s total_time={total_time:.1f}s"
        )

    def _process_example(self, idx: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Process a single example: format -> tokenize -> loss mask."""
        example = self.dataset[self._indices[idx]]
        messages = extract_messages_from_example(example)
        if messages is None:
            return None

        # Check that there's at least one assistant response
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not has_assistant:
            return None

        # Format with chat template
        text, role_spans = self.formatter(messages)

        # Tokenize and generate loss mask
        result = tokenize_with_loss_mask(
            text=text,
            role_spans=role_spans,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            mask_prompt_loss=self.mask_prompt_loss,
        )

        return result

    def next(self):
        """Get next batch.

        Without packing:
            Returns (inputs, targets, loss_mask) 3-tuple.

        With packing:
            Returns (inputs, targets, loss_mask, cu_seqlens) 4-tuple.
            cu_seqlens is a list of int32 tensors, one per batch element,
            each of shape [num_docs_i + 1] marking document boundaries.

        Shapes:
            inputs: [local_batch_size, seq_len] long tensor
            targets: [local_batch_size, seq_len] long tensor
            loss_mask: [local_batch_size, seq_len] float tensor
            cu_seqlens: list of [num_docs_i + 1] int32 tensors (packing only)
        """
        if self.packing_enabled and self._packed_bins is not None:
            return self._next_packed()
        return self._next_unpacked()

    def _next_unpacked(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original unpacked batch loading."""
        batch_inputs = []
        batch_targets = []
        batch_masks = []

        while len(batch_inputs) < self.local_batch_size:
            if self._index >= len(self._indices):
                # Wrap around (new epoch)
                self._epoch += 1
                self._index = 0
                logger.info(f"SFT loader: starting epoch {self._epoch}")

            result = self._process_example(self._index)
            self._index += 1

            if result is None:
                continue

            inputs, targets, loss_mask = result
            batch_inputs.append(inputs)
            batch_targets.append(targets)
            batch_masks.append(loss_mask)

        self._step += 1
        self._total_tokens += self.local_batch_size * self.seq_len

        inputs = torch.stack(batch_inputs).to(self.device, non_blocking=True)
        targets = torch.stack(batch_targets).to(self.device, non_blocking=True)
        loss_mask = torch.stack(batch_masks).to(self.device, non_blocking=True)

        return inputs, targets, loss_mask

    def _next_packed(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Packed batch loading with cu_seqlens for document-isolated attention."""
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        batch_cu_seqlens = []

        while len(batch_inputs) < self.local_batch_size:
            if self._bin_index >= len(self._packed_bins):
                # Wrap around (new epoch)
                self._epoch += 1
                self._bin_index = 0
                logger.info(f"SFT loader (packed): starting epoch {self._epoch}")

            pbin = self._packed_bins[self._bin_index]
            self._bin_index += 1

            batch_inputs.append(pbin.inputs)
            batch_targets.append(pbin.targets)
            batch_masks.append(pbin.loss_mask)
            batch_cu_seqlens.append(pbin.cu_seqlens)

        self._step += 1
        self._total_tokens += self.local_batch_size * self.seq_len

        inputs = torch.stack(batch_inputs).to(self.device, non_blocking=True)
        targets = torch.stack(batch_targets).to(self.device, non_blocking=True)
        loss_mask = torch.stack(batch_masks).to(self.device, non_blocking=True)
        # cu_seqlens are moved to device individually (variable length per batch element)
        cu_seqlens = [cs.to(self.device, non_blocking=True) for cs in batch_cu_seqlens]

        return inputs, targets, loss_mask, cu_seqlens

    def state_dict(self) -> Dict[str, Any]:
        """Save loader state for checkpointing."""
        state = {
            "index": self._index,
            "epoch": self._epoch,
            "step": self._step,
            "total_tokens": self._total_tokens,
            "packing_enabled": self.packing_enabled,
        }
        if self.packing_enabled:
            state["bin_index"] = self._bin_index
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore loader state from checkpoint."""
        self._index = state.get("index", 0)
        self._epoch = state.get("epoch", 0)
        self._step = state.get("step", 0)
        self._total_tokens = state.get("total_tokens", 0)
        if self.packing_enabled:
            self._bin_index = state.get("bin_index", 0)


# ============================================================================
# Builder function (matches build_loader interface)
# ============================================================================

def build_sft_loader(
    cfg,
    dp_rank: int,
    dp_world_size: int,
    print_fn=print,
) -> SFTLoader:
    """Build SFT data loader from config.

    This is the main entry point for SFT data loading. Called from train.py
    when cfg.sft_enabled is True.

    CRITICAL: Pass dp_world_size (e.g., 16), NOT total world_size (e.g., 128).
    With EP=8/DP=16, each DP replica sees 1/16 of the data, not 1/128.

    Args:
        cfg: Config with SFT settings (sft_data_path, sft_prompt_format, etc.)
        dp_rank: This rank's DP index (0 to dp_world_size-1).
        dp_world_size: Data parallelism group size.
        print_fn: Logging function.

    Returns:
        SFTLoader instance.
    """
    sft_data_path = getattr(cfg, "sft_data_path", None)
    if not sft_data_path:
        raise ValueError("sft_data_path is required when sft_enabled=True")

    # Determine tokenizer path
    # For DeepSeek V3.2, use the local model directory (NOT "deepseek-ai/DeepSeek-V3"
    # which is the V3 tokenizer with vocab_size=163840, wrong for V3.2 vocab_size=129280).
    tokenizer_path = getattr(cfg, "tokenizer_path", None)
    if tokenizer_path is None:
        if getattr(cfg, "sft_prompt_format", "chatml") == "deepseekv32":
            # Search for local V3.2 tokenizer (LlamaTokenizerFast, vocab=129280)
            import os
            _v32_candidates = [
                "/home/nourdine/DeepSeek-V3.2-REAP-345B-NVFP4A16-v2",
                "/home/nourdine/DeepSeek-V3.2-REAP-345B-BF16",
            ]
            tokenizer_path = None
            for cand in _v32_candidates:
                if os.path.isfile(os.path.join(cand, "tokenizer.json")):
                    tokenizer_path = cand
                    break
            if tokenizer_path is None:
                raise ValueError(
                    "DeepSeek V3.2 tokenizer not found. Set tokenizer_path in config "
                    "to a directory containing tokenizer.json from DeepSeek-V3.2. "
                    f"Searched: {_v32_candidates}"
                )
        else:
            tokenizer_path = getattr(cfg, "tokenizer", "o200k_harmony")

    prompt_format = getattr(cfg, "sft_prompt_format", "chatml")
    mask_prompt_loss = getattr(cfg, "sft_mask_prompt_loss", True)
    packing_enabled = getattr(cfg, "sft_packing_enabled", False)
    packing_max_docs = getattr(cfg, "sft_packing_max_docs_per_bin", 16)
    accum_steps = int(getattr(cfg, "gradient_accumulation_steps", 1))

    # Compute micro-batch size for logging
    micro_batch_size = cfg.batch_size // (dp_world_size * accum_steps)

    if dp_rank == 0:
        print_fn(f"[SFT] dataset: {sft_data_path}")
        print_fn(f"[SFT] tokenizer: {tokenizer_path}")
        print_fn(f"[SFT] format: {prompt_format}")
        print_fn(f"[SFT] mask_prompt_loss: {mask_prompt_loss}")
        print_fn(f"[SFT] packing: {'ENABLED' if packing_enabled else 'disabled'}")
        if packing_enabled:
            print_fn(f"[SFT] packing_max_docs_per_bin: {packing_max_docs}")
        print_fn(f"[SFT] dp_world_size: {dp_world_size} (each rank sees 1/{dp_world_size} of data)")
        print_fn(f"[SFT] gradient_accumulation_steps: {accum_steps}")
        print_fn(f"[SFT] micro_batch_size: {micro_batch_size} (T per fwd = {micro_batch_size * cfg.seq_len})")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SFT data loader")
    loader = SFTLoader(
        dataset_path=sft_data_path,
        tokenizer_path=tokenizer_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        gradient_accumulation_steps=accum_steps,
        prompt_format=prompt_format,
        mask_prompt_loss=mask_prompt_loss,
        device="cuda" if torch.cuda.is_available() else "cpu",
        packing_enabled=packing_enabled,
        packing_max_docs_per_bin=packing_max_docs,
    )

    return loader
