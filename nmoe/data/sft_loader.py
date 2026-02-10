"""SFT data loader for nmoe training.

Loads instruction-following datasets, applies chat templates, tokenizes,
generates loss masks (0=prompt, 1=response), and returns 3-tuples
(inputs, targets, loss_mask) for supervised fine-tuning.

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

    # Pad if needed (right-padding)
    pad_len = seq_len - inputs.shape[0]
    if pad_len > 0:
        inputs = torch.nn.functional.pad(inputs, (0, pad_len), value=0)
        targets = torch.nn.functional.pad(targets, (0, pad_len), value=0)
        loss_mask = torch.nn.functional.pad(loss_mask, (0, pad_len), value=0.0)

    # Ensure no gradient flows from padding
    # Padding tokens get loss_mask=0 automatically (from the pad)

    return inputs, targets, loss_mask


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
    """SFT data loader that returns (inputs, targets, loss_mask) 3-tuples.

    Loads data from a HuggingFace dataset, applies chat formatting,
    tokenizes, and generates per-token loss masks.

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
        prompt_format: str = "deepseekv32",
        mask_prompt_loss: bool = True,
        device: str = "cuda",
        seed: int = 42,
        max_examples: Optional[int] = None,
        dataset_split: str = "train",
    ):
        """Initialize SFT loader.

        Args:
            dataset_path: HuggingFace dataset path or local directory.
            tokenizer_path: Path to tokenizer (HF model name or local dir).
            seq_len: Maximum sequence length.
            batch_size: Number of sequences per batch (global, divided by dp_world_size).
            dp_rank: This rank's DP index (0 to dp_world_size-1).
            dp_world_size: Data parallelism group size (NOT total world size).
            prompt_format: Chat template format ('deepseekv32', 'chatml', 'llama3').
            mask_prompt_loss: If True, set loss_mask=0 for prompt tokens.
            device: Target device for tensors.
            seed: Random seed for shuffling.
            max_examples: Limit number of examples (for debugging).
            dataset_split: Dataset split to load.
        """
        self.seq_len = seq_len
        self.global_batch_size = batch_size
        self.local_batch_size = batch_size // dp_world_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.mask_prompt_loss = mask_prompt_loss
        self.device = device

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

        if os.path.isdir(dataset_path):
            # Local directory with JSON/JSONL/Parquet files
            dataset = load_dataset(
                "json",
                data_dir=dataset_path,
                split=split,
            )
        else:
            # HuggingFace Hub dataset
            dataset = load_dataset(
                dataset_path,
                split=split,
                trust_remote_code=True,
            )

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

    def next(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get next batch of (inputs, targets, loss_mask).

        Returns:
            inputs: [local_batch_size, seq_len] long tensor
            targets: [local_batch_size, seq_len] long tensor
            loss_mask: [local_batch_size, seq_len] float tensor
        """
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

    def state_dict(self) -> Dict[str, Any]:
        """Save loader state for checkpointing."""
        return {
            "index": self._index,
            "epoch": self._epoch,
            "step": self._step,
            "total_tokens": self._total_tokens,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore loader state from checkpoint."""
        self._index = state.get("index", 0)
        self._epoch = state.get("epoch", 0)
        self._step = state.get("step", 0)
        self._total_tokens = state.get("total_tokens", 0)


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

    if dp_rank == 0:
        print_fn(f"[SFT] dataset: {sft_data_path}")
        print_fn(f"[SFT] tokenizer: {tokenizer_path}")
        print_fn(f"[SFT] format: {prompt_format}")
        print_fn(f"[SFT] mask_prompt_loss: {mask_prompt_loss}")
        print_fn(f"[SFT] dp_world_size: {dp_world_size} (each rank sees 1/{dp_world_size} of data)")

    loader = SFTLoader(
        dataset_path=sft_data_path,
        tokenizer_path=tokenizer_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        prompt_format=prompt_format,
        mask_prompt_loss=mask_prompt_loss,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return loader
