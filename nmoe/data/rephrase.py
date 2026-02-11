"""
K2-style knowledge rephrasing with style diversity and chunk-wise processing.

Key features:
- Style-diverse prompts (formal, casual, technical, simplified)
- Chunk-wise processing for long documents (>2k tokens)
- Batched generation via BatchedGenerator
- Fidelity verification support

Usage (batched):
    from nmoe.data.model import BatchedGenerator
    from nmoe.data.rephrase import rephrase_batch

    gen = BatchedGenerator("/checkpoints/gpt-oss-20b", max_batch=32)
    enc = get_encoding()

    # Generate 10 diverse rephrasings per document
    results = rephrase_batch(
        gen, enc, texts,
        num_versions=10,
        chunk_size=2048,
        use_style_diversity=True
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, Role,
    StreamableParser, load_harmony_encoding,
    SystemContent, DeveloperContent, ReasoningEffort,
)

_encoding = None

# K2-style style-diverse prompts for linguistic variation
STYLE_PROMPTS = {
    "formal": "Rewrite this text in formal academic style while preserving all factual information and key concepts.",
    "casual": "Rephrase this content in clear, conversational language while keeping all important facts and ideas.",
    "technical": "Rewrite this using precise technical terminology while maintaining complete accuracy and detail.",
    "simplified": "Rephrase this in simpler, more accessible language suitable for general audiences, preserving all essential information.",
}

DEFAULT_INSTRUCTION = "Rewrite this text in different words while preserving all information."


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding


def get_encoding():
    """Public access to Harmony encoding."""
    return _get_encoding()


def format_prompt(text: str, instruction: str = DEFAULT_INSTRUCTION, reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH) -> list[int]:
    """Format text as Harmony chat prompt for rephrasing with extended thinking.

    Args:
        text: Text to rephrase
        instruction: Developer instructions for rephrasing style
        reasoning_effort: Reasoning budget (HIGH for quality rephrasing)

    Returns:
        Tokenized prompt with system + developer + user messages
    """
    enc = _get_encoding()

    # System message with HIGH reasoning effort for quality rephrasing
    system_content = SystemContent.new().with_reasoning_effort(reasoning_effort)
    system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)

    # Developer message with rephrasing instructions
    developer_content = DeveloperContent.new().with_instructions(instruction)
    developer_msg = Message.from_role_and_content(Role.DEVELOPER, developer_content)

    # User message with text to rephrase
    user_msg = Message.from_role_and_content(Role.USER, text)

    conv = Conversation.from_messages([system_msg, developer_msg, user_msg])
    return list(enc.render_conversation_for_completion(conv, Role.ASSISTANT))


def create_parser() -> StreamableParser:
    """Create a StreamableParser for streaming token processing."""
    return StreamableParser(_get_encoding(), role=Role.ASSISTANT)


def stop_tokens() -> set[int]:
    """Get Harmony stop tokens for generation."""
    return set(_get_encoding().stop_tokens_for_assistant_actions())


def sample_top_p(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9) -> int:
    """Sample from logits using top-p (nucleus) sampling."""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices[idx].item()


@dataclass
class _SeqState:
    """Track parser state for one sequence."""
    parser: StreamableParser
    final_buf: str = ""  # Accumulated content from "final" channel


def chunk_text(text: str, enc, chunk_size: int = 2048, overlap: int = 256) -> List[str]:
    """Chunk long text into overlapping segments for coherent rephrasing.

    K2 methodology: Segment documents, individually rephrase chunks, then reassemble
    to preserve global coherence and avoid information loss in long documents.

    Args:
        text: Input text to chunk
        enc: Harmony encoding for tokenization
        chunk_size: Max tokens per chunk (default: 2048)
        overlap: Overlap tokens between chunks for context (default: 256)

    Returns:
        List of text chunks
    """
    tokens = list(enc.encode(text))

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward with overlap for context
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def rephrase_batch(
    gen,  # BatchedGenerator
    enc,  # Harmony encoding
    texts: List[str],
    *,
    num_versions: int = 10,
    chunk_size: int = 2048,
    use_style_diversity: bool = True,
    max_new: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> List[List[str]]:
    """K2-style batched rephrasing with style diversity and chunking.

    Args:
        gen: BatchedGenerator instance
        enc: Harmony encoding
        texts: Input texts to rephrase
        num_versions: Number of rephrased versions per text (default: 10 for K2)
        chunk_size: Max tokens per chunk for long documents (default: 2048)
        use_style_diversity: Use style-diverse prompts for variation (default: True)
        max_new: Max tokens to generate per chunk
        temperature: Sampling temperature (0.8 for diversity)
        top_p: Nucleus sampling threshold

    Returns:
        List of lists: results[i] contains num_versions rephrased versions of texts[i]
    """
    stops = stop_tokens()
    style_list = list(STYLE_PROMPTS.keys()) if use_style_diversity else [None]

    # Track which text/version/chunk each request corresponds to
    active: Dict[int, Tuple[int, int, int, _SeqState]] = {}  # sid -> (text_idx, version_idx, chunk_idx, state)
    results: List[List[List[str]]] = [[[]] for _ in texts]  # results[text][version][chunk]

    # Submit all requests (text × version × chunk)
    for text_idx, text in enumerate(texts):
        chunks = chunk_text(text, enc, chunk_size)
        # Keep chunk ordering stable even with batched/async completion.
        # We fill by chunk_idx and join in order at the end.
        results[text_idx] = [["" for _ in range(len(chunks))] for _ in range(num_versions)]

        for version_idx in range(num_versions):
            # Select style prompt (rotate through styles)
            if use_style_diversity:
                style_key = style_list[version_idx % len(style_list)]
                instruction = STYLE_PROMPTS[style_key]
            else:
                instruction = DEFAULT_INSTRUCTION

            for chunk_idx, chunk in enumerate(chunks):
                prompt_toks = format_prompt(chunk, instruction)
                sid = gen.add(prompt_toks, max_tokens=max_new)
                active[sid] = (text_idx, version_idx, chunk_idx, _SeqState(parser=create_parser()))

    def _alive_seq_ids() -> set[int]:
        """Return seq_ids that still exist in the generator.

        BatchedGenerator schedules either newly-admitted prefill sequences *or*
        currently running decode sequences. Therefore, a sid not appearing in a
        given `gen.step()` output does not imply it finished.
        """
        alive: set[int] = set()
        running = getattr(gen, "running", None)
        if running:
            alive.update(int(s.seq_id) for s in running)
        waiting = getattr(gen, "waiting", None)
        if waiting:
            alive.update(int(s.seq_id) for s in waiting)
        return alive

    def _save_and_forget(sid: int) -> None:
        text_idx, version_idx, chunk_idx, st = active[sid]
        rephrased_chunk = st.final_buf.strip()
        if rephrased_chunk:
            results[text_idx][version_idx][chunk_idx] = rephrased_chunk
        del active[sid]

    # Decode loop following reference implementation pattern
    while active:
        out = gen.step()
        if out is None:
            # Generator is idle: flush anything remaining. This mirrors the
            # reference pattern which finalizes after generation ends, even if
            # no explicit stop token was emitted.
            for sid in list(active.keys()):
                _save_and_forget(sid)
            break

        logits, sids = out

        # Process all returned sequences
        for i, sid in enumerate(sids):
            if sid not in active:
                continue

            text_idx, version_idx, chunk_idx, st = active[sid]

            # Sample token
            tok = sample_top_p(logits[i], temperature=temperature, top_p=top_p)

            # Process token through parser
            try:
                st.parser.process(tok)
            except Exception:
                pass  # Parser errors are non-fatal

            # Accumulate content from "final" channel (reference implementation pattern)
            if st.parser.current_channel == "final" and st.parser.last_content_delta:
                st.final_buf += st.parser.last_content_delta

            # Check for stop token
            is_stop = tok in stops

            # Update generator
            gen.update(sid, tok, finished=is_stop)

            if is_stop:
                _save_and_forget(sid)

        # Detect sequences that no longer exist in the generator (e.g., max_tokens reached).
        alive = _alive_seq_ids()
        finished_sids = [sid for sid in list(active.keys()) if sid not in alive]
        for sid in finished_sids:
            _save_and_forget(sid)

    # Reassemble chunks into complete rephrased documents
    final_results: List[List[str]] = []
    for text_idx in range(len(texts)):
        text_versions = []
        for version_idx in range(num_versions):
            chunks = [c for c in results[text_idx][version_idx] if c]
            # Join chunks (K2: reassemble to preserve global coherence)
            rephrased_text = " ".join(chunks) if chunks else ""
            text_versions.append(rephrased_text)
        final_results.append(text_versions)

    return final_results


def compute_fidelity(
    model,  # Transformer model
    enc,  # Encoding
    original_texts: List[str],
    rephrased_texts: List[str],
    layer: int = 18,
    device: torch.device = None,
) -> List[float]:
    """Compute semantic fidelity scores between original and rephrased texts.

    K2 methodology: Verify semantic alignment to ensure rephrasing preserves
    factual information and key concepts.

    Uses cosine similarity of mean-pooled hidden states from specified layer.

    Args:
        model: Transformer model for embeddings
        enc: Encoding for tokenization
        original_texts: Original texts
        rephrased_texts: Rephrased versions
        layer: Layer to extract hidden states from (default: 18)
        device: Device for computation

    Returns:
        List of cosine similarity scores [0, 1] per pair
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for semantic similarity scoring")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import tiktoken
    tok_enc = tiktoken.get_encoding("o200k_harmony")

    def get_embedding(text: str) -> torch.Tensor:
        """Get mean-pooled embedding from specified layer."""
        tokens = tok_enc.encode_ordinary(text)[:4096]  # Truncate to context limit
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        positions = torch.arange(len(tokens), dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            _, hidden_states = model(
                input_ids,
                positions,
                return_hidden_states=True,
                up_to_layer=layer + 1,
                no_logits=True,
            )
            h = hidden_states.get(layer + 1)  # Layer indexing (layer 18 → captured at end of layer 17)
            if h is None:
                raise ValueError(f"Layer {layer} not in hidden states")

            # Mean pool over sequence
            embedding = h.mean(dim=1).squeeze(0)  # [hidden_size]
            return torch.nn.functional.normalize(embedding, p=2, dim=0)

    scores = []
    for orig, reph in zip(original_texts, rephrased_texts):
        try:
            emb_orig = get_embedding(orig)
            emb_reph = get_embedding(reph)
            cosine_sim = float(torch.dot(emb_orig, emb_reph).item())
            scores.append(max(0.0, min(1.0, cosine_sim)))  # Clamp to [0, 1]
        except Exception:
            scores.append(0.0)  # Failed computation

    return scores


def filter_by_fidelity(
    rephrased_versions: List[List[str]],
    fidelity_scores: List[List[float]],
    threshold: float = 0.85,
) -> List[List[str]]:
    """Filter rephrased versions by fidelity threshold.

    K2 methodology: Only keep rephrasings that preserve semantic content above threshold.

    Args:
        rephrased_versions: [n_docs][n_versions] rephrased texts
        fidelity_scores: [n_docs][n_versions] fidelity scores
        threshold: Minimum fidelity score to keep (default: 0.85)

    Returns:
        Filtered rephrased versions
    """
    filtered = []
    for doc_versions, doc_scores in zip(rephrased_versions, fidelity_scores):
        kept_versions = [
            ver for ver, score in zip(doc_versions, doc_scores)
            if score >= threshold
        ]
        filtered.append(kept_versions)
    return filtered
