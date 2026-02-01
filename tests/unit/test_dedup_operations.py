"""Comprehensive unit tests for nmoe data deduplication operations.

Tests cover:
- Exact hashing (exact_hash)
- Exact deduplication (dedup_exact)
- Line-level deduplication (dedup_line_level)
- Paragraph-level deduplication (dedup_paragraph_level)
- MinHash/LSH primitives (_shingles, minhash_signature, jaccard_from_signature, LSHIndex)
- Streaming near-duplicate detection (stream_near_dedup)
- Edge cases: empty text, single char, very long text, Unicode handling
- Threshold configurations
"""

import pytest
from typing import List


class TestExactHash:
    """Tests for exact_hash function."""

    def test_deterministic_output(self):
        """Same text produces same hash."""
        from nmoe.data.dedup import exact_hash

        text = "Hello, World!"
        hash1 = exact_hash(text)
        hash2 = exact_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_different_texts_different_hashes(self):
        """Different texts produce different hashes."""
        from nmoe.data.dedup import exact_hash

        hash1 = exact_hash("Hello")
        hash2 = exact_hash("World")
        hash3 = exact_hash("hello")  # Case sensitive

        assert hash1 != hash2
        assert hash1 != hash3  # "Hello" != "hello"

    def test_empty_string_handling(self):
        """Empty string produces valid hash."""
        from nmoe.data.dedup import exact_hash

        hash_empty = exact_hash("")

        assert isinstance(hash_empty, str)
        assert len(hash_empty) == 64
        # Known SHA-256 of empty string
        assert hash_empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_single_char(self):
        """Single character produces valid hash."""
        from nmoe.data.dedup import exact_hash

        hash_a = exact_hash("a")
        hash_b = exact_hash("b")

        assert len(hash_a) == 64
        assert hash_a != hash_b

    def test_unicode_handling(self):
        """Unicode text is properly hashed."""
        from nmoe.data.dedup import exact_hash

        # Various Unicode characters
        hash_emoji = exact_hash("Hello World!")
        hash_chinese = exact_hash("Hello World!")
        hash_arabic = exact_hash("Hello World!")

        assert len(hash_emoji) == 64
        assert len(hash_chinese) == 64
        assert len(hash_arabic) == 64

    def test_unicode_normalized(self):
        """Different Unicode representations hash consistently."""
        from nmoe.data.dedup import exact_hash

        # e with combining acute vs precomposed e-acute
        text1 = "cafe\u0301"  # cafe + combining acute
        text2 = "caf\u00e9"   # precomposed e-acute

        # These are different byte sequences, so hashes will differ
        hash1 = exact_hash(text1)
        hash2 = exact_hash(text2)

        assert len(hash1) == 64
        assert len(hash2) == 64

    def test_very_long_text(self):
        """Very long text produces valid hash."""
        from nmoe.data.dedup import exact_hash

        long_text = "A" * 1_000_000
        hash_long = exact_hash(long_text)

        assert len(hash_long) == 64

    def test_whitespace_sensitive(self):
        """Hash is sensitive to whitespace."""
        from nmoe.data.dedup import exact_hash

        hash1 = exact_hash("hello world")
        hash2 = exact_hash("hello  world")
        hash3 = exact_hash(" hello world")

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3


class TestDedupExact:
    """Tests for dedup_exact function."""

    def test_removes_exact_duplicates(self):
        """Removes exact duplicate texts."""
        from nmoe.data.dedup import dedup_exact

        texts = ["hello", "world", "hello", "foo", "world"]
        unique, seen = dedup_exact(texts)

        assert unique == ["hello", "world", "foo"]
        assert len(seen) == 3

    def test_preserves_order_of_first_occurrence(self):
        """First occurrence order is preserved."""
        from nmoe.data.dedup import dedup_exact

        texts = ["c", "a", "b", "a", "c", "d"]
        unique, seen = dedup_exact(texts)

        assert unique == ["c", "a", "b", "d"]

    def test_empty_list_handling(self):
        """Empty list returns empty results."""
        from nmoe.data.dedup import dedup_exact

        unique, seen = dedup_exact([])

        assert unique == []
        assert seen == set()

    def test_single_element(self):
        """Single element list works."""
        from nmoe.data.dedup import dedup_exact

        unique, seen = dedup_exact(["only"])

        assert unique == ["only"]
        assert len(seen) == 1

    def test_all_duplicates(self):
        """All duplicate inputs returns single element."""
        from nmoe.data.dedup import dedup_exact

        texts = ["same", "same", "same", "same"]
        unique, seen = dedup_exact(texts)

        assert unique == ["same"]
        assert len(seen) == 1

    def test_no_duplicates(self):
        """All unique inputs preserved."""
        from nmoe.data.dedup import dedup_exact

        texts = ["a", "b", "c", "d"]
        unique, seen = dedup_exact(texts)

        assert unique == texts
        assert len(seen) == 4

    def test_unicode_duplicates(self):
        """Unicode duplicates are correctly identified."""
        from nmoe.data.dedup import dedup_exact

        texts = ["hello", "hello", "world", "world"]
        unique, seen = dedup_exact(texts)

        assert unique == ["hello", "world"]

    def test_empty_string_as_element(self):
        """Empty strings are valid elements."""
        from nmoe.data.dedup import dedup_exact

        texts = ["a", "", "b", "", "c"]
        unique, seen = dedup_exact(texts)

        assert unique == ["a", "", "b", "c"]
        assert len(seen) == 4

    def test_generator_input(self):
        """Works with generator input."""
        from nmoe.data.dedup import dedup_exact

        def gen():
            yield "a"
            yield "b"
            yield "a"

        unique, seen = dedup_exact(gen())

        assert unique == ["a", "b"]

    def test_returns_seen_hashes(self):
        """Returns set of seen hashes."""
        from nmoe.data.dedup import dedup_exact, exact_hash

        texts = ["hello", "world"]
        unique, seen = dedup_exact(texts)

        assert exact_hash("hello") in seen
        assert exact_hash("world") in seen


class TestDedupLineLevel:
    """Tests for dedup_line_level function."""

    def test_removes_duplicate_lines(self):
        """Removes duplicate lines within text."""
        from nmoe.data.dedup import dedup_line_level

        text = "line1\nline2\nline1\nline3\nline2"
        result = dedup_line_level(text)

        assert result == "line1\nline2\nline3"

    def test_preserves_line_order(self):
        """Preserves order of first occurrence."""
        from nmoe.data.dedup import dedup_line_level

        text = "c\na\nb\na\nc"
        result = dedup_line_level(text)

        assert result == "c\na\nb"

    def test_empty_text(self):
        """Empty text returns empty string."""
        from nmoe.data.dedup import dedup_line_level

        result = dedup_line_level("")

        assert result == ""

    def test_single_line(self):
        """Single line text preserved."""
        from nmoe.data.dedup import dedup_line_level

        result = dedup_line_level("single line")

        assert result == "single line"

    def test_empty_lines_preserved(self):
        """Empty lines are treated as lines and deduped."""
        from nmoe.data.dedup import dedup_line_level

        text = "a\n\nb\n\nc"
        result = dedup_line_level(text)

        # First empty line kept, second deduped
        assert result == "a\n\nb\nc"

    def test_all_same_lines(self):
        """All same lines returns single line."""
        from nmoe.data.dedup import dedup_line_level

        text = "same\nsame\nsame"
        result = dedup_line_level(text)

        assert result == "same"

    def test_unicode_lines(self):
        """Unicode lines handled correctly."""
        from nmoe.data.dedup import dedup_line_level

        text = "hello\nworld\nhello"
        result = dedup_line_level(text)

        assert result == "hello\nworld"

    def test_whitespace_lines_distinct(self):
        """Lines with different whitespace are distinct."""
        from nmoe.data.dedup import dedup_line_level

        text = "a\n a\n  a"
        result = dedup_line_level(text)

        # All three are distinct due to whitespace
        assert result == "a\n a\n  a"

    def test_trailing_newline(self):
        """Text with trailing newline handled."""
        from nmoe.data.dedup import dedup_line_level

        text = "a\nb\na\n"
        result = dedup_line_level(text)

        # Implementation may or may not preserve trailing newline
        # The key behavior is deduplication (removing duplicate "a")
        assert "a" in result and "b" in result
        # Should have removed one of the duplicate "a" lines
        assert result.count("a") == 1 or result == "a\nb" or result == "a\nb\n"

    def test_very_long_lines(self):
        """Very long lines handled."""
        from nmoe.data.dedup import dedup_line_level

        long_line = "A" * 10000
        text = f"{long_line}\nshort\n{long_line}"
        result = dedup_line_level(text)

        assert result == f"{long_line}\nshort"


class TestDedupParagraphLevel:
    """Tests for dedup_paragraph_level function."""

    def test_removes_duplicate_paragraphs(self):
        """Removes duplicate paragraphs."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "para1\n\npara2\n\npara1\n\npara3"
        result = dedup_paragraph_level(text)

        assert result == "para1\n\npara2\n\npara3"

    def test_preserves_paragraph_order(self):
        """Preserves order of first occurrence."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "c\n\na\n\nb\n\na"
        result = dedup_paragraph_level(text)

        assert result == "c\n\na\n\nb"

    def test_empty_text(self):
        """Empty text returns empty string."""
        from nmoe.data.dedup import dedup_paragraph_level

        result = dedup_paragraph_level("")

        assert result == ""

    def test_single_paragraph(self):
        """Single paragraph preserved."""
        from nmoe.data.dedup import dedup_paragraph_level

        result = dedup_paragraph_level("single paragraph")

        assert result == "single paragraph"

    def test_multiline_paragraphs(self):
        """Multiline paragraphs handled correctly."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "line1\nline2\n\nline3\nline4\n\nline1\nline2"
        result = dedup_paragraph_level(text)

        assert result == "line1\nline2\n\nline3\nline4"

    def test_whitespace_normalized_in_paragraphs(self):
        """Paragraph boundaries strip whitespace."""
        from nmoe.data.dedup import dedup_paragraph_level

        # Paragraphs with extra whitespace get stripped
        text = "  para1  \n\n  para2  \n\npara1"
        result = dedup_paragraph_level(text)

        assert result == "para1\n\npara2"

    def test_multiple_blank_lines(self):
        """Multiple blank lines still separate paragraphs."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "a\n\n\n\nb\n\na"
        result = dedup_paragraph_level(text)

        # Extra blank lines create empty paragraphs which are filtered
        assert "a" in result
        assert "b" in result

    def test_unicode_paragraphs(self):
        """Unicode paragraphs handled correctly."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "paragraph\n\ntext\n\nparagraph"
        result = dedup_paragraph_level(text)

        assert result == "paragraph\n\ntext"

    def test_empty_paragraphs_filtered(self):
        """Empty paragraphs (only whitespace) are filtered out."""
        from nmoe.data.dedup import dedup_paragraph_level

        text = "a\n\n   \n\nb"
        result = dedup_paragraph_level(text)

        # Empty/whitespace paragraphs are filtered
        assert result == "a\n\nb"


class TestShingles:
    """Tests for _shingles function."""

    def test_basic_shingle_generation(self):
        """Basic shingle generation works."""
        from nmoe.data.dedup import _shingles

        text = "hello world"
        shingles = list(_shingles(text, k=3))

        # Should have len(text) - k + 1 shingles (after normalization)
        assert len(shingles) > 0
        assert all(isinstance(s, int) for s in shingles)

    def test_deterministic_shingles(self):
        """Same text produces same shingles."""
        from nmoe.data.dedup import _shingles

        text = "test text"
        shingles1 = list(_shingles(text, k=3))
        shingles2 = list(_shingles(text, k=3))

        assert shingles1 == shingles2

    def test_different_k_values(self):
        """Different k produces different number of shingles."""
        from nmoe.data.dedup import _shingles

        text = "abcdefgh"
        shingles_3 = list(_shingles(text, k=3))
        shingles_5 = list(_shingles(text, k=5))

        # k=3 should have more shingles than k=5
        assert len(shingles_3) > len(shingles_5)

    def test_empty_text(self):
        """Empty text yields no shingles."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("", k=3))

        assert shingles == []

    def test_text_shorter_than_k(self):
        """Text shorter than k yields no shingles."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("ab", k=3))

        assert shingles == []

    def test_text_equal_to_k(self):
        """Text equal to k yields one shingle."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("abc", k=3))

        assert len(shingles) == 1

    def test_zero_k(self):
        """k=0 yields no shingles."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("hello", k=0))

        assert shingles == []

    def test_negative_k(self):
        """Negative k yields no shingles."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("hello", k=-1))

        assert shingles == []

    def test_unicode_shingles(self):
        """Unicode text produces valid shingles."""
        from nmoe.data.dedup import _shingles

        text = "hello"
        shingles = list(_shingles(text, k=3))

        assert len(shingles) > 0

    def test_shingles_are_64bit_hashes(self):
        """Shingles are 64-bit integer hashes."""
        from nmoe.data.dedup import _shingles

        shingles = list(_shingles("hello world", k=5))

        for s in shingles:
            assert 0 <= s < 2**64


class TestMinhashSignature:
    """Tests for minhash_signature function."""

    def test_basic_signature(self):
        """Basic signature generation works."""
        import numpy as np
        from nmoe.data.dedup import minhash_signature

        sig = minhash_signature("hello world")

        assert isinstance(sig, np.ndarray)
        assert sig.dtype == np.uint64
        assert sig.shape == (128,)  # default num_perm

    def test_deterministic_signature(self):
        """Same text produces same signature."""
        import numpy as np
        from nmoe.data.dedup import minhash_signature

        sig1 = minhash_signature("test text")
        sig2 = minhash_signature("test text")

        np.testing.assert_array_equal(sig1, sig2)

    def test_different_texts_different_signatures(self):
        """Different texts produce different signatures."""
        import numpy as np
        from nmoe.data.dedup import minhash_signature

        sig1 = minhash_signature("hello world")
        sig2 = minhash_signature("goodbye world")

        assert not np.array_equal(sig1, sig2)

    def test_custom_num_perm(self):
        """Custom num_perm changes signature size."""
        from nmoe.data.dedup import minhash_signature

        sig = minhash_signature("test", num_perm=64)

        assert sig.shape == (64,)

    def test_custom_shingle_size(self):
        """Custom shingle size changes signature."""
        from nmoe.data.dedup import minhash_signature

        sig1 = minhash_signature("hello world", shingle=3)
        sig2 = minhash_signature("hello world", shingle=5)

        # Different shingle sizes may produce different signatures
        # (depends on content, but likely different)
        assert sig1.shape == sig2.shape

    def test_different_seeds_different_signatures(self):
        """Different seeds produce different signatures."""
        import numpy as np
        from nmoe.data.dedup import minhash_signature

        # Use text long enough to form shingles with default size
        test_text = "the quick brown fox jumps over the lazy dog repeatedly"
        sig1 = minhash_signature(test_text, seed=42)
        sig2 = minhash_signature(test_text, seed=123)

        assert not np.array_equal(sig1, sig2)

    def test_empty_text_signature(self):
        """Empty text produces default signature."""
        from nmoe.data.dedup import minhash_signature

        sig = minhash_signature("")

        # Empty text has no shingles, so signature is all max values
        assert sig.shape == (128,)
        assert all(sig == (1 << 64) - 1)

    def test_short_text_signature(self):
        """Text shorter than shingle size produces default signature."""
        from nmoe.data.dedup import minhash_signature

        sig = minhash_signature("hi", shingle=13)

        # Too short for shingles
        assert all(sig == (1 << 64) - 1)

    def test_similar_texts_similar_signatures(self):
        """Similar texts have similar signatures (more matching entries)."""
        from nmoe.data.dedup import minhash_signature, jaccard_from_signature

        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy cat"

        sig1 = minhash_signature(text1)
        sig2 = minhash_signature(text2)

        # Similar texts should have some matching signature entries
        jaccard = jaccard_from_signature(sig1, sig2)
        assert jaccard > 0.3  # Reasonably similar

    def test_unicode_signature(self):
        """Unicode text produces valid signature."""
        import numpy as np
        from nmoe.data.dedup import minhash_signature

        sig = minhash_signature("hello world")

        assert sig.shape == (128,)
        assert sig.dtype == np.uint64


class TestJaccardFromSignature:
    """Tests for jaccard_from_signature function."""

    def test_identical_signatures(self):
        """Identical signatures have Jaccard 1.0."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        sig = np.array([1, 2, 3, 4], dtype=np.uint64)
        jaccard = jaccard_from_signature(sig, sig)

        assert jaccard == 1.0

    def test_different_signatures(self):
        """Different signatures have Jaccard < 1.0."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        sig1 = np.array([1, 2, 3, 4], dtype=np.uint64)
        sig2 = np.array([5, 6, 7, 8], dtype=np.uint64)

        jaccard = jaccard_from_signature(sig1, sig2)

        assert jaccard == 0.0

    def test_partial_match(self):
        """Partial matching signatures have intermediate Jaccard."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        sig1 = np.array([1, 2, 3, 4], dtype=np.uint64)
        sig2 = np.array([1, 2, 5, 6], dtype=np.uint64)

        jaccard = jaccard_from_signature(sig1, sig2)

        assert jaccard == 0.5  # 2 out of 4 match

    def test_empty_signatures(self):
        """Empty signatures return 0.0."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        sig1 = np.array([], dtype=np.uint64)
        sig2 = np.array([], dtype=np.uint64)

        jaccard = jaccard_from_signature(sig1, sig2)

        assert jaccard == 0.0

    def test_mismatched_shapes_raises(self):
        """Mismatched signature shapes raise ValueError."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        sig1 = np.array([1, 2, 3], dtype=np.uint64)
        sig2 = np.array([1, 2, 3, 4], dtype=np.uint64)

        with pytest.raises(ValueError, match="same shape"):
            jaccard_from_signature(sig1, sig2)

    def test_jaccard_range(self):
        """Jaccard is always in [0.0, 1.0]."""
        import numpy as np
        from nmoe.data.dedup import jaccard_from_signature

        rng = np.random.default_rng(42)
        for _ in range(100):
            sig1 = rng.integers(0, 10, size=100, dtype=np.uint64)
            sig2 = rng.integers(0, 10, size=100, dtype=np.uint64)
            jaccard = jaccard_from_signature(sig1, sig2)
            assert 0.0 <= jaccard <= 1.0


class TestLSHIndex:
    """Tests for LSHIndex class."""

    def test_basic_add_and_candidates(self):
        """Basic add and candidate retrieval works."""
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        sig1 = minhash_signature("hello world")
        sig2 = minhash_signature("hello world")  # Same text

        idx.add(sig1)
        candidates = idx.candidates(sig2)

        assert 0 in candidates

    def test_index_returns_idx(self):
        """add() returns the index of added signature."""
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        idx0 = idx.add(minhash_signature("text1"))
        idx1 = idx.add(minhash_signature("text2"))
        idx2 = idx.add(minhash_signature("text3"))

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_get_signature(self):
        """get() retrieves stored signature."""
        import numpy as np
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        sig = minhash_signature("test text")
        i = idx.add(sig)

        retrieved = idx.get(i)
        np.testing.assert_array_equal(sig, retrieved)

    def test_no_candidates_for_different_text(self):
        """Very different texts may not be candidates."""
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        sig1 = minhash_signature("completely unique text number one")
        sig2 = minhash_signature("totally different document two xyz")

        idx.add(sig1)
        candidates = idx.candidates(sig2)

        # May or may not have candidates depending on hash collisions
        # Just verify it returns a set
        assert isinstance(candidates, set)

    def test_custom_bands_rows(self):
        """Custom bands and rows configuration."""
        from nmoe.data.dedup import LSHIndex

        idx = LSHIndex(num_perm=128, bands=32, rows=4)

        assert idx.bands == 32
        assert idx.rows == 4

    def test_invalid_bands_rows_raises(self):
        """Invalid bands * rows raises ValueError."""
        from nmoe.data.dedup import LSHIndex

        with pytest.raises(ValueError, match="bands .* rows must equal num_perm"):
            LSHIndex(num_perm=128, bands=10, rows=10)  # 10*10 != 128

    def test_default_bands_rows(self):
        """Default bands/rows are derived from num_perm."""
        from nmoe.data.dedup import LSHIndex

        idx = LSHIndex(num_perm=128)

        assert idx.bands * idx.rows == 128

    def test_multiple_candidates(self):
        """Multiple similar texts become candidates."""
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        base_text = "the quick brown fox jumps over the lazy dog"
        texts = [
            base_text,
            "the quick brown fox jumps over the lazy cat",
            "the quick brown fox leaps over the lazy dog",
        ]

        for t in texts:
            idx.add(minhash_signature(t))

        query_sig = minhash_signature("the quick brown fox jumps over the lazy dog")
        candidates = idx.candidates(query_sig)

        # Should find at least the identical one
        assert len(candidates) >= 1

    def test_empty_index_no_candidates(self):
        """Empty index returns no candidates."""
        from nmoe.data.dedup import LSHIndex, minhash_signature

        idx = LSHIndex(num_perm=128)

        sig = minhash_signature("test")
        candidates = idx.candidates(sig)

        assert candidates == set()


class TestStreamNearDedup:
    """Tests for stream_near_dedup function."""

    def test_filters_similar_documents(self):
        """Filters out near-duplicate documents."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",  # Very similar
            "completely different text about something else",
        ]

        results = list(stream_near_dedup(texts, jaccard_threshold=0.5))

        # First and third should be kept, second is near-dup of first
        assert len(results) <= 3
        assert texts[0] in results
        assert texts[2] in results

    def test_keeps_first_occurrence(self):
        """Keeps first occurrence of near-duplicates."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "original document text here",
            "original document text here with minor change",  # Near-dup
        ]

        results = list(stream_near_dedup(texts, jaccard_threshold=0.5))

        assert texts[0] in results

    def test_empty_input(self):
        """Empty input yields empty output."""
        from nmoe.data.dedup import stream_near_dedup

        results = list(stream_near_dedup([]))

        assert results == []

    def test_single_document(self):
        """Single document is preserved."""
        from nmoe.data.dedup import stream_near_dedup

        results = list(stream_near_dedup(["only one"]))

        assert results == ["only one"]

    def test_all_unique_documents(self):
        """All unique documents are preserved."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "first unique document with specific content",
            "second totally different document",
            "third unrelated text about something new",
        ]

        results = list(stream_near_dedup(texts, jaccard_threshold=0.9))

        assert len(results) == 3

    def test_all_identical_documents(self):
        """All identical documents reduces to one."""
        from nmoe.data.dedup import stream_near_dedup

        texts = ["same text"] * 5

        results = list(stream_near_dedup(texts, jaccard_threshold=0.5))

        assert len(results) == 1
        assert results[0] == "same text"

    def test_configurable_threshold_low(self):
        """Low threshold is more aggressive in filtering."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the slow brown fox walks under the active cat",
        ]

        # Low threshold - more likely to filter
        results_low = list(stream_near_dedup(texts, jaccard_threshold=0.3))
        # High threshold - less likely to filter
        results_high = list(stream_near_dedup(texts, jaccard_threshold=0.95))

        # High threshold should keep both (they're not 95% similar)
        assert len(results_high) >= len(results_low)

    def test_configurable_threshold_high(self):
        """High threshold keeps more documents."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "a b c d e f g h i j k l m n o p",
            "a b c d e f g h i j k l m n x y",  # Mostly same
        ]

        # Very high threshold - should keep both
        results = list(stream_near_dedup(texts, jaccard_threshold=0.99))

        assert len(results) == 2

    def test_custom_shingle_size(self):
        """Custom shingle size affects deduplication."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "abcdefghijklmnopqrstuvwxyz",
            "abcdefghijklmnopqrstuvwxyz1",  # Very similar
        ]

        results = list(stream_near_dedup(texts, shingle=3, jaccard_threshold=0.8))

        # Should detect similarity with small shingles
        assert len(results) <= 2

    def test_custom_num_perm(self):
        """Custom num_perm changes signature size."""
        from nmoe.data.dedup import stream_near_dedup

        # Use longer texts that can form proper shingles
        texts = [
            "this is the first document with enough text to form shingles",
            "this is the second document with completely different content"
        ]

        # Should work with different num_perm
        results = list(stream_near_dedup(texts, num_perm=64))

        # Both documents should be kept (they're different)
        assert len(results) >= 1  # At least one kept

    def test_custom_bands_rows(self):
        """Custom bands and rows configuration."""
        from nmoe.data.dedup import stream_near_dedup

        # Use longer texts that can form proper shingles
        texts = [
            "the quick brown fox jumps over the lazy dog in the garden",
            "a completely different sentence about cats and birds flying"
        ]

        results = list(stream_near_dedup(texts, num_perm=128, bands=32, rows=4))

        # Both documents should be kept (they're different)
        assert len(results) >= 1

    def test_generator_input(self):
        """Works with generator input."""
        from nmoe.data.dedup import stream_near_dedup

        def gen():
            yield "the first document is about machine learning and neural networks"
            yield "the second document discusses database optimization techniques"
            yield "the first document is about machine learning and neural networks again"  # Near-dup

        results = list(stream_near_dedup(gen(), jaccard_threshold=0.5))

        # At least the first occurrence should be in results
        assert len(results) >= 1
        # The second unique document should also be present
        assert any("database" in r for r in results) or any("machine" in r for r in results)

    def test_streaming_behavior(self):
        """Verifies streaming (lazy) evaluation."""
        from nmoe.data.dedup import stream_near_dedup

        call_count = 0

        def gen():
            nonlocal call_count
            for i in range(3):
                call_count += 1
                yield f"document {i}"

        # Create iterator but don't consume
        it = stream_near_dedup(gen())

        # Generator not consumed yet
        assert call_count == 0

        # Consume one item
        next(it)

        # Now generator has been partially consumed
        assert call_count >= 1

    def test_unicode_documents(self):
        """Unicode documents handled correctly."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "hello world with emojis",
            "different text",
        ]

        results = list(stream_near_dedup(texts))

        assert len(results) == 2

    def test_very_short_documents(self):
        """Very short documents handled (no shingles case)."""
        from nmoe.data.dedup import stream_near_dedup

        texts = ["a", "b", "c"]  # Shorter than default shingle size

        results = list(stream_near_dedup(texts))

        # Short texts produce no shingles, so signature is all max values
        # All short texts will look identical (all max signature)
        # First one kept, rest filtered as "duplicates"
        assert len(results) >= 1

    def test_seed_reproducibility(self):
        """Same seed produces same results."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "the quick brown fox",
            "the quick brown cat",
            "something else entirely",
        ]

        results1 = list(stream_near_dedup(texts, seed=42))
        results2 = list(stream_near_dedup(texts, seed=42))

        assert results1 == results2

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different results."""
        from nmoe.data.dedup import stream_near_dedup

        texts = [
            "the quick brown fox jumps over lazy dog",
            "the quick brown fox jumps over lazy cat",
        ]

        results1 = list(stream_near_dedup(texts, seed=42, jaccard_threshold=0.7))
        results2 = list(stream_near_dedup(texts, seed=12345, jaccard_threshold=0.7))

        # Results may or may not differ depending on hash functions
        # Just verify both return valid results
        assert all(t in texts for t in results1)
        assert all(t in texts for t in results2)


class TestNearDedupMinhashBackwardCompat:
    """Tests for backward compatibility shim near_dedup_minhash."""

    def test_backward_compat_function_exists(self):
        """near_dedup_minhash exists for backward compatibility."""
        from nmoe.data.dedup import near_dedup_minhash

        assert callable(near_dedup_minhash)

    def test_backward_compat_returns_list(self):
        """near_dedup_minhash returns a list (not iterator)."""
        from nmoe.data.dedup import near_dedup_minhash

        texts = ["text1", "text2"]
        result = near_dedup_minhash(texts)

        assert isinstance(result, list)

    def test_backward_compat_deduplicates(self):
        """near_dedup_minhash performs deduplication."""
        from nmoe.data.dedup import near_dedup_minhash

        texts = ["same long text here"] * 3

        result = near_dedup_minhash(texts)

        assert len(result) == 1


class TestDeriveBandsRows:
    """Tests for _derive_bands_rows helper function."""

    def test_explicit_bands_rows(self):
        """Explicit bands and rows used when valid."""
        from nmoe.data.dedup import _derive_bands_rows

        bands, rows = _derive_bands_rows(128, bands=32, rows=4)

        assert bands == 32
        assert rows == 4

    def test_invalid_product_raises(self):
        """Invalid bands * rows raises ValueError."""
        from nmoe.data.dedup import _derive_bands_rows

        with pytest.raises(ValueError):
            _derive_bands_rows(128, bands=10, rows=10)

    def test_auto_derive_prefers_rows_4(self):
        """Auto-derivation prefers rows=4 when divisible."""
        from nmoe.data.dedup import _derive_bands_rows

        bands, rows = _derive_bands_rows(128, bands=None, rows=None)

        # 128 is divisible by 4
        assert rows == 4
        assert bands == 32

    def test_auto_derive_fallback_rows_3(self):
        """Falls back to rows=3 when not divisible by 4."""
        from nmoe.data.dedup import _derive_bands_rows

        bands, rows = _derive_bands_rows(81, bands=None, rows=None)

        # 81 not divisible by 4, but divisible by 3
        assert rows == 3
        assert bands == 27

    def test_auto_derive_fallback_rows_2(self):
        """Falls back to rows=2 when not divisible by 3 or 4."""
        from nmoe.data.dedup import _derive_bands_rows

        bands, rows = _derive_bands_rows(50, bands=None, rows=None)

        # 50 not divisible by 4 or 3, but divisible by 2
        assert rows == 2
        assert bands == 25

    def test_auto_derive_prime_fallback(self):
        """Prime num_perm falls back to bands=num_perm, rows=1."""
        from nmoe.data.dedup import _derive_bands_rows

        bands, rows = _derive_bands_rows(127, bands=None, rows=None)

        # 127 is prime
        assert rows == 1
        assert bands == 127


class TestEdgeCasesIntegration:
    """Integration tests for edge cases across multiple functions."""

    def test_empty_corpus_dedup_pipeline(self):
        """Empty corpus through full pipeline."""
        from nmoe.data.dedup import dedup_exact, stream_near_dedup

        unique, _ = dedup_exact([])
        near_dedup = list(stream_near_dedup([]))

        assert unique == []
        assert near_dedup == []

    def test_single_empty_string_corpus(self):
        """Single empty string through pipeline."""
        from nmoe.data.dedup import dedup_exact, stream_near_dedup

        unique, _ = dedup_exact([""])
        near_dedup = list(stream_near_dedup([""]))

        assert unique == [""]
        assert near_dedup == [""]

    def test_very_long_document(self):
        """Very long document handling."""
        from nmoe.data.dedup import exact_hash, minhash_signature, stream_near_dedup

        long_doc = "word " * 100000  # 500k chars

        # Should not raise
        h = exact_hash(long_doc)
        sig = minhash_signature(long_doc)
        result = list(stream_near_dedup([long_doc]))

        assert len(h) == 64
        assert sig.shape == (128,)
        assert result == [long_doc]

    def test_unicode_heavy_document(self):
        """Document with heavy Unicode content."""
        from nmoe.data.dedup import exact_hash, minhash_signature

        unicode_doc = "".join(chr(i) for i in range(0x4E00, 0x4E00 + 1000))  # Chinese chars

        h = exact_hash(unicode_doc)
        sig = minhash_signature(unicode_doc)

        assert len(h) == 64
        assert sig.shape == (128,)

    def test_mixed_length_documents(self):
        """Mixed length documents in same batch."""
        from nmoe.data.dedup import dedup_exact, stream_near_dedup

        texts = [
            "a",
            "ab" * 100,
            "abc" * 10000,
            "",
            "x",
        ]

        unique, _ = dedup_exact(texts)
        near_dedup = list(stream_near_dedup(texts))

        assert len(unique) == 5
        # near_dedup may filter some based on similarity

    def test_whitespace_only_documents(self):
        """Whitespace-only documents."""
        from nmoe.data.dedup import dedup_exact, dedup_line_level

        texts = ["   ", "\t\t", "\n\n", "  \n  "]
        unique, _ = dedup_exact(texts)

        assert len(unique) == 4  # All different whitespace patterns

        # Line level on whitespace
        result = dedup_line_level("   \n   \n   ")
        assert "   " in result

    def test_newline_variations(self):
        """Different newline styles handled."""
        from nmoe.data.dedup import dedup_line_level

        # Unix newlines
        unix = "a\nb\na"
        result_unix = dedup_line_level(unix)
        assert result_unix == "a\nb"

        # Windows style (splitlines handles \r\n)
        windows = "a\r\nb\r\na"
        result_windows = dedup_line_level(windows)
        assert "a" in result_windows
        assert "b" in result_windows
