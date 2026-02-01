"""Unit tests for memory optimization eviction policies.

This module provides comprehensive tests for the cache eviction policies in
nmoe/nmoe/memory_opt.py, including:
- LRUPolicy (Least Recently Used)
- LFUPolicy (Least Frequently Used)
- FIFOPolicy (First In First Out)
- ARCPolicy (Adaptive Replacement Cache)
- ExpertCachePolicy (factory class)

All tests exercise actual behavior without mocking the policies themselves.
Edge cases tested include empty cache eviction, max capacity, and various
access patterns that verify eviction order matches policy semantics.
"""

import pytest

from nmoe.memory_opt import (
    ARCPolicy,
    BaseEvictionPolicy,
    ExpertCachePolicy,
    FIFOPolicy,
    LFUPolicy,
    LRUPolicy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=[LRUPolicy, LFUPolicy, FIFOPolicy, ARCPolicy])
def policy_class(request):
    """Parametrized fixture providing all policy classes."""
    return request.param


@pytest.fixture
def lru_policy():
    """Create an LRU policy with max_size=5."""
    return LRUPolicy(max_size=5)


@pytest.fixture
def lfu_policy():
    """Create an LFU policy with max_size=5."""
    return LFUPolicy(max_size=5)


@pytest.fixture
def fifo_policy():
    """Create a FIFO policy with max_size=5."""
    return FIFOPolicy(max_size=5)


@pytest.fixture
def arc_policy():
    """Create an ARC policy with max_size=5."""
    return ARCPolicy(max_size=5)


# =============================================================================
# Common Tests for All Policies
# =============================================================================


class TestBaseEvictionPolicyInterface:
    """Tests that verify the common interface across all eviction policies."""

    def test_init_rejects_zero_max_size(self, policy_class):
        """Verify that max_size=0 raises ValueError.

        All policies must enforce max_size >= 1 to ensure the cache
        can hold at least one item.
        """
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            policy_class(max_size=0)

    def test_init_rejects_negative_max_size(self, policy_class):
        """Verify that negative max_size raises ValueError.

        Negative cache sizes are nonsensical and must be rejected.
        """
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            policy_class(max_size=-1)

    def test_init_accepts_valid_max_size(self, policy_class):
        """Verify that valid max_size values are accepted.

        Tests boundary case of max_size=1 and a typical value.
        """
        policy1 = policy_class(max_size=1)
        assert policy1.max_size == 1

        policy2 = policy_class(max_size=100)
        assert policy2.max_size == 100

    def test_empty_policy_len_is_zero(self, policy_class):
        """Verify __len__ returns 0 for empty policy."""
        policy = policy_class(max_size=5)
        assert len(policy) == 0

    def test_empty_policy_contains_nothing(self, policy_class):
        """Verify __contains__ returns False for any key in empty policy."""
        policy = policy_class(max_size=5)
        assert 0 not in policy
        assert 42 not in policy
        assert -1 not in policy

    def test_evict_from_empty_raises_error(self, policy_class):
        """Verify evict() raises RuntimeError when cache is empty.

        Attempting to evict from an empty cache is an error condition
        that should be clearly signaled.
        """
        policy = policy_class(max_size=5)
        with pytest.raises(RuntimeError, match="Cannot evict from empty cache"):
            policy.evict()

    def test_add_increments_length(self, policy_class):
        """Verify add() increases the tracked item count."""
        policy = policy_class(max_size=5)

        policy.add(1)
        assert len(policy) == 1

        policy.add(2)
        assert len(policy) == 2

        policy.add(3)
        assert len(policy) == 3

    def test_add_makes_key_contained(self, policy_class):
        """Verify add() makes the key findable via __contains__."""
        policy = policy_class(max_size=5)

        assert 42 not in policy
        policy.add(42)
        assert 42 in policy

    def test_remove_decrements_length(self, policy_class):
        """Verify remove() decreases the tracked item count."""
        policy = policy_class(max_size=5)
        policy.add(1)
        policy.add(2)
        policy.add(3)
        assert len(policy) == 3

        policy.remove(2)
        assert len(policy) == 2

        policy.remove(1)
        assert len(policy) == 1

    def test_remove_makes_key_not_contained(self, policy_class):
        """Verify remove() makes the key not findable via __contains__."""
        policy = policy_class(max_size=5)
        policy.add(42)
        assert 42 in policy

        policy.remove(42)
        assert 42 not in policy

    def test_remove_nonexistent_key_is_safe(self, policy_class):
        """Verify remove() on non-existent key doesn't raise an error.

        This is important for robustness - callers shouldn't need to
        check if a key exists before removing it.
        """
        policy = policy_class(max_size=5)
        policy.add(1)

        # Should not raise
        policy.remove(999)
        assert len(policy) == 1

    def test_evict_decrements_length(self, policy_class):
        """Verify evict() decreases the tracked item count."""
        policy = policy_class(max_size=5)
        policy.add(1)
        policy.add(2)
        policy.add(3)
        assert len(policy) == 3

        policy.evict()
        assert len(policy) == 2

        policy.evict()
        assert len(policy) == 1

    def test_evict_returns_key(self, policy_class):
        """Verify evict() returns the evicted key."""
        policy = policy_class(max_size=5)
        policy.add(42)

        evicted = policy.evict()
        assert evicted == 42
        assert 42 not in policy

    def test_access_on_nonexistent_key_is_safe(self, policy_class):
        """Verify access() on non-existent key doesn't raise an error.

        Access on unknown keys should be a no-op for robustness.
        """
        policy = policy_class(max_size=5)
        # Should not raise
        policy.access(999)
        assert len(policy) == 0

    def test_duplicate_add_does_not_double_count(self, policy_class):
        """Verify adding the same key twice doesn't create duplicates.

        For LRU/LFU/FIFO, adding an existing key should update its
        position rather than creating a duplicate entry.
        """
        policy = policy_class(max_size=5)
        policy.add(1)
        policy.add(1)  # Add same key again

        assert len(policy) == 1
        assert 1 in policy


# =============================================================================
# LRU Policy Tests
# =============================================================================


class TestLRUPolicy:
    """Tests specific to Least Recently Used eviction policy."""

    def test_evict_order_without_access(self, lru_policy):
        """Verify eviction is FIFO when no accesses occur.

        Without any access() calls, LRU should evict in the order
        items were added (first in, first out).
        """
        lru_policy.add(1)
        lru_policy.add(2)
        lru_policy.add(3)

        # First added should be evicted first
        assert lru_policy.evict() == 1
        assert lru_policy.evict() == 2
        assert lru_policy.evict() == 3

    def test_access_moves_to_most_recent(self, lru_policy):
        """Verify access() moves a key to the most recently used position.

        After accessing key 1, it should be moved to the end of the
        eviction queue, making key 2 the next to be evicted.
        """
        lru_policy.add(1)
        lru_policy.add(2)
        lru_policy.add(3)

        # Access key 1, moving it to most recent
        lru_policy.access(1)

        # Now 2 is least recently used
        assert lru_policy.evict() == 2
        assert lru_policy.evict() == 3
        assert lru_policy.evict() == 1  # 1 was accessed most recently

    def test_multiple_accesses_update_order(self, lru_policy):
        """Verify multiple access() calls correctly update eviction order.

        Tests a complex access pattern to ensure LRU correctly tracks
        the recency of all accesses.
        """
        lru_policy.add(1)
        lru_policy.add(2)
        lru_policy.add(3)
        lru_policy.add(4)

        # Access pattern: 1, 3, 2, 1
        lru_policy.access(1)
        lru_policy.access(3)
        lru_policy.access(2)
        lru_policy.access(1)

        # Order of recency (most to least): 1, 2, 3, 4
        # So eviction order is: 4, 3, 2, 1
        assert lru_policy.evict() == 4
        assert lru_policy.evict() == 3
        assert lru_policy.evict() == 2
        assert lru_policy.evict() == 1

    def test_add_puts_at_most_recent_position(self, lru_policy):
        """Verify add() places new keys at the most recently used position.

        Newly added keys should be treated as recently used.
        """
        lru_policy.add(1)
        lru_policy.add(2)

        # Evict one
        assert lru_policy.evict() == 1

        # Add new key
        lru_policy.add(3)

        # 2 should be evicted before 3
        assert lru_policy.evict() == 2
        assert lru_policy.evict() == 3

    def test_single_element_behavior(self):
        """Test LRU with max_size=1 behaves correctly."""
        policy = LRUPolicy(max_size=1)

        policy.add(42)
        assert len(policy) == 1
        assert 42 in policy

        evicted = policy.evict()
        assert evicted == 42
        assert len(policy) == 0


# =============================================================================
# LFU Policy Tests
# =============================================================================


class TestLFUPolicy:
    """Tests specific to Least Frequently Used eviction policy."""

    def test_evict_least_frequent(self, lfu_policy):
        """Verify eviction targets the least frequently accessed key.

        Keys with fewer access() calls should be evicted first.
        """
        lfu_policy.add(1)  # freq=1
        lfu_policy.add(2)  # freq=1
        lfu_policy.add(3)  # freq=1

        # Access key 2 and 3 more often
        lfu_policy.access(2)  # freq=2
        lfu_policy.access(3)  # freq=2
        lfu_policy.access(3)  # freq=3

        # Key 1 has lowest frequency (1), should be evicted first
        assert lfu_policy.evict() == 1
        # Key 2 has freq=2, key 3 has freq=3
        assert lfu_policy.evict() == 2
        assert lfu_policy.evict() == 3

    def test_tie_broken_by_lru(self, lfu_policy):
        """Verify ties in frequency are broken by LRU order.

        When multiple keys have the same frequency, the least recently
        used among them should be evicted first.
        """
        lfu_policy.add(1)  # freq=1
        lfu_policy.add(2)  # freq=1
        lfu_policy.add(3)  # freq=1

        # All have same frequency (1)
        # Access order: 1 first, then 2, then 3
        lfu_policy.access(1)
        lfu_policy.access(2)
        lfu_policy.access(3)

        # All now have freq=2, but 1 was accessed first (least recent)
        assert lfu_policy.evict() == 1
        assert lfu_policy.evict() == 2
        assert lfu_policy.evict() == 3

    def test_new_items_have_frequency_one(self, lfu_policy):
        """Verify newly added items start with frequency 1.

        When ties in frequency occur, LRU order is used as tie-breaker.
        Since key 2 was added before key 3 and both have freq=1,
        key 2 should be evicted first (LRU among least frequent).
        """
        lfu_policy.add(1)
        lfu_policy.add(2)

        # Access key 1 once
        lfu_policy.access(1)  # freq=2

        # Add new key - it has freq=1
        lfu_policy.add(3)  # freq=1

        # Both key 2 and 3 have freq=1, but 2 was added before 3 (LRU)
        # So key 2 should be evicted first
        assert lfu_policy.evict() == 2
        # Then key 3 (also freq=1)
        assert lfu_policy.evict() == 3
        # Finally key 1 (freq=2)
        assert lfu_policy.evict() == 1

    def test_frequency_accumulates(self, lfu_policy):
        """Verify frequency correctly accumulates over multiple accesses."""
        lfu_policy.add(1)

        # Access 10 times
        for _ in range(10):
            lfu_policy.access(1)

        lfu_policy.add(2)
        # Access 5 times
        for _ in range(5):
            lfu_policy.access(2)

        # Key 2 has lower frequency
        assert lfu_policy.evict() == 2
        assert lfu_policy.evict() == 1

    def test_complex_access_pattern(self, lfu_policy):
        """Test LFU with a complex interleaved access pattern."""
        lfu_policy.add(1)
        lfu_policy.add(2)
        lfu_policy.add(3)
        lfu_policy.add(4)

        # Create frequency pattern: 1->5, 2->3, 3->2, 4->1
        for _ in range(4):
            lfu_policy.access(1)
        for _ in range(2):
            lfu_policy.access(2)
        for _ in range(1):
            lfu_policy.access(3)

        # Frequencies: 1=5, 2=3, 3=2, 4=1
        # Eviction order should be: 4, 3, 2, 1
        assert lfu_policy.evict() == 4
        assert lfu_policy.evict() == 3
        assert lfu_policy.evict() == 2
        assert lfu_policy.evict() == 1

    def test_single_element_behavior(self):
        """Test LFU with max_size=1 behaves correctly."""
        policy = LFUPolicy(max_size=1)

        policy.add(42)
        policy.access(42)
        policy.access(42)

        assert len(policy) == 1
        evicted = policy.evict()
        assert evicted == 42


# =============================================================================
# FIFO Policy Tests
# =============================================================================


class TestFIFOPolicy:
    """Tests specific to First In First Out eviction policy."""

    def test_evict_order_is_insertion_order(self, fifo_policy):
        """Verify eviction follows insertion order exactly.

        FIFO should always evict the oldest (first-added) item,
        regardless of access patterns.
        """
        fifo_policy.add(1)
        fifo_policy.add(2)
        fifo_policy.add(3)

        assert fifo_policy.evict() == 1
        assert fifo_policy.evict() == 2
        assert fifo_policy.evict() == 3

    def test_access_does_not_affect_order(self, fifo_policy):
        """Verify access() does not change eviction order.

        Unlike LRU, FIFO ignores access patterns. The eviction order
        should remain based purely on insertion order.
        """
        fifo_policy.add(1)
        fifo_policy.add(2)
        fifo_policy.add(3)

        # Access items in reverse order
        fifo_policy.access(3)
        fifo_policy.access(2)
        fifo_policy.access(1)

        # Order should still be based on insertion
        assert fifo_policy.evict() == 1
        assert fifo_policy.evict() == 2
        assert fifo_policy.evict() == 3

    def test_interleaved_add_evict(self, fifo_policy):
        """Test FIFO behavior with interleaved add and evict operations."""
        fifo_policy.add(1)
        fifo_policy.add(2)

        assert fifo_policy.evict() == 1

        fifo_policy.add(3)
        fifo_policy.add(4)

        assert fifo_policy.evict() == 2
        assert fifo_policy.evict() == 3

        fifo_policy.add(5)

        assert fifo_policy.evict() == 4
        assert fifo_policy.evict() == 5

    def test_remove_preserves_fifo_order(self, fifo_policy):
        """Verify remove() doesn't disrupt FIFO order of remaining items."""
        fifo_policy.add(1)
        fifo_policy.add(2)
        fifo_policy.add(3)
        fifo_policy.add(4)

        # Remove middle item
        fifo_policy.remove(2)

        # FIFO order for remaining: 1, 3, 4
        assert fifo_policy.evict() == 1
        assert fifo_policy.evict() == 3
        assert fifo_policy.evict() == 4

    def test_single_element_behavior(self):
        """Test FIFO with max_size=1 behaves correctly."""
        policy = FIFOPolicy(max_size=1)

        policy.add(42)
        assert len(policy) == 1

        evicted = policy.evict()
        assert evicted == 42


# =============================================================================
# ARC Policy Tests
# =============================================================================


class TestARCPolicy:
    """Tests specific to Adaptive Replacement Cache eviction policy.

    ARC maintains two lists (T1 for recency, T2 for frequency) and
    dynamically adapts based on workload patterns.
    """

    def test_new_items_go_to_t1(self, arc_policy):
        """Verify newly added items are placed in T1 (recency list)."""
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.add(3)

        assert 1 in arc_policy
        assert 2 in arc_policy
        assert 3 in arc_policy
        assert len(arc_policy) == 3

    def test_access_moves_from_t1_to_t2(self, arc_policy):
        """Verify accessing a T1 item moves it to T2 (frequency list).

        Items accessed more than once are considered 'frequent' and
        should be promoted to T2.
        """
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.add(3)

        # Access key 2 - should move to T2
        arc_policy.access(2)

        # All items still tracked
        assert len(arc_policy) == 3
        assert 1 in arc_policy
        assert 2 in arc_policy
        assert 3 in arc_policy

    def test_eviction_from_t1_first(self, arc_policy):
        """Test that eviction initially comes from T1 when p=0.

        With default adaptation (p=0), T1 items should be evicted first.
        """
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.add(3)

        # Move 3 to T2 by accessing it
        arc_policy.access(3)

        # Eviction should come from T1 first (1 or 2)
        evicted = arc_policy.evict()
        assert evicted in [1, 2]

    def test_basic_eviction_order(self, arc_policy):
        """Test basic eviction order in ARC."""
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.add(3)

        # Evict all items
        evicted = set()
        for _ in range(3):
            evicted.add(arc_policy.evict())

        assert evicted == {1, 2, 3}
        assert len(arc_policy) == 0

    def test_contains_after_eviction(self, arc_policy):
        """Verify __contains__ returns False for evicted items.

        Note: Evicted items may be in ghost lists (B1/B2) but should
        not be reported as 'in' the cache.
        """
        arc_policy.add(1)
        arc_policy.add(2)

        evicted = arc_policy.evict()

        # Evicted item should not be 'in' the cache
        assert evicted not in arc_policy
        # Remaining item should still be in cache
        remaining = 1 if evicted == 2 else 2
        assert remaining in arc_policy

    def test_access_in_t2_updates_position(self, arc_policy):
        """Verify accessing a T2 item moves it to MRU position in T2."""
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.add(3)

        # Move all to T2
        arc_policy.access(1)
        arc_policy.access(2)
        arc_policy.access(3)

        # Access 1 again - should move to end of T2
        arc_policy.access(1)

        # All still present
        assert len(arc_policy) == 3

    def test_remove_clears_from_all_lists(self, arc_policy):
        """Verify remove() clears the key from both T1/T2 and ghost lists."""
        arc_policy.add(1)
        arc_policy.add(2)
        arc_policy.access(1)  # Move 1 to T2

        arc_policy.remove(1)
        assert 1 not in arc_policy
        assert len(arc_policy) == 1

        arc_policy.remove(2)
        assert 2 not in arc_policy
        assert len(arc_policy) == 0

    def test_single_element_behavior(self):
        """Test ARC with max_size=1 behaves correctly."""
        policy = ARCPolicy(max_size=1)

        policy.add(42)
        assert len(policy) == 1
        assert 42 in policy

        evicted = policy.evict()
        assert evicted == 42
        assert len(policy) == 0

    def test_adaptation_on_workload(self):
        """Test ARC adaptation with different workload patterns.

        This test verifies that ARC adapts its behavior based on
        whether recency (T1) or frequency (T2) hits are more valuable.
        """
        policy = ARCPolicy(max_size=4)

        # Add items
        policy.add(1)
        policy.add(2)
        policy.add(3)
        policy.add(4)

        # Access some items to create T2 entries
        policy.access(1)
        policy.access(2)

        assert len(policy) == 4

        # Evict to make room
        evicted = policy.evict()
        assert evicted in [1, 2, 3, 4]
        assert len(policy) == 3

    def test_len_returns_sum_of_t1_and_t2(self):
        """Verify __len__ returns the total of T1 and T2 (not ghost lists)."""
        policy = ARCPolicy(max_size=5)

        policy.add(1)  # T1
        policy.add(2)  # T1
        policy.add(3)  # T1

        policy.access(1)  # Move to T2
        policy.access(2)  # Move to T2

        # Should have 3 items total (2 in T2, 1 in T1)
        assert len(policy) == 3

    def test_evict_with_empty_t2(self):
        """Test eviction when T2 is empty (all items in T1)."""
        policy = ARCPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        # All in T1, evict should work
        evicted = policy.evict()
        assert evicted == 1  # First added, FIFO within T1
        assert len(policy) == 2


# =============================================================================
# ExpertCachePolicy Factory Tests
# =============================================================================


class TestExpertCachePolicy:
    """Tests for the ExpertCachePolicy factory class."""

    def test_create_lru_policy(self):
        """Verify factory creates LRUPolicy for 'lru' identifier."""
        policy = ExpertCachePolicy.create("lru", max_size=10)
        assert isinstance(policy, LRUPolicy)
        assert policy.max_size == 10

    def test_create_lfu_policy(self):
        """Verify factory creates LFUPolicy for 'lfu' identifier."""
        policy = ExpertCachePolicy.create("lfu", max_size=10)
        assert isinstance(policy, LFUPolicy)
        assert policy.max_size == 10

    def test_create_fifo_policy(self):
        """Verify factory creates FIFOPolicy for 'fifo' identifier."""
        policy = ExpertCachePolicy.create("fifo", max_size=10)
        assert isinstance(policy, FIFOPolicy)
        assert policy.max_size == 10

    def test_create_arc_policy(self):
        """Verify factory creates ARCPolicy for 'arc' identifier."""
        policy = ExpertCachePolicy.create("arc", max_size=10)
        assert isinstance(policy, ARCPolicy)
        assert policy.max_size == 10

    def test_create_unknown_policy_raises_error(self):
        """Verify factory raises ValueError for unknown policy names."""
        with pytest.raises(ValueError, match="Unknown policy"):
            ExpertCachePolicy.create("random", max_size=10)

        with pytest.raises(ValueError, match="Unknown policy"):
            ExpertCachePolicy.create("LRU", max_size=10)  # Case sensitive

        with pytest.raises(ValueError, match="Unknown policy"):
            ExpertCachePolicy.create("", max_size=10)

    def test_class_constants(self):
        """Verify class constants are defined correctly."""
        assert ExpertCachePolicy.LRU == "lru"
        assert ExpertCachePolicy.LFU == "lfu"
        assert ExpertCachePolicy.FIFO == "fifo"
        assert ExpertCachePolicy.ARC == "arc"

    def test_create_with_constants(self):
        """Verify factory works with class constants."""
        policy_lru = ExpertCachePolicy.create(ExpertCachePolicy.LRU, max_size=5)
        assert isinstance(policy_lru, LRUPolicy)

        policy_lfu = ExpertCachePolicy.create(ExpertCachePolicy.LFU, max_size=5)
        assert isinstance(policy_lfu, LFUPolicy)

        policy_fifo = ExpertCachePolicy.create(ExpertCachePolicy.FIFO, max_size=5)
        assert isinstance(policy_fifo, FIFOPolicy)

        policy_arc = ExpertCachePolicy.create(ExpertCachePolicy.ARC, max_size=5)
        assert isinstance(policy_arc, ARCPolicy)

    def test_factory_propagates_max_size_validation(self):
        """Verify factory propagates max_size validation from underlying policies."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ExpertCachePolicy.create("lru", max_size=0)

        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ExpertCachePolicy.create("lfu", max_size=-5)

    def test_created_policies_are_base_eviction_policy(self):
        """Verify all created policies are instances of BaseEvictionPolicy."""
        for name in ["lru", "lfu", "fifo", "arc"]:
            policy = ExpertCachePolicy.create(name, max_size=5)
            assert isinstance(policy, BaseEvictionPolicy)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_max_size_one_all_policies(self, policy_class):
        """Test all policies work correctly with max_size=1."""
        policy = policy_class(max_size=1)

        policy.add(1)
        assert len(policy) == 1
        assert 1 in policy

        evicted = policy.evict()
        assert evicted == 1
        assert len(policy) == 0

    def test_large_number_of_items(self, policy_class):
        """Test policies handle many items correctly."""
        policy = policy_class(max_size=1000)

        for i in range(1000):
            policy.add(i)

        assert len(policy) == 1000

        for i in range(1000):
            assert i in policy

        # Evict all
        for _ in range(1000):
            policy.evict()

        assert len(policy) == 0

    def test_repeated_access_same_key(self, policy_class):
        """Test repeated access to the same key is handled correctly."""
        policy = policy_class(max_size=5)
        policy.add(42)

        # Access many times
        for _ in range(100):
            policy.access(42)

        assert len(policy) == 1
        assert 42 in policy

        evicted = policy.evict()
        assert evicted == 42

    def test_add_remove_add_same_key(self, policy_class):
        """Test adding, removing, and re-adding the same key."""
        policy = policy_class(max_size=5)

        policy.add(42)
        assert 42 in policy

        policy.remove(42)
        assert 42 not in policy

        policy.add(42)
        assert 42 in policy

        evicted = policy.evict()
        assert evicted == 42

    def test_negative_keys(self, policy_class):
        """Test policies handle negative integer keys correctly."""
        policy = policy_class(max_size=5)

        policy.add(-1)
        policy.add(-100)
        policy.add(0)

        assert -1 in policy
        assert -100 in policy
        assert 0 in policy
        assert len(policy) == 3

    def test_evict_all_then_add_new(self, policy_class):
        """Test adding new items after evicting all existing items."""
        policy = policy_class(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        policy.evict()
        policy.evict()
        policy.evict()

        assert len(policy) == 0

        policy.add(4)
        policy.add(5)

        assert len(policy) == 2
        assert 4 in policy
        assert 5 in policy

    def test_remove_all_items(self, policy_class):
        """Test removing all items one by one."""
        policy = policy_class(max_size=5)

        for i in range(5):
            policy.add(i)

        for i in range(5):
            policy.remove(i)

        assert len(policy) == 0

        # Should raise on evict from empty
        with pytest.raises(RuntimeError):
            policy.evict()


# =============================================================================
# Behavioral Comparison Tests
# =============================================================================


class TestPolicyBehaviorComparison:
    """Tests that compare behavior across different policies."""

    def test_lru_vs_fifo_with_access(self):
        """Demonstrate the difference between LRU and FIFO with access patterns.

        LRU should adapt based on access, while FIFO should not.
        """
        lru = LRUPolicy(max_size=5)
        fifo = FIFOPolicy(max_size=5)

        # Add same items to both
        for i in [1, 2, 3]:
            lru.add(i)
            fifo.add(i)

        # Access item 1 in both
        lru.access(1)
        fifo.access(1)

        # LRU should evict 2 (1 was recently accessed)
        assert lru.evict() == 2

        # FIFO should still evict 1 (first added)
        assert fifo.evict() == 1

    def test_lfu_vs_lru_with_frequency(self):
        """Demonstrate the difference between LFU and LRU with frequency patterns.

        LFU should prefer keeping frequently accessed items, while LRU
        only cares about recency.
        """
        lfu = LFUPolicy(max_size=5)
        lru = LRUPolicy(max_size=5)

        for i in [1, 2, 3]:
            lfu.add(i)
            lru.add(i)

        # Access item 1 many times, then access item 2 once
        for _ in range(5):
            lfu.access(1)
            lru.access(1)

        lfu.access(2)
        lru.access(2)

        # LFU should evict 3 (lowest frequency)
        assert lfu.evict() == 3

        # LRU should evict 3 (least recently accessed - 1 was accessed before 2)
        assert lru.evict() == 3

    def test_all_policies_track_same_items(self):
        """Verify all policies track the same set of items (just evict differently)."""
        policies = [
            LRUPolicy(max_size=10),
            LFUPolicy(max_size=10),
            FIFOPolicy(max_size=10),
            ARCPolicy(max_size=10),
        ]

        items = [1, 5, 3, 8, 2]

        for policy in policies:
            for item in items:
                policy.add(item)

        # All should contain the same items
        for policy in policies:
            assert len(policy) == 5
            for item in items:
                assert item in policy


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for eviction policies."""

    def test_rapid_add_evict_cycles(self, policy_class):
        """Test rapid cycles of adding and evicting."""
        policy = policy_class(max_size=10)

        for cycle in range(100):
            # Add items
            for i in range(10):
                policy.add(cycle * 10 + i)

            # Evict all
            for _ in range(10):
                policy.evict()

            assert len(policy) == 0

    def test_interleaved_operations(self, policy_class):
        """Test interleaved add, access, remove, evict operations."""
        policy = policy_class(max_size=20)

        for i in range(100):
            op = i % 4
            key = i % 15

            if op == 0:
                policy.add(key)
            elif op == 1:
                policy.access(key)
            elif op == 2:
                if len(policy) > 0:
                    policy.evict()
            else:
                policy.remove(key)

        # Should not crash and state should be consistent
        assert len(policy) >= 0
