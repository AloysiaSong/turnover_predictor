"""
Hard Negative Mining for Preference Learning
============================================
Sample hard negatives based on:
1. In-batch hard negatives (highest scores among negatives)
2. Cached global hard negatives (periodically updated)
3. Mixed sampling strategy (random + hard)

This significantly improves preference learning by providing challenging negatives.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


class HardNegativeSampler:
    """
    Hard Negative Sampler for preference learning.

    Maintains a cache of hard negatives (high-scoring but non-preferred items)
    and samples a mix of hard and random negatives during training.

    Args:
        num_items: Total number of items (posts) in the dataset
        hard_ratio: Proportion of hard negatives vs random (0.0-1.0)
        cache_size: Number of hard negatives to cache per user
        update_freq: How often (in epochs) to update the cache
    """

    def __init__(
        self,
        num_items: int,
        hard_ratio: float = 0.7,
        cache_size: int = 5,
        update_freq: int = 5,
    ) -> None:
        self.num_items = num_items
        self.hard_ratio = hard_ratio
        self.cache_size = cache_size
        self.update_freq = update_freq

        # Cache structure: {user_id: [item_ids]}
        self.hard_neg_cache: Dict[int, List[int]] = defaultdict(list)
        self.epoch_counter = 0

    def update_cache(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        positive_pairs: torch.Tensor,
    ) -> None:
        """
        Update the hard negative cache.

        For each user, find items with highest scores that are NOT preferred.

        Args:
            user_emb: [num_users, dim] user embeddings
            item_emb: [num_items, dim] item embeddings
            positive_pairs: [num_pairs, 3] tensor of (user, preferred_item, dispreferred_item)
        """
        self.epoch_counter += 1

        # Only update every update_freq epochs
        if self.epoch_counter % self.update_freq != 0:
            return

        self.hard_neg_cache.clear()

        with torch.no_grad():
            # Compute all user-item scores: [num_users, num_items]
            scores = torch.matmul(user_emb, item_emb.T)

            # Build positive mask (items that users prefer)
            positive_mask = torch.zeros_like(scores, dtype=torch.bool)
            for user_id, preferred_item, _ in positive_pairs:
                if 0 <= user_id < scores.size(0) and 0 <= preferred_item < scores.size(1):
                    positive_mask[user_id, preferred_item] = True

            # Mask out positive items (set scores to -inf)
            scores_masked = scores.clone()
            scores_masked[positive_mask] = float('-inf')

            # For each user, select top-k hard negatives
            for user_id in range(user_emb.size(0)):
                # Get top-k highest-scoring non-preferred items
                top_scores, top_items = torch.topk(
                    scores_masked[user_id],
                    k=min(self.cache_size, self.num_items),
                    largest=True,
                )

                # Filter out -inf scores (when all items are positive or not enough items)
                valid_mask = top_scores > float('-inf')
                valid_items = top_items[valid_mask]

                if valid_items.numel() > 0:
                    self.hard_neg_cache[user_id] = valid_items.cpu().tolist()

    def sample_negatives(
        self,
        user_ids: torch.Tensor,
        preferred_items: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample negative items for each (user, preferred_item) pair.

        Uses a mixture of hard negatives (from cache) and random negatives.

        Args:
            user_ids: [batch_size] tensor of user IDs
            preferred_items: [batch_size] tensor of preferred item IDs
            device: torch device

        Returns:
            negative_items: [batch_size] tensor of sampled negative item IDs
        """
        batch_size = user_ids.size(0)
        negatives = []

        for i in range(batch_size):
            user_id = user_ids[i].item()
            preferred_item = preferred_items[i].item()

            # Decide whether to use hard or random negative
            use_hard = (
                torch.rand(1).item() < self.hard_ratio
                and user_id in self.hard_neg_cache
                and len(self.hard_neg_cache[user_id]) > 0
            )

            if use_hard:
                # Sample from cached hard negatives
                candidates = [
                    item for item in self.hard_neg_cache[user_id]
                    if item != preferred_item
                ]

                if candidates:
                    neg_item = candidates[torch.randint(len(candidates), (1,)).item()]
                else:
                    # Fallback to random
                    neg_item = self._random_negative(preferred_item)
            else:
                # Sample random negative
                neg_item = self._random_negative(preferred_item)

            negatives.append(neg_item)

        return torch.tensor(negatives, dtype=torch.long, device=device)

    def _random_negative(self, preferred_item: int) -> int:
        """
        Sample a random negative item (different from preferred item).

        Args:
            preferred_item: The preferred item to avoid

        Returns:
            A random item ID different from preferred_item
        """
        max_attempts = 100
        for _ in range(max_attempts):
            neg_item = torch.randint(0, self.num_items, (1,)).item()
            if neg_item != preferred_item:
                return neg_item

        # Fallback: just return a different item
        return (preferred_item + 1) % self.num_items

    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about the current cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.hard_neg_cache:
            return {
                "num_users_cached": 0,
                "avg_cache_size": 0.0,
                "cache_fill_rate": 0.0,
            }

        cache_sizes = [len(items) for items in self.hard_neg_cache.values()]

        return {
            "num_users_cached": len(self.hard_neg_cache),
            "avg_cache_size": sum(cache_sizes) / len(cache_sizes),
            "cache_fill_rate": sum(cache_sizes) / (len(cache_sizes) * self.cache_size),
        }
