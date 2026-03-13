# Copyright 2025 Individual Contributor: dubin555
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CPU test to verify RLOO non-vectorized handles single-sample groups correctly.

Bug: In compute_rloo_outcome_advantage (non-vectorized), when a group has only 1 sample,
the leave-one-out advantage should be 0 (no baseline possible). But the code skips the
update entirely, leaving the raw score as the advantage.

The vectorized version (compute_rloo_vectorized_outcome_advantage) correctly zeros out
single-sample groups via `* (c > 1)`.

This creates an inconsistency where switching from "rloo" to "rloo_vectorized" changes
training behavior for single-sample groups.
"""

from collections import defaultdict

import numpy as np
import torch


def rloo_non_vec_buggy(token_level_rewards, response_mask, index):
    """Original (buggy) non-vectorized RLOO: single-sample groups keep raw score."""
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
            # BUG: no else branch → single-sample groups keep raw score
        scores = scores.unsqueeze(-1) * response_mask
    return scores


def rloo_non_vec_fixed(token_level_rewards, response_mask, index):
    """Fixed non-vectorized RLOO: single-sample groups get advantage 0."""
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
            else:
                scores[i] = 0.0
        scores = scores.unsqueeze(-1) * response_mask
    return scores


def rloo_vectorized(token_level_rewards, response_mask, index):
    """Vectorized RLOO (reference implementation, known correct)."""
    scores = token_level_rewards.sum(dim=-1)
    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)
        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)
        adv = adv.unsqueeze(-1) * response_mask
    return adv


def test_buggy_single_group_nonzero():
    """Demonstrate bug: non-vectorized RLOO gives non-zero advantage for single-sample groups."""
    rewards = torch.tensor(
        [
            [0, 0, 5.0, 0],  # group A (single sample), score=5
            [0, 0, 3.0, 0],  # group B, score=3
            [0, 0, 7.0, 0],  # group B, score=7
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "B", "B"])

    adv_buggy = rloo_non_vec_buggy(rewards.clone(), mask, index)

    # Bug: single-sample group A gets advantage = 5.0 (raw score) instead of 0
    assert adv_buggy[0, 2].item() == 5.0, (
        f"Expected buggy advantage = 5.0 for single group, got {adv_buggy[0, 2].item()}"
    )


def test_vectorized_single_group_zero():
    """Vectorized RLOO correctly gives 0 advantage for single-sample groups."""
    rewards = torch.tensor(
        [
            [0, 0, 5.0, 0],
            [0, 0, 3.0, 0],
            [0, 0, 7.0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "B", "B"])

    adv_vec = rloo_vectorized(rewards.clone(), mask, index)

    # Correct: single-sample group A gets advantage = 0
    assert adv_vec[0, 2].item() == 0.0, (
        f"Expected vectorized advantage = 0.0 for single group, got {adv_vec[0, 2].item()}"
    )


def test_fixed_matches_vectorized():
    """Fixed non-vectorized RLOO matches vectorized for all cases."""
    rewards = torch.tensor(
        [
            [0, 0, 5.0, 0],  # group A (single)
            [0, 0, 3.0, 0],  # group B
            [0, 0, 7.0, 0],  # group B
            [0, 0, 10.0, 0],  # group C (single)
            [0, 0, 1.0, 0],  # group D
            [0, 0, 2.0, 0],  # group D
            [0, 0, 3.0, 0],  # group D
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "B", "B", "C", "D", "D", "D"])

    adv_fixed = rloo_non_vec_fixed(rewards.clone(), mask, index)
    adv_vec = rloo_vectorized(rewards.clone(), mask, index)

    assert torch.allclose(adv_fixed, adv_vec, atol=1e-5), (
        f"Fixed non-vec should match vectorized!\n"
        f"Fixed: {adv_fixed[:, 2].tolist()}\n"
        f"Vec:   {adv_vec[:, 2].tolist()}\n"
        f"Max diff: {(adv_fixed - adv_vec).abs().max().item()}"
    )


def test_multi_sample_groups_unchanged():
    """Fix should not change behavior for multi-sample groups."""
    rewards = torch.tensor(
        [
            [0, 0, 3.0, 0],
            [0, 0, 7.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 5.0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "A", "B", "B"])

    adv_buggy = rloo_non_vec_buggy(rewards.clone(), mask, index)
    adv_fixed = rloo_non_vec_fixed(rewards.clone(), mask, index)

    # No single-sample groups → both should be identical
    assert torch.allclose(adv_buggy, adv_fixed, atol=1e-5), (
        f"Multi-sample groups should be unchanged by fix!\n"
        f"Buggy: {adv_buggy[:, 2].tolist()}\n"
        f"Fixed: {adv_fixed[:, 2].tolist()}"
    )


def test_all_single_groups():
    """Edge case: every group has exactly 1 sample."""
    rewards = torch.tensor(
        [
            [0, 0, 5.0, 0],
            [0, 0, 10.0, 0],
            [0, 0, -3.0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "B", "C"])

    adv_fixed = rloo_non_vec_fixed(rewards.clone(), mask, index)
    adv_vec = rloo_vectorized(rewards.clone(), mask, index)

    # All advantages should be 0
    assert adv_fixed.sum().item() == 0.0, (
        f"All single groups → all advantages should be 0, got sum={adv_fixed.sum().item()}"
    )
    assert torch.allclose(adv_fixed, adv_vec, atol=1e-5)


def test_rloo_formula_correctness():
    """Verify the RLOO formula: A_i = r_i - (1/(N-1)) * sum_{j!=i} r_j."""
    rewards = torch.tensor(
        [
            [0, 0, 2.0, 0],
            [0, 0, 4.0, 0],
            [0, 0, 6.0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 1.0, 0],
        ]
    )
    index = np.array(["A", "A", "A"])

    # Manual RLOO computation:
    # A_0 = 2 - (4+6)/2 = 2 - 5 = -3
    # A_1 = 4 - (2+6)/2 = 4 - 4 = 0
    # A_2 = 6 - (2+4)/2 = 6 - 3 = 3
    expected = torch.tensor(
        [
            [0, 0, -3.0, 0],
            [0, 0, 0.0, 0],
            [0, 0, 3.0, 0],
        ]
    )

    adv_fixed = rloo_non_vec_fixed(rewards.clone(), mask, index)
    adv_vec = rloo_vectorized(rewards.clone(), mask, index)

    assert torch.allclose(adv_fixed, expected, atol=1e-5), (
        f"RLOO formula mismatch!\nExpected: {expected[:, 2].tolist()}\nGot: {adv_fixed[:, 2].tolist()}"
    )
    assert torch.allclose(adv_vec, expected, atol=1e-5)


if __name__ == "__main__":
    test_buggy_single_group_nonzero()
    print("PASS: test_buggy_single_group_nonzero")

    test_vectorized_single_group_zero()
    print("PASS: test_vectorized_single_group_zero")

    test_fixed_matches_vectorized()
    print("PASS: test_fixed_matches_vectorized")

    test_multi_sample_groups_unchanged()
    print("PASS: test_multi_sample_groups_unchanged")

    test_all_single_groups()
    print("PASS: test_all_single_groups")

    test_rloo_formula_correctness()
    print("PASS: test_rloo_formula_correctness")

    print("\nAll tests passed!")
