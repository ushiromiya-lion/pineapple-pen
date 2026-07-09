import numpy as np

from genio.card_rewards import (
    CARD_REWARD_POOL,
    MIN_RARE_OFFSET,
    card_from_reward_def,
    generate_typist_card_rewards,
    roll_card_rarity,
    update_rare_offset,
)


def test_sts_reward_offset_updates():
    assert update_rare_offset(MIN_RARE_OFFSET, "common") == -4
    assert update_rare_offset(10, "uncommon") == 10
    assert update_rare_offset(10, "rare") == MIN_RARE_OFFSET
    assert update_rare_offset(40, "common") == 40


def test_initial_negative_offset_cannot_roll_rare():
    rng = np.random.default_rng(1)

    rolled = {roll_card_rarity(MIN_RARE_OFFSET, rng) for _ in range(100)}

    assert "rare" not in rolled


def test_typist_card_rewards_are_three_unique_template_cards():
    result = generate_typist_card_rewards(rng=np.random.default_rng(2))

    assert len(result.cards) == 3
    assert len({card.name for card in result.cards}) == 3
    assert all(card.is_prefix_card() for card in result.cards)
    assert all(card.energy_cost == 1 for card in result.cards)


def test_reward_pool_contains_requested_cards():
    assert {card.title for card in CARD_REWARD_POOL} == {
        "*___ Attack",
        "*___ Block",
        "*___ Trick",
        "Re____",
        "^*____",
        "*___ism",
        "____",
    }


def test_only_visible_reward_roles_are_used():
    cards_by_title = {
        reward.title: card_from_reward_def(reward, np.random.default_rng(0))
        for reward in CARD_REWARD_POOL
    }

    assert cards_by_title["*___ Attack"].prefix_role == "Attack"
    assert cards_by_title["*___ Attack"].rarity == "common"
    assert cards_by_title["Re____"].rarity == "uncommon"
    assert cards_by_title["*___ism"].prefix_role is None
    assert cards_by_title["*___ism"].name.endswith("ism")
