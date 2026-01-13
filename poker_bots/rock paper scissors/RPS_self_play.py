import numpy as np

NUM_ACTIONS = 3
ROCK, PAPER, SCISSORS = 0, 1, 2
TRAIN_ITERATIONS = 100000


def regret_matching(regret_sum, strategy_sum):
    """Return current strategy from regrets and accumulate into strategy_sum."""
    strategy = [max(r, 0.0) for r in regret_sum]
    norm = sum(strategy)
    if norm > 0:
        strategy = [s / norm for s in strategy]
    else:
        strategy = [1.0 / NUM_ACTIONS for _ in range(NUM_ACTIONS)]

    for i in range(NUM_ACTIONS):
        strategy_sum[i] += strategy[i]
    return strategy


def sample_action(strategy):
    return np.random.choice(len(strategy), p=strategy)


def payoff_vs_action(opp_action):
    """Utilities of Rock/Paper/Scissors against a specific opponent action."""
    utils = [0.0] * NUM_ACTIONS
    utils[opp_action] = 0.0
    utils[(opp_action + 1) % NUM_ACTIONS] = 1.0   # action that beats opp_action
    utils[(opp_action + 2) % NUM_ACTIONS] = -1.0  # action that loses
    return utils


def avg_strategy(strategy_sum):
    total = sum(strategy_sum)
    if total == 0:
        return [1.0 / NUM_ACTIONS for _ in range(NUM_ACTIONS)]
    return [s / total for s in strategy_sum]


def train_self_play(iters=TRAIN_ITERATIONS):
    # Both players start with zero regrets and no prior strategy mass.
    regret_sum_a = [0.0] * NUM_ACTIONS
    regret_sum_b = [0.0] * NUM_ACTIONS
    strategy_sum_a = [0.0] * NUM_ACTIONS
    strategy_sum_b = [0.0] * NUM_ACTIONS

    for _ in range(iters):
        # Current mixed strategies from regret matching.
        strat_a = regret_matching(regret_sum_a, strategy_sum_a)
        strat_b = regret_matching(regret_sum_b, strategy_sum_b)

        # Sample actions.
        action_a = sample_action(strat_a)
        action_b = sample_action(strat_b)

        # Compute utilities for each pure action against the opponent's sampled action.
        util_a = payoff_vs_action(action_b)
        util_b = payoff_vs_action(action_a)

        # Instant regret update.
        for i in range(NUM_ACTIONS):
            regret_sum_a[i] += util_a[i] - util_a[action_a]
            regret_sum_b[i] += util_b[i] - util_b[action_b]

    # Return average strategies (these are the convergent policies).
    return avg_strategy(strategy_sum_a), avg_strategy(strategy_sum_b)


if __name__ == "__main__":
    avg_a, avg_b = train_self_play()
    print(f"Average strategy (Player A): {avg_a}")
    print(f"Average strategy (Player B): {avg_b}")

