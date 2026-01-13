import numpy as np
import random as rand

NUM_ACTIONS = 3
ROCK, PAPER, SCISSORS = 0, 1, 2
TRAIN_ITERATIONS = 100000


def get_strategy(regret_sum, strategy, strategy_sum):
    norm_sum = 0
    for i in range(NUM_ACTIONS):
        strategy[i] = regret_sum[i] if regret_sum[i] > 0 else 0
        norm_sum += strategy[i]

    if norm_sum > 0:
        strategy = [x/norm_sum for x in strategy]
    else:
        strategy = [1/3 for _ in range(NUM_ACTIONS)]

    for i in range(NUM_ACTIONS):
        strategy_sum[i] += strategy[i]

    return strategy


def get_action(strategy):
    return np.random.choice(len(strategy), p=strategy)


def get_avg_strategy(strategy_sum):
    avg_strategy = [0 for _ in range(NUM_ACTIONS)]
    normalizing_sum = sum(strategy_sum)

    print(f"normalizing_sum={normalizing_sum}")
    if normalizing_sum == 0:
        avg_strategy = [1/NUM_ACTIONS for _ in range(NUM_ACTIONS)]

    else:
        avg_strategy = [action_sum /
                        normalizing_sum for action_sum in strategy_sum]

    return avg_strategy


def train():
    regret_sum = [0 for _ in range(NUM_ACTIONS)]
    strategy = [0 for _ in range(NUM_ACTIONS)]
    strategy_sum = [0 for _ in range(NUM_ACTIONS)]
    opponent_strategy = [0.4, 0.3, 0.3]

    my_strategy = get_strategy(regret_sum, strategy, strategy_sum)

    for _ in range(TRAIN_ITERATIONS):
        action_utilities = [0 for _ in range(NUM_ACTIONS)]
        my_action = get_action(my_strategy)
        opp_action = get_action(opponent_strategy)

        action_utilities[opp_action] = 0
        action_utilities[(opp_action+1) % 3] = 1
        action_utilities[opp_action-1] = -1

        for i in range(NUM_ACTIONS):
            regret_sum[i] += (action_utilities[i]-action_utilities[my_action])
        
        my_strategy = get_strategy(regret_sum, strategy, strategy_sum)

    my_strategy = get_avg_strategy(strategy_sum)
    print(f"Converged strategy = {my_strategy}")


if __name__ == "__main__":
    train()
