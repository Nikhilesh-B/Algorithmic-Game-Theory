from kuhn_poker_env import KuhnPokerEnv
from agents import RandomAgent, HeuristicAgent

def play_match(agent0, agent1, num_hands=1000):
    env = KuhnPokerEnv()
    scores = {0: 0, 1: 0}
    
    for _ in range(num_hands):
        state = env.reset()
        done = False
        
        # Determine who starts based on env state (always starts with player 0)
        current_player = 0
        
        while not done:
            if current_player == 0:
                action = agent0.act(state)
            else:
                action = agent1.act(state)
                
            next_state, payoffs, done = env.step(action)
            
            if done:
                scores[0] += payoffs[0]
                scores[1] += payoffs[1]
            else:
                state = next_state
                current_player = 1 - current_player
                
    return scores

if __name__ == "__main__":
    bot_random = RandomAgent()
    bot_heuristic = HeuristicAgent()
    
    # 1. Random vs Random
    print("Running Random vs Random...")
    res = play_match(bot_random, bot_random, 10000)
    print(f"Result: {res}")
    
    # 2. Heuristic vs Random
    print("\nRunning Heuristic vs Random...")
    res = play_match(bot_heuristic, bot_random, 10000)
    print(f"Result: {res}")
    print("Heuristic bot should be winning significantly.")