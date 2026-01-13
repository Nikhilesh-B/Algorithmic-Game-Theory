from leduc_env import LeducHoldemEnv
from agents import RandomAgent, HeuristicAgent
import random

def play_match(agent0, agent1, num_hands=1000):
    env = LeducHoldemEnv()
    scores = {0: 0, 1: 0}
    
    for _ in range(num_hands):
        state = env.reset()
        done = False
        
        # In Leduc/HeadsUp, active_player might switch round to round,
        # but our env handles active_player internally.
        
        while not done:
            current_player = env.active_player
            legal_actions = env.get_legal_actions()
            
            # Get action from agent
            if current_player == 0:
                # Hack: Just try until legal for RandomAgent
                # Real agents should know legal actions.
                # Let's wrap this
                action = -1
                while action not in legal_actions:
                    action = agent0.act(state)
            else:
                action = -1
                while action not in legal_actions:
                    action = agent1.act(state)
                
            state, payoffs, done = env.step(action)
            
            if done:
                scores[0] += payoffs[0]
                scores[1] += payoffs[1]
                
    return scores

if __name__ == "__main__":
    bot_random = RandomAgent()
    bot_heuristic = HeuristicAgent()
    
    # 1. Random vs Random
    print("Running Random vs Random (1000 hands)...")
    res = play_match(bot_random, bot_random, 1000)
    print(f"Result: {res}")
    
    # 2. Heuristic vs Random
    print("\nRunning Heuristic vs Random (1000 hands)...")
    res = play_match(bot_heuristic, bot_random, 1000)
    print(f"Result: {res}")
    print("Heuristic bot should win easily.")


