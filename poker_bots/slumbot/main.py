import argparse
import sys
from .client import SlumbotClient
from .agent import RandomAgent
from .rebel_agent import ReBeLAgent

def main():
    parser = argparse.ArgumentParser(description='Play against Slumbot')
    parser.add_argument('--username', type=str, help='Slumbot username')
    parser.add_argument('--password', type=str, help='Slumbot password')
    parser.add_argument('--hands', type=int, default=10, help='Number of hands to play')
    parser.add_argument('--agent', type=str, default='random', choices=['random', 'rebel'], help='Agent type')
    args = parser.parse_args()
    
    client = SlumbotClient(args.username, args.password)
    
    if args.agent == 'rebel':
        print("Initializing ReBeL Agent...")
        agent = ReBeLAgent()
    else:
        agent = RandomAgent()
    
    total_winnings = 0
    
    for i in range(args.hands):
        print(f"\n--- Hand {i+1} ---")
        if hasattr(agent, 'reset_hand'):
            agent.reset_hand()
            
        try:
            r = client.new_hand()
            
            while True:
                if 'winnings' in r:
                    winnings = r['winnings']
                    print(f"Hand Over. Winnings: {winnings}")
                    total_winnings += winnings
                    break
                
                action_str = r.get('action', '')
                hole_cards = r.get('hole_cards')
                board = r.get('board')
                
                print(f"Board: {board}, Hole: {hole_cards}, Action: {action_str}")
                
                state = client.parse_action(action_str)
                if 'error' in state:
                    print(f"Error parsing action: {state['error']}")
                    break
                    
                my_action = agent.get_action(state, hole_cards, board)
                print(f"Agent Action: {my_action}")
                
                r = client.act(my_action)
                
        except Exception as e:
            print(f"Error in hand {i+1}: {e}")
            break
            
    print(f"\nTotal Winnings: {total_winnings}")
    print(f"Average bb/100: {(total_winnings / 100.0) / (args.hands / 100.0) if args.hands > 0 else 0}")

if __name__ == "__main__":
    main()
