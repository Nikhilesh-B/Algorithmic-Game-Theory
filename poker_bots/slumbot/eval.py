import argparse
import statistics
import time
from typing import Tuple, Dict, Any

from .client import SlumbotClient
from .agent import RandomAgent
from .rebel_agent import ReBeLAgent


def play_hand(client: SlumbotClient, agent, verbose: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    Plays a single hand vs Slumbot with the provided agent.
    Returns (winnings, stats).
    """
    if hasattr(agent, "reset_hand"):
        agent.reset_hand()

    r = client.new_hand()
    stats = {"showdown": False, "folded_early": False}

    while True:
        if "winnings" in r:
            winnings = r["winnings"]
            board = r.get("board", [])
            stats["showdown"] = len(board) == 5
            stats["final_board_len"] = len(board)
            return winnings, stats

        action_str = r.get("action", "")
        hole_cards = r.get("hole_cards")
        board = r.get("board")

        if verbose:
            print(f"Board: {board}, Hole: {hole_cards}, Action: {action_str}")

        state = client.parse_action(action_str)
        if "error" in state:
            raise RuntimeError(f"Error parsing action: {state['error']}")

        # Attach full action string for belief/history use
        state["action_full"] = action_str

        my_action = agent.get_action(state, hole_cards, board)
        if verbose:
            print(f"Agent Action: {my_action}")

        r = client.act(my_action)


def evaluate(agent_name: str, hands: int, username: str = None, password: str = None, verbose: bool = False):
    client = SlumbotClient(username, password)
    agent = ReBeLAgent() if agent_name == "rebel" else RandomAgent()

    winnings_list = []
    showdowns = 0
    folds_before_river = 0

    for i in range(hands):
        if verbose:
            print(f"\n--- Hand {i+1} ---")
        try:
            winnings, stats = play_hand(client, agent, verbose=verbose)
            winnings_list.append(winnings)
            showdowns += 1 if stats.get("showdown") else 0
            folds_before_river += 1 if not stats.get("showdown") else 0
        except Exception as e:
            print(f"Error in hand {i+1}: {e}")
            break
        # Be nice to the API
        time.sleep(0.05)

    total = sum(winnings_list)
    hands_played = len(winnings_list)
    bb100 = (total / 100.0) / (hands_played / 100.0) if hands_played > 0 else 0.0
    stdev = statistics.pstdev(winnings_list) if len(winnings_list) > 1 else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Hands played:      {hands_played}")
    print(f"Total winnings:    {total}")
    print(f"Avg bb/100:        {bb100:.2f}")
    print(f"Std dev (chips):   {stdev:.1f}")
    print(f"Showdowns:         {showdowns}")
    print(f"Folds pre-river:   {folds_before_river}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent vs Slumbot")
    parser.add_argument("--username", type=str, help="Slumbot username")
    parser.add_argument("--password", type=str, help="Slumbot password")
    parser.add_argument("--hands", type=int, default=50, help="Number of hands to play")
    parser.add_argument("--agent", type=str, default="rebel", choices=["rebel", "random"])
    parser.add_argument("--verbose", action="store_true", help="Print per-hand details")
    args = parser.parse_args()

    evaluate(agent_name=args.agent, hands=args.hands, username=args.username, password=args.password, verbose=args.verbose)


if __name__ == "__main__":
    main()


