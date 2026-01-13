import torch
import numpy as np
from .game import LeducRules, GameConstants
from .features import get_features

class Node:
    def __init__(self, history, board_rank, player_to_act, valid_actions):
        self.history = history
        self.board_rank = board_rank
        self.player_to_act = player_to_act
        self.valid_actions = valid_actions
        
        self.regret_sum = {a: np.zeros(GameConstants.NUM_CARDS) for a in valid_actions}
        self.strategy_sum = {a: np.zeros(GameConstants.NUM_CARDS) for a in valid_actions}
        self.strategy = {a: np.zeros(GameConstants.NUM_CARDS) for a in valid_actions}
        
        self.children = {} 
        self.is_terminal = False
        self.is_leaf = False 

class CFRSolver:
    def __init__(self, value_net, iterations=100, device='cpu'):
        self.value_net = value_net
        self.iterations = iterations
        self.device = device
        self.nodes = {} 
        
    def _get_node(self, history, board_rank, bets):
        key = str(history) + "_" + str(board_rank)
        
        if key not in self.nodes:
            is_term_round = LeducRules.is_terminal_round(history)
            
            # Check fold
            if len(history) > 0 and history[-1] == 0:
                node = Node(history, board_rank, -1, [])
                node.is_terminal = True
                self.nodes[key] = node
                return node
                
            if is_term_round:
                if board_rank is not None:
                    # Round 2 end -> Showdown
                    node = Node(history, board_rank, -1, [])
                    node.is_terminal = True
                else:
                    # Round 1 end -> Leaf
                    node = Node(history, board_rank, -1, [])
                    node.is_leaf = True
                
                self.nodes[key] = node
                return node
            
            active = len(history) % 2
            raises = history.count(2)
            valid = LeducRules.get_legal_actions(history, raises)
            
            node = Node(history, board_rank, active, valid)
            self.nodes[key] = node
            
        return self.nodes[key]

    def solve(self, history, board_rank, bets, range_p0, range_p1):
        self.nodes = {} 
        r0 = range_p0.cpu().numpy()
        r1 = range_p1.cpu().numpy()
        
        for i in range(self.iterations):
            self._cfr(history, board_rank, bets, r0, r1)
            
        root = self._get_node(history, board_rank, bets)
        avg_strat = self._get_average_strategy(root)
        
        # Calculate EV of avg strategy
        value = self._compute_ev(history, board_rank, bets, r0, r1, avg_strat)
        return avg_strat, value

    def _cfr(self, history, board_rank, bets, r0, r1):
        node = self._get_node(history, board_rank, bets)
        
        if node.is_terminal:
            return self._get_terminal_payoffs(history, board_rank, bets, r0, r1)
            
        if node.is_leaf:
            return self._get_value_net_payoffs(history, board_rank, bets, r0, r1)
            
        # Regret Matching
        sum_pos_regret = np.zeros(6)
        for a in node.valid_actions:
            pos_r = np.maximum(node.regret_sum[a], 0)
            node.strategy[a] = pos_r
            sum_pos_regret += pos_r
            
        for a in node.valid_actions:
            mask = (sum_pos_regret > 1e-9)
            node.strategy[a][mask] /= sum_pos_regret[mask]
            node.strategy[a][~mask] = 1.0 / len(node.valid_actions)
            
        ev_actions = {}
        if node.player_to_act == 0:
            current_range = r0
            opp_range = r1
        else:
            current_range = r1
            opp_range = r0
            
        for a in node.valid_actions:
            new_hist = history + [a]
            new_bets = bets.copy()
            active = node.player_to_act
            opponent = 1 - active
            
            if a == 1: # Check/Call
                diff = new_bets[opponent] - new_bets[active]
                if diff > 0: new_bets[active] += diff
            elif a == 2: # Bet/Raise
                diff = new_bets[opponent] - new_bets[active]
                amount = 4.0 if board_rank is not None else 2.0
                new_bets[active] += diff + amount
                
            if node.player_to_act == 0:
                next_r0 = r0 * node.strategy[a]
                ev = self._cfr(new_hist, board_rank, new_bets, next_r0, r1)
            else:
                next_r1 = r1 * node.strategy[a]
                ev = self._cfr(new_hist, board_rank, new_bets, r0, next_r1)
            ev_actions[a] = ev
            
        node_ev = {0: np.zeros(6), 1: np.zeros(6)}
        for a in node.valid_actions:
            strat_a = node.strategy[a]
            for p in [0, 1]:
                node_ev[p] += strat_a * ev_actions[a][p]
                
        p = node.player_to_act
        for a in node.valid_actions:
            regret = ev_actions[a][p] - node_ev[p]
            node.regret_sum[a] += regret
            node.strategy_sum[a] += node.strategy[a] * current_range 
            
        return node_ev

    def _compute_ev(self, history, board_rank, bets, r0, r1, strategy_map):
        # Walk the tree using fixed strategy
        node = self._get_node(history, board_rank, bets)
        
        if node.is_terminal:
            return self._get_terminal_payoffs(history, board_rank, bets, r0, r1)
        if node.is_leaf:
            return self._get_value_net_payoffs(history, board_rank, bets, r0, r1)
            
        # Get strategy from map (or uniform if missing)
        # Note: strategy_map is for ROOT only?
        # No, we need strategy for the whole subtree.
        # But 'solve' returns strategy for ROOT.
        # CFR accumulation happens in nodes.
        # So we can just use `node.strategy_sum` to compute average strategy on the fly
        # OR `_get_average_strategy` returns a map of action->prob for THIS node.
        # I need to fetch the average strategy for the *current* node from the `nodes` cache.
        
        # Recalculate average strategy for this node
        avg_strat = self._get_average_strategy(node)
        
        ev_actions = {}
        for a in node.valid_actions:
            new_hist = history + [a]
            new_bets = bets.copy()
            active = node.player_to_act
            opponent = 1 - active
            if a == 1: 
                diff = new_bets[opponent] - new_bets[active]
                if diff > 0: new_bets[active] += diff
            elif a == 2:
                diff = new_bets[opponent] - new_bets[active]
                amount = 4.0 if board_rank is not None else 2.0
                new_bets[active] += diff + amount
                
            if node.player_to_act == 0:
                next_r0 = r0 * avg_strat[a]
                ev = self._compute_ev(new_hist, board_rank, new_bets, next_r0, r1, strategy_map)
            else:
                next_r1 = r1 * avg_strat[a]
                ev = self._compute_ev(new_hist, board_rank, new_bets, r0, next_r1, strategy_map)
            ev_actions[a] = ev
            
        node_ev = {0: np.zeros(6), 1: np.zeros(6)}
        for a in node.valid_actions:
            strat_a = avg_strat[a]
            for p in [0, 1]:
                node_ev[p] += strat_a * ev_actions[a][p]
        return node_ev

    def _get_terminal_payoffs(self, history, board_rank, bets, r0, r1):
        if history[-1] == 0: # Fold
            loser = (len(history) - 1) % 2
            winner = 1 - loser
            rew = LeducRules.get_payoffs_from_bets(bets, winner, folded=True)
            return {0: np.full(6, rew[0]), 1: np.full(6, rew[1])}
            
        # Showdown
        payoffs_0 = np.zeros(6)
        payoffs_1 = np.zeros(6)
        
        # P0
        for c0 in range(6):
            ev = 0
            for c1 in range(6):
                if c0 == c1: continue
                # We need unnormalized r1? 
                # Yes, CFR handles reaches. 
                # If we are at terminal, we sum Payoff * r_opp[c_opp].
                rew = LeducRules.get_payoffs_from_bets(bets, LeducRules.get_winner(c0, c1, board_rank))
                ev += r1[c1] * rew[0]
            payoffs_0[c0] = ev
            
        # P1
        for c1 in range(6):
            ev = 0
            for c0 in range(6):
                if c0 == c1: continue
                rew = LeducRules.get_payoffs_from_bets(bets, LeducRules.get_winner(c0, c1, board_rank))
                ev += r0[c0] * rew[1]
            payoffs_1[c1] = ev
            
        return {0: payoffs_0, 1: payoffs_1}

    def _get_value_net_payoffs(self, history, board_rank, bets, r0, r1):
        # End of Round 1.
        pot = bets[0] + bets[1]
        inputs = []
        boards = [0, 1, 2]
        
        # Normalize ranges for NN
        # r0, r1 are proportional to reach.
        s0 = r0.sum()
        s1 = r1.sum()
        nr0 = r0 / s0 if s0 > 1e-9 else r0
        nr1 = r1 / s1 if s1 > 1e-9 else r1
        
        for b_rank in boards:
            ft = get_features(
                torch.tensor(nr0, dtype=torch.float32),
                torch.tensor(nr1, dtype=torch.float32),
                b_rank, [], pot
            ).to(self.device)
            inputs.append(ft)
            
        inputs = torch.stack(inputs)
        with torch.no_grad():
            # values shape (3, 12)
            values_pred = self.value_net(inputs).cpu().numpy()
            
        # Aggregate
        # val[p][card] = sum_b P(b | card) * V_pred(b, card)
        
        vals_0 = np.zeros(6)
        vals_1 = np.zeros(6)
        
        for i_b, b_rank in enumerate(boards):
            v_p0 = values_pred[i_b, 0:6]
            v_p1 = values_pred[i_b, 6:12]
            
            # P(Board=b | MyCard=c)
            # 5 unknown cards.
            # If b matches c rank: 1 match left. Prob = 1/5.
            # If b differs c rank: 2 matches left. Prob = 2/5.
            
            for c in range(6):
                rank_c = c // 2
                if rank_c == b_rank:
                    prob = 1.0 / 5.0
                else:
                    prob = 2.0 / 5.0
                
                vals_0[c] += prob * v_p0[c]
                vals_1[c] += prob * v_p1[c]
                
        return {0: vals_0, 1: vals_1}

    def _get_average_strategy(self, node):
        avg_strat = {}
        for a in node.valid_actions:
            s_sum = node.strategy_sum[a]
            sum_s = np.zeros(6)
            for a2 in node.valid_actions:
                sum_s += node.strategy_sum[a2]
            
            mask = (sum_s > 1e-9)
            prob = np.zeros(6)
            prob[mask] = s_sum[mask] / sum_s[mask]
            prob[~mask] = 1.0 / len(node.valid_actions)
            avg_strat[a] = prob
        return avg_strat
