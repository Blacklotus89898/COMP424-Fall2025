import numpy as np
import math
import copy
import random
import time
from agents.agent import Agent
from helpers import random_move, get_valid_moves, execute_move
from store import register_agent

class SearchTimeout(Exception):
    pass

class MCTSNode:
    def __init__(self, board, parent=None, move_from_parent=None, player_just_moved=None):
        self.board = board
        self.parent = parent
        self.move = move_from_parent
        self.player_just_moved = player_just_moved
        self.children = []
        self.visits = 0
        self.wins = 0.0 
        # Get moves for the NEXT player
        self.untried_moves = [] 
        # (We populate untried_moves on demand to save time)
        # Create a transposition table

    def uct_select_child(self, c_param=1.41):
        # Select the child with the highest UCT score
        best_child = None
        best_score = float("-inf")
        
        # We want to optimize from the perspective of the player at this node
        for child in self.children:
            if child.visits == 0:
                return child
            
            # Exploitation: win rate
            exploitation = child.wins / child.visits
            
            # Exploration: standard UCT formula
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

# TODO:
# - Heurisitics to consider, the evaluation function, the simulation policy, moves classification
# - Optimize deepcopy usage, transposition tables
# - Fine-tune parameters: exploration constant, simulation depth, switching ratio

@register_agent("steve_agent")
class SteveAgent(Agent):
    def __init__(self):
        super(SteveAgent, self).__init__()
        self.name = "SteveAgent"
        self.autoplay = True
        self.time_limit = 1.98 # seconds per move 

    def step(self, board, color, opponent):
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        # Calculate board fullness
        total_squares = board.size
        filled_squares = np.count_nonzero(board)
        fill_ratio = filled_squares / total_squares

        # STRATEGY PIVOT:
        # Use MCTS for the first 85% of the game.
        # Use Minimax only for the absolute endgame (last 15%) or very small boards.
        use_minimax = (fill_ratio < 30 or fill_ratio > 0.70 ) or (board.shape[0] <= 4)

        use_minimax = True
        if use_minimax:
            return self.run_iterative_minimax(board, color, opponent, end_time)
        else:
            return self.run_mcts(board, color, opponent, end_time)

    # ====================================================
    # STRATEGY 1: OPTIMIZED MCTS
    # ====================================================
    def run_mcts(self, board, color, opponent, end_time):
        # Root node represents the state BEFORE we make a move. 
        # So the "player_just_moved" is the opponent.
        root = MCTSNode(board=board, player_just_moved=opponent)
        root.untried_moves = get_valid_moves(board, color)

        if not root.untried_moves:
            return None

        # Main MCTS Loop
        while time.time() < end_time:
            node = root
            curr_player = color

            # 1. SELECTION & 2. EXPANSION
            # Traverse down until we find a node that is not fully expanded
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                curr_player = opponent if curr_player == color else color

            # If this node has untried moves, expand one
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                
                # OPTIMIZATION: Use numpy copy instead of deepcopy
                sim_board = node.board.copy() 
                execute_move(sim_board, move, curr_player)
                
                curr_player_just_moved = curr_player
                curr_player = opponent if curr_player == color else color # Next player
                
                new_node = MCTSNode(sim_board, parent=node, move_from_parent=move, player_just_moved=curr_player_just_moved)
                new_node.untried_moves = get_valid_moves(sim_board, curr_player)
                
                node.untried_moves.remove(move)
                node.children.append(new_node)
                node = new_node
            
            # 3. SIMULATION (Rollout)
            # Simulate result from the perspective of the player who made the move at 'node'
            result = self.simulate_game(node.board, curr_player, color, opponent)

            # 4. BACKPROPAGATION
            while node is not None:
                node.visits += 1
                # If the result (1 for Hero Win) helps the player who just moved, add to wins.
                if node.player_just_moved == color:
                    node.wins += result
                else:
                    # If opponent moved, and result is 0 (Hero Loss), that's good for opponent (1 point).
                    # If result is 1 (Hero Win), that's bad for opponent (0 points).
                    node.wins += (1 - result)
                node = node.parent

        if not root.children:
            return random.choice(get_valid_moves(board, color))

        # Return the move that was visited the most (Robust Child)
        return max(root.children, key=lambda c: c.visits).move

    def simulate_game(self, board, current_turn, hero_color, villain_color):
        """
        Simulates X moves ahead using a GREEDY / HEURISTIC approach.
        """
        sim_board = board.copy() 
        curr = current_turn
        
        # We keep depth small because sorting moves is expensive
        depth = 3
        
        for _ in range(depth):
            moves = get_valid_moves(sim_board, curr)
            if not moves:
                break
            
            # --- MODIFIED LOGIC STARTS HERE ---
            # Instead of random.choice(moves), we evaluate them.
            scored_moves = []
            
            for move in moves:
                # We must copy to test the move
                test_board = sim_board.copy()
                execute_move(test_board, move, curr)
                
                # FAST HEURISTIC for Simulation (Speed is critical here!)
                # We cannot use the full 'heuristic' function because calculating mobility 
                # (get_valid_moves for opponent) inside a loop inside a simulation is O(N^2) and too slow.
                p_count = np.count_nonzero(test_board == curr)
                
                # Add corner bonus (cheap to calculate)
                n = test_board.shape[0]
                corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
                corner_score = sum(10 for (r, c) in corners if test_board[r, c] == curr)
                
                score = p_count + corner_score
                scored_moves.append((score, move))
            
            # Sort moves by score descending (best moves first)
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            
            # CRITICAL: MCTS requires some randomness (Stochastic). 
            # If we strictly pick index [0], the simulation becomes deterministic and MCTS breaks.
            # We pick randomly from the Top 3 moves.
            top_k = min(len(scored_moves), 7)
            best_candidates = [m for s, m in scored_moves[:top_k]]
            move = random.choice(best_candidates)
            # --- MODIFIED LOGIC ENDS HERE ---
            
            execute_move(sim_board, move, curr)
            curr = hero_color if curr == villain_color else villain_color

        return self.evaluate_mcts_result(sim_board, hero_color, villain_color)

    def evaluate_mcts_result(self, board, hero, villain):
        # Normalized score between 0 and 1
        score = self.heuristic(board, hero, villain)
        
        # Sigmoid-ish squash to 0-1
        try:
            return 1 / (1 + math.exp(-0.1 * score))
        except OverflowError:
            return 1.0 if score > 0 else 0.0

    # ====================================================
    # STRATEGY 2: MINIMAX + Pruning + Iterative Deepening + Timeout
    # ====================================================
    def run_iterative_minimax(self, board, color, opponent, end_time):
        legal_moves = get_valid_moves(board, color)
        if not legal_moves:
            return None
        
        best_move = random.choice(legal_moves)
        max_depth_reached = 0
        
        try:
            for depth in range(1, 100):
                current_best = None
                current_score = float("-inf")
                alpha = float("-inf")
                beta = float("inf")

                for move in legal_moves:
                    sim_board = board.copy()
                    execute_move(sim_board, move, color)
                    val = self.minimax(sim_board, depth - 1, opponent, color, opponent, alpha, beta, end_time)
                    if val > current_score:
                        current_score = val
                        current_best = move
                    alpha = max(alpha, current_score)
                
                best_move = current_best
                max_depth_reached = depth   # ✅ update after successfully finishing this depth

        except SearchTimeout:
            pass
        
        print(f"Max depth reached: {max_depth_reached}")  # ✅ print after search ends
        return best_move


    # Minimax with Alpha-Beta Pruning
    def minimax(self, board_state, depth, current_player, hero, villain, alpha, beta, end_time):
        if time.time() >= end_time: raise SearchTimeout()
        
        legal = get_valid_moves(board_state, current_player)
        if depth == 0 or not legal:
            return self.heuristic(board_state, hero, villain)

        if current_player == hero:
            max_eval = float("-inf")
            for move in legal:
                # Optimization: Use numpy copy
                next_board = board_state.copy()
                execute_move(next_board, move, current_player)
                eval_val = self.minimax(next_board, depth - 1, villain, hero, villain, alpha, beta, end_time)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal:
                next_board = board_state.copy()
                execute_move(next_board, move, current_player)
                eval_val = self.minimax(next_board, depth - 1, hero, hero, villain, alpha, beta, end_time)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val) 
                if beta <= alpha: break
            return min_eval

    # ====================================================
    # MASTER HEURISTIC Can be improved Further with tests
    # ====================================================
    def heuristic(self, board, color, opponent):
        # 1. PIECE DIFFERENCE
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count
        
        # 2. CORNER CONTROL
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5
        
        # 3. MOBILITY (Computationally expensive, used only in Master Heuristic, NOT in simulation)
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        
        return score_diff + corner_bonus + mobility_penalty