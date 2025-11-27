import numpy as np
import math
import copy
import random
import time
from agents.agent import Agent
from helpers import get_valid_moves, execute_move
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
        self.untried_moves = [] 
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    def uct_select_child(self, c_param=1.41):
        best_child = None
        best_score = float("-inf")
        for child in self.children:
            if child.visits == 0:
                return child
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

@register_agent("steve_agent")
class SteveAgent(Agent):
    def __init__(self):
        super(SteveAgent, self).__init__()
        self.name = "SteveAgent"
        self.autoplay = True
        self.time_limit = 1.98
        self.mcts_root = None
        
        # --- NEW: Transposition Table ---
        # Stores: { board_bytes: (score, depth, flag, best_move) }
        self.tt = {} 
        # --------------------------------

    def step(self, board, color, opponent):
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        # Memory Management: Clear TT if it gets too huge to prevent crashes
        if len(self.tt) > 200000:
            self.tt = {}

        total_squares = board.size
        filled_squares = np.count_nonzero(board)
        fill_ratio = filled_squares / total_squares

        # Strategy Switcher
        use_minimax = (fill_ratio < 0.2 or fill_ratio > 0.85) or (board.shape[0] <= 4)
        use_minimax = True
        if use_minimax:
            self.mcts_root = None
            return self.run_iterative_minimax(board, color, opponent, end_time)
        else:
            return self.run_mcts(board, color, opponent, end_time)

    # ====================================================
    # STRATEGY 1: OPTIMIZED MCTS (Unchanged)
    # ====================================================
    def run_mcts(self, board, color, opponent, end_time):
        root = None
        if self.mcts_root is not None:
            found_child = None
            for child in self.mcts_root.children:
                if np.array_equal(child.board, board):
                    found_child = child
                    break
            if found_child:
                root = found_child
                root.parent = None 
            else:
                root = None
        
        if root is None:
            root = MCTSNode(board=board, player_just_moved=opponent)
            root.untried_moves = self.get_prioritized_moves(board, color)

        max_depth_reached = root.depth

        if not root.untried_moves and not root.children:
            return None

        while time.time() < end_time:
            node = root
            curr_player = color
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                curr_player = opponent if curr_player == color else color

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                sim_board = node.board.copy() 
                execute_move(sim_board, move, curr_player)
                
                curr_player_just_moved = curr_player
                curr_player = opponent if curr_player == color else color
                
                new_node = MCTSNode(sim_board, parent=node, move_from_parent=move, player_just_moved=curr_player_just_moved)
                new_node.untried_moves = self.get_prioritized_moves(sim_board, curr_player)
                
                node.untried_moves.remove(move)
                node.children.append(new_node)
                node = new_node
                if node.depth > max_depth_reached:
                    max_depth_reached = node.depth
            
            result = self.simulate_game(node.board, curr_player, color, opponent)

            while node is not None:
                node.visits += 1
                if node.player_just_moved == color:
                    node.wins += result
                else:
                    node.wins += (1 - result)
                node = node.parent

        if not root.children:
            return random.choice(self.get_prioritized_moves(board, color))

        best_child = max(root.children, key=lambda c: c.visits)
        best_child.parent = None
        relative_depth = max_depth_reached - root.depth
        print(f"MCTS Max Depth Explored: {relative_depth}")
        self.mcts_root = best_child 
        return best_child.move

    def simulate_game(self, board, current_turn, hero_color, villain_color):
        sim_board = board.copy() 
        curr = current_turn
        depth = 3
        for _ in range(depth):
            moves = self.get_prioritized_moves(sim_board, curr)
            if not moves: break
            
            scored_moves = []
            for move in moves:
                test_board = sim_board.copy()
                execute_move(test_board, move, curr)
                p_count = np.count_nonzero(test_board == curr)
                n = test_board.shape[0]
                corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
                corner_score = sum(10 for (r, c) in corners if test_board[r, c] == curr)
                score = p_count + corner_score
                scored_moves.append((score, move))
            
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            top_k = min(len(scored_moves), 7)
            best_candidates = [m for s, m in scored_moves[:top_k]]
            move = random.choice(best_candidates)
            execute_move(sim_board, move, curr)
            curr = hero_color if curr == villain_color else villain_color

        return self.evaluate_mcts_result(sim_board, hero_color, villain_color)

    def evaluate_mcts_result(self, board, hero, villain):
        score = self.heuristic(board, hero, villain)
        try:
            return 1 / (1 + math.exp(-0.1 * score))
        except OverflowError:
            return 1.0 if score > 0 else 0.0

    # ====================================================
    # STRATEGY 2: MINIMAX + Pruning + Iterative Deepening + Transposition Table
    # ====================================================
    def run_iterative_minimax(self, board, color, opponent, end_time):
        legal_moves = self.get_prioritized_moves(board, color)
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

                # Note: We rely on the Recursive Minimax to check the TT for ordering.
                # But we can also check the TT for the root here:
                root_key = board.tobytes()
                if root_key in self.tt:
                    # Move the move found in previous iteration to front of list
                    prev_best_move = self.tt[root_key][3]
                    if prev_best_move in legal_moves:
                        legal_moves.remove(prev_best_move)
                        legal_moves.insert(0, prev_best_move)

                for move in legal_moves:
                    sim_board = board.copy()
                    execute_move(sim_board, move, color)
                    val = self.minimax(sim_board, depth - 1, opponent, color, opponent, alpha, beta, end_time)
                    if val > current_score:
                        current_score = val
                        current_best = move
                    alpha = max(alpha, current_score)
                
                best_move = current_best
                max_depth_reached = depth 

        except SearchTimeout:
            pass
        
        print(f"Minimax Max Depth Reached: {max_depth_reached}")
        return best_move

    def minimax(self, board_state, depth, current_player, hero, villain, alpha, beta, end_time):
        if time.time() >= end_time: raise SearchTimeout()
        
        # --- 1. TT READ ---
        state_key = board_state.tobytes()
        original_alpha = alpha # Save for TT flag calculation later

        if state_key in self.tt:
            tt_val, tt_depth, tt_flag, tt_move = self.tt[state_key]
            # Only use cache if it explored at least as deep as we want now
            if tt_depth >= depth:
                if tt_flag == 'EXACT':
                    return tt_val
                elif tt_flag == 'LOWERBOUND':
                    alpha = max(alpha, tt_val)
                elif tt_flag == 'UPPERBOUND':
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val
        # ------------------

        legal = self.get_prioritized_moves(board_state, current_player)
        if depth == 0 or not legal:
            return self.heuristic(board_state, hero, villain)

        # --- 2. TT MOVE ORDERING ---
        # If the TT has a best move for this state, try it first!
        if state_key in self.tt:
            tt_move = self.tt[state_key][3]
            if tt_move in legal:
                legal.remove(tt_move)
                legal.insert(0, tt_move)
        # ---------------------------

        best_move_found = None
        
        if current_player == hero:
            max_eval = float("-inf")
            for move in legal:
                next_board = board_state.copy()
                execute_move(next_board, move, current_player)
                eval_val = self.minimax(next_board, depth - 1, villain, hero, villain, alpha, beta, end_time)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move_found = move
                
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            
            final_val = max_eval
        else:
            min_eval = float("inf")
            for move in legal:
                next_board = board_state.copy()
                execute_move(next_board, move, current_player)
                eval_val = self.minimax(next_board, depth - 1, hero, hero, villain, alpha, beta, end_time)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move_found = move
                
                beta = min(beta, eval_val)
                if beta <= alpha: break
            
            final_val = min_eval

        # --- 3. TT WRITE ---
        tt_flag = 'EXACT'
        if final_val <= original_alpha:
            tt_flag = 'UPPERBOUND'
        elif final_val >= beta:
            tt_flag = 'LOWERBOUND'
        
        self.tt[state_key] = (final_val, depth, tt_flag, best_move_found)
        # -------------------

        return final_val

    def heuristic(self, board, color, opponent):
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count
        
        # n = board.shape[0]
        corner_bonus = 0
        # Hardcode the corners for 7x7 board (or use dynamic checking)
        size = board.shape[0] - 1
        if board[0, 0] == color: corner_bonus += 5
        if board[0, size] == color: corner_bonus += 5
        if board[size, 0] == color: corner_bonus += 5
        if board[size, size] == color: corner_bonus += 5
        
        # Note: Calculating mobility for opponent is slow, removed for speed
        # opp_moves = len(self.get_prioritized_moves(board, opponent))
        # mobility_penalty = -opp_moves
        
        return score_diff + corner_bonus 

    def get_prioritized_moves(self, board, player):
        raw_moves = get_valid_moves(board, player)
        duplications = []
        jumps = []
        for move in raw_moves:
            r_src = move.row_src
            c_src = move.col_src
            r_dest = move.row_dest
            c_dest = move.col_dest
            dist = max(abs(r_src - r_dest), abs(c_src - c_dest))
            if dist <= 1:
                duplications.append(move)
            else:
                jumps.append(move)
        return duplications + jumps