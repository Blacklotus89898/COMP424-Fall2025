# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.time_limit = 1.975
    # self.mcts_root = None # was used to reuse the MCTS tree between moves, if needed
    
    #  Transposition Table Scheme
    # Stores: { board_bytes: (score, depth, flag, best_move) }, schema recommended by Russell & Norvig
    self.tt = {} 


  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()

    end_time = start_time + self.time_limit
    
    # Memory Management: Clear TT if it gets too huge to prevent crashes
    if len(self.tt) > 200000:
        self.tt = {}

    # total_squares = board.size
    # filled_squares = np.count_nonzero(board)
    # fill_ratio = filled_squares / total_squares

    # Strategy Switcher
    # use_minimax = (fill_ratio < 0.2 or fill_ratio > 0.85) or (board.shape[0] <= 4)
    use_minimax = True
    if use_minimax:
        self.mcts_root = None
        result = self.run_iterative_minimax(chess_board, player, opponent, end_time)

    print("My AI's turn took ", time.time() - start_time, "seconds.")
    return result
    # else:
        # return self.run_mcts(board, player, opponent, end_time)
        # MCTS did not perform well in testing, so disabled for now

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)

# ====================================================
  # STRATEGY 2: MINIMAX + Pruning + Iterative Deepening + Transposition Table
  # ====================================================
  def run_iterative_minimax(self, board, player, opponent, end_time):
      legal_moves = self.get_prioritized_moves(board, player)
      if not legal_moves:
          return None
      
      best_move = legal_moves[0]
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
                  prev_best_move = self.tt[root_key][3]
                  if prev_best_move in legal_moves:
                      legal_moves.remove(prev_best_move)
                      legal_moves.insert(0, prev_best_move) # Move the move found in previous iteration to front of list

              for move in legal_moves:
                  sim_board = board.copy()
                  execute_move(sim_board, move, player)
                  val = self.minimax(sim_board, depth - 1, opponent, player, opponent, alpha, beta, end_time)
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
      state_key = board_state.tobytes() # Hashing the board state
      original_alpha = alpha # Save for TT flag calculation later

      if state_key in self.tt:
          tt_val, tt_depth, tt_flag, tt_move = self.tt[state_key]
          if tt_depth >= depth: # Only use cache if it explored at least as deep as we want now
              if tt_flag == 'EXACT': # the exact value of this state is known
                  return tt_val
              elif tt_flag == 'LOWERBOUND': # value is at least tt_val, beta cut-off happened
                  alpha = max(alpha, tt_val)
              elif tt_flag == 'UPPERBOUND': # value is at most tt_val, alpha cut-off happened
                  beta = min(beta, tt_val)
              if alpha >= beta:
                  return tt_val

      legal_moves = self.get_prioritized_moves(board_state, current_player)
      if depth == 0 or not legal_moves: # no more depth or no legal_moves moves
          return self.heuristic(board_state, hero, villain)

      # --- 2. TT MOVE ORDERING ---
      if state_key in self.tt: # If the TT has a best move for this state, try it first!
          tt_move = self.tt[state_key][3]
          if tt_move in legal_moves:
              legal_moves.remove(tt_move)
              legal_moves.insert(0, tt_move) # try the TT move first

      best_move_found = None
      
      if current_player == hero:
          max_eval = float("-inf")
          for move in legal_moves:
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
          for move in legal_moves:
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
          tt_flag = 'UPPERBOUND' # value is at most final_val
      elif final_val >= beta:
          tt_flag = 'LOWERBOUND' # value is at least final_val
      
      self.tt[state_key] = (final_val, depth, tt_flag, best_move_found)

      return final_val

  def heuristic(self, board, player, opponent):
      player_count = np.count_nonzero(board == player)
      opp_count = np.count_nonzero(board == opponent)
      score_diff = player_count - opp_count
      
      corner_bonus = 0
      # Hardcode the corners for 7x7 board
      size = board.shape[0] - 1
      if board[0, 0] == player: corner_bonus += 5
      if board[0, size] == player: corner_bonus += 5
      if board[size, 0] == player: corner_bonus += 5
      if board[size, size] == player: corner_bonus += 5
      
      # Calculating mobility for opponent is slow, removed for speed
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
  
class SearchTimeout(Exception):
    pass

# def avram_heuristic(self, board, player, opponent):
#     score = float("-inf")


#     # to see how much player gains from a move
#     ppts_prev = np.count_nonzero(prev_board == p)
#     ppts_sim = np.count_nonzero(sim_board == p)

#     ppts_gain = ppts_sim - ppts_prev
#     ppts_gain


#     # to see how much the opponent gets from the player's move
#     opts_prev = np.count_nonzero(prev_board == o)
#     opts_sim = np.count_nonzero(sim_board == o)

#     opts_gain = opts_sim - opts_prev

#     # comparing the gain of the player with the gain of the opponent
#     # idea is that if players gets some, and opponent loses, the player should
#     # see the opponent losing points as a positive too
#     total_change = ppts_gain - opts_gain


#     if(total_change > score) : score = total_change
