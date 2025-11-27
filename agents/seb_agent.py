# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, get_directions, get_two_tile_directions

@register_agent("seb_agent")
class SebAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """


  def __init__(self):
    super(SebAgent, self).__init__()
    self.name = "SebAgent"
    self.turn = 0
    self.initial_empty_tiles = 0
    self.curr_empty_tiles = 0
    self.chess_board = None
    self.me = -1
    self.opp = -1
    

  # def is_duplication(src:tuple, dest:tuple):
  #   return (abs(dest[0] - src[0]) < 2) or (abs(dest[1] - src[1]) < 2)

  def get_board_size(self):
    return self.chess_board.shape[0]
  
  
  def get_disk_positions(self, player):
    board_size = self.get_board_size()
    postions = []

    for r in range(board_size):
      for c in range(board_size):
        if self.chess_board[r, c] == player:
          postions.append((r,c))

    return postions
  

  def check_center_ctrl(self, board, player, opp):
    board_size = board.shape[0]
    center_board = board[2:board_size-2, 2:board_size-2]
    center_size = center_board.shape[0]

    agent_score = 0
    opp_score = 0

    for r in range(center_size):
      for c in range(center_size):
        if center_board[r, c] == player:
          agent_score += 1

        elif center_board[r, c] == opp:
          opp_score += 1

    return agent_score - opp_score
  

  def check_clusters(self, board, player):
    agent_disks = set()
    cluster_score = 0

    board_size = board.shape[0]
    for r in range(board_size):
      for c in range(board_size):
        found_cluster = False
        curr = (r,c)
        if board[r, c] == player:

          adjs = [(r, c+1),
                  (r+1, c+1),
                  (r+1, c),
                  (r+1, c-1)]
          
          for next in adjs:
            if (next[0] < board_size) and (0 <= next[1] < board_size):
              if board[next[0], next[1]] == player:
                found_cluster = True

                if next not in agent_disks:
                  cluster_score += 1
                  agent_disks.add(next)

          if found_cluster and curr not in agent_disks:
            cluster_score += 1
            

          found_cluster = False
          agent_disks.add(curr)  
          

    return cluster_score



  def check_safety(self, board, player, opp):
    opponent_moves = get_valid_moves(board, opp)
    opponent_dests = set()
    disk_positions = self.get_disk_positions(player)
    directions = get_directions()
    board_size = board.shape[0]
    safe_pieces = 0

    for move in opponent_moves:
      opponent_dests.add(move.get_dest())

    for pos in disk_positions:
      safe = True
      
      for dir in directions:
        curr = (pos[0] + dir[0], pos[1] + dir[1])

        if 0 <= curr[0] < board_size and 0 <= curr[1] < board_size:
          if curr in opponent_dests:
            safe = False

      if safe:
        safe_pieces += 1

    return safe_pieces

  def check_potential_flips(self, board, player, opp):
    moves = get_valid_moves(board, player)
    dests = set()
    directions = get_directions()
    board_size = self.get_board_size()
    flips = 0

    for move in moves:
      dests.add(move.get_dest())

    for pos in dests:
      for dir in directions:
        curr = (pos[0] + dir[0], pos[1] + dir[1])

        if 0 <= curr[0] < board_size and 0 <= curr[1] < board_size:
          if self.chess_board[curr[0], curr[1]] == opp:
            flips += 1

    return flips
  
  def first_half_eval(self, board, player, opp):
    center = self.check_center_ctrl(board, player, opp)
    mobility = len(get_valid_moves(board, player)) - len(get_valid_moves(board, opp))
    cluster = self.check_clusters(board, player)
    safe = self.check_safety(board, player, opp)

    return (5 * center) + (7 * mobility) - (4 * cluster) + (2 * safe)


  def second_half_eval(self, board, player, opp):
    _, p1, p2 = check_endgame(board)
    score = 0

    if player == 1:
      score = p1 - p2

    else:
      score = p2 - p1

    flips = self.check_potential_flips(board, player, opp)
    cluster = self.check_center_ctrl(board, player, opp)
    safe = self.check_safety(board, player, opp)

    return (10 * score) + (6 * flips) - (3 * cluster) + (safe)
  
  
  def eval(self, board, player, opp):
    if self.curr_empty_tiles > (self.initial_empty_tiles / 2):
      return self.first_half_eval(board, player, opp)
    
    else:
      return self.second_half_eval(board, player, opp)


  def minimax(self, board, depth, alpha, beta, isMax, end_time):
    endgame, _, _ = check_endgame(board)

    if depth == 0 or endgame or time.time() >= end_time:
      return self.eval(board, self.me, self.opp)
    
    if isMax:
      best = - np.inf
      moves = get_valid_moves(board, self.me)

      for move in moves:
        next_state = deepcopy(board)
        execute_move(next_state, move, self.me)

        val = self.minimax(next_state, depth-1, alpha, beta, False, end_time)
        best = max(best, val)
        alpha = max(best, alpha)

        if beta <= alpha:
          break

      return best
    
    else:
      best = np.inf
      moves = get_valid_moves(board, self.opp)

      for move in moves:
        next_state = deepcopy(board)
        execute_move(next_state, move, self.opp)

        val = self.minimax(next_state, depth - 1, alpha, beta, True, end_time)
        best = min(best, val)
        beta = min(best, beta)

        if beta <= alpha:
          break
      
      return best


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
    end_time = start_time + 1.8
    # Check how many empty tiles there are initially and currently.
    
    self.me = player
    self.opp = opponent
    self.chess_board = deepcopy(chess_board)
    board_size = chess_board.shape[0]
    self.curr_empty_tiles = 0
    if self.turn == 0:
      for r in range(board_size):
        for c in range (board_size):
          if chess_board[r, c] == 0:
            self.initial_empty_tiles += 1
          
      self.curr_empty_tiles = self.initial_empty_tiles
    
    else:
      for r in range(board_size):
        for c in range(board_size):
          if chess_board[r, c] == 0:
            self.curr_empty_tiles += 1

    best_move = None
    best_val = - np.inf

    moves = get_valid_moves(chess_board, player)

    for move in moves:
      next_state = deepcopy(chess_board)
      execute_move(next_state, move, player)

      val = self.minimax(next_state, 2, -np.inf, np.inf, False, end_time)

      if val > best_val:
        best_val = val
        best_move = move
    
    self.turn += 1
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best_move

