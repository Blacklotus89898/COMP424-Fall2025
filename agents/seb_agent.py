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
  

  def check_center_ctrl(self):
    board_size = self.chess_board.shape[0]
    center_board = self.chess_board[2:board_size-2, 2:board_size-2]
    center_size = center_board.shape[0]

    agent_score = 0
    opp_score = 0

    for r in range(center_size):
      for c in range(center_size):
        if center_board[r, c] == self.me:
          agent_score += 1

        elif center_board[r, c] == self.opp:
          opp_score += 1

    return agent_score - opp_score
  

  def check_clusters(self):
    agent_disks = set()
    cluster_score = 0

    board_size = self.chess_board.shape[0]
    for r in range(board_size):
      for c in range(board_size):
        found_cluster = False
        curr = (r,c)
        if self.chess_board[r, c] == self.me:

          adjs = [(r, c+1),
                  (r+1, c+1),
                  (r+1, c),
                  (r+1, c-1)]
          
          for next in adjs:
            if (next[0] < board_size) and (0 <= next[1] < board_size):
              if self.chess_board[next[0], next[1]] == self.me:
                found_cluster = True

                if next not in agent_disks:
                  cluster_score += 1
                  agent_disks.add(next)

          if found_cluster and curr not in agent_disks:
            cluster_score += 1
            

          found_cluster = False
          agent_disks.add(curr)  
          

    return cluster_score



  def check_safety(self):
    opponent_moves = get_valid_moves(self.chess_board, self.opp)
    opponent_dests = set()
    disk_positions = self.get_disk_positions(self.me)
    directions = get_directions()
    board_size = self.get_board_size()
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

  def check_potential_flips(self):
    moves = get_valid_moves(self.chess_board, self.me)
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
          if self.chess_board[curr[0], curr[1]] == self.opp:
            flips += 1

    return flips
  
 


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
    
    # Check how many empty tiles there are initially and currently.
    
    self.me = player
    self.opp = opponent
    self.chess_board = deepcopy(chess_board)
    board_size = chess_board.shape[0]
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

    self.turn += 1
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)

