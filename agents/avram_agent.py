# Author : Avram Bidi 261171284

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

def do_minimax(self, board, p : Agent, o : Agent):
  """
  params
  self
  board := chess_board
  p := player
  o := opponent

  
  Do a form of mini max to find optimal move.
  """


  '''Attempt 0
  # get the moves to look at
  list_moves = get_valid_moves(board, p)

  # print(len(list_moves))

  scores = []

  for i in list_moves:
    temp_board = deepcopy(board)

    execute_move(temp_board, i, p)

    second_list = get_valid_moves(temp_board, o)

    # return max_move
  return list_moves[scores.index(max(scores))]
  '''


  """ attempt 1

  # get the moves to look at
  list_moves = get_valid_moves(board, p)

  # print(len(list_moves))

  scores = []

  for i in list_moves:
    temp_board = deepcopy(board)

    execute_move(temp_board, i, p)

    second_list = get_valid_moves(temp_board, o)

    min_scores = []
    for j in second_list:
      second_temp = deepcopy(temp_board)

      execute_move(second_temp, j, o)

      min_scores.append(np.sum(temp_board == p))

    min_move = min_scores.index(min(min_scores))

    scores.append(min_move)
  get the max move for p considering the min move from o
    scores.append(np.sum(temp_board == p))
  """


  '''Attempt 2
  
  '''
  # 1. get P moves and decide on one
  # 2. get O moves and try to figure our wich one they'll pick
  # 3. get another set of moves for P to figure out which 
  # opponent's move would be least costly and to pick 1st move
  # accordingly


  return None

@register_agent("avram_agent")
class AvramAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(AvramAgent, self).__init__()
    self.name = "a_agent"
    self.timelimit = 1.9

  def evaluate_board(self, sim_board, prev_board, p, o):
    """
    Evaluate the board state based on multiple factors.

    Parameters:
    - board: 2D numpy array representing the game board.
    - p: given player making a move
    - o: opponent of p

    Returns:
    - int: The evaluated score of the board.
    """

    score = float("-inf")

    # # temporary for testing
    # if (np.count_nonzero(sim_board == p) > score) : score = np.count_nonzero(sim_board == p)


    # to see how much player gains from a move
    ppts_prev = np.count_nonzero(prev_board == p)
    ppts_sim = np.count_nonzero(sim_board == p)

    ppts_gain = ppts_sim - ppts_prev
    ppts_gain


    # to see how much the opponent gets from the player's move
    opts_prev = np.count_nonzero(prev_board == o)
    opts_sim = np.count_nonzero(sim_board == o)

    opts_gain = opts_sim - opts_prev
    
    # comparing the gain of the player with the gain of the opponent
    # idea is that if players gets some, and opponent loses, the player should
    # see the opponent losing points as a positive too
    total_change = ppts_gain - opts_gain

    
    

    if(total_change > score) : score = total_change

    return score
    
    

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

    # get legal moves 
    legal_moves = get_valid_moves(chess_board, player)

    if not legal_moves:
      return None  # No valid moves available, pass turn

    # attempt doing some sort of mini-max perchance??
    # chosen_move = do_minimax(self, chess_board, player, opponent)

    chosen_move = None
    chosen_score = float("-inf")

    for move in legal_moves:
      sim_board = deepcopy(chess_board)
      execute_move(sim_board, move, player)

      score = self.evaluate_board(sim_board, chess_board, player, opponent)

      if (score > chosen_score):
        chosen_score = score
        chosen_move = move



    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    return chosen_move

    # this is just to test the learner funcs lol
    # return valid_moves[0]


    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)

