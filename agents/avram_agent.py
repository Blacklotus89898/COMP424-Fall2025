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
    pts_gain = 1.5 * ppts_gain - 0.9 * opts_gain


    # check move restrictions
    # !!!! this doesn't feel like a very good heuristic...
    p_moves = len(get_valid_moves(prev_board, p))
    o_moves = len(get_valid_moves(prev_board, o))
    p_moves_after = len(get_valid_moves(sim_board, p))
    o_moves_after = len(get_valid_moves(sim_board, o))

    # moves_score = 0.8 * (p_moves_after - p_moves - (o_moves_after - o_moves))
    moves_score = 0.3 * (p_moves - o_moves)


    # corner bonus
    n = prev_board.shape[0]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    corner_bonus = sum(1 for (i, j) in corners if prev_board[i, j] == p) * 5.0


    score = pts_gain + corner_bonus + moves_score

    return score
    

  def doMax(self, sim_board, prev_board, p, o, depth, alpha, beta):
    """
    Does a max evaluation of the given move, so scores from the player's
    perspective.


    """

    # check for end
    if (depth==0 or check_endgame(sim_board)[0] == True):
      return self.evaluate_board(sim_board, prev_board, p, o)

    val = float("-inf")

    moves = get_valid_moves(sim_board, p)
    if not moves:
      return self.doMin(sim_board, prev_board, p, o, depth-1, alpha, beta)
    
    for move in moves:
      next_sim_board = deepcopy(sim_board)
      execute_move(next_sim_board, move, p)

      child_val = self.doMin(next_sim_board, sim_board, p, o, depth-1, alpha, beta)

      val = max(val, child_val)

      alpha = max(alpha, val)
      if (beta <= alpha): break

    return val
  

  def doMin(self, sim_board, prev_board, p, o, depth, alpha, beta):
    """
    Does a min eval from opponent's perspective
    
    """

    # check for end
    if (depth==0 or check_endgame(sim_board)[0] == True):
      return self.evaluate_board(sim_board, prev_board, p, o)

    val = float("+inf")

    moves = get_valid_moves(sim_board, o)
    if not moves:
      return self.doMax(sim_board, prev_board, p, o, depth-1, alpha, beta)
    
    for move in moves:
      next_sim_board = deepcopy(sim_board)
      execute_move(next_sim_board, move, o)

      child_val = self.doMax(next_sim_board, sim_board, p, o, depth-1, alpha, beta)

      val = min(val, child_val)

      beta = min(beta, val)
      if (beta <= alpha): break

    return val

  def stone_count_heuristic(self, sim_board, prev_board, p, o, is_jump : bool,
                            row_src, row_dest, col_src, col_dest):
    score = 0
    s1 = 1.0
    s2 = 0.4
    s3 = 0.7
    s4 = 0.4

    # enemy stones taken
    x1 = np.count_nonzero(sim_board == o) - np.count_nonzero(prev_board == o)

    # player stones around target
    x2 = 3


    # if move is a jump move => 1, else => 0
    # x3 = is_jump

    # each player stone around source tile (if move is jump)
    x4 = 0
    lenr, lenc = prev_board.shape
    if (is_jump and row_dest-1 >= 0 and col_dest-1 >= 0 and
        row_dest+1 <= lenr-1 and col_dest+1 <= lenc-1):
      
      source = prev_board[row_src, col_src]
      targets = prev_board[row_dest-1:row_dest+1 , col_dest-1:col_dest+1]
      
      x4 = np.count_nonzero((targets == source)) - 1

    elif ((row_src == 0 and col_src == 0) or
             (row_src == 0 and col_src == lenc-1) or
             (row_src == lenr-1 and col_src == 0) or
             (row_src == lenr-1 and col_src == lenc-1)):
      x4 = 0.25
    else:
      x4 = 0.5

    formula = s1 * x1 + s2 * x2 + s3 * is_jump - s4 * x4
    return formula
    

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 representws Player 2's discs (Brown).
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
    
    alpha = 0.2
    beta = 0.8

    # attempt doing some sort of mini-max perchance??
    # chosen_move = do_minimax(self, chess_board, player, opponent)

    chosen_move = None
    chosen_score = float("-inf")


    """ BAREBONES
    
    """

    # for move in legal_moves:
    #   sim_board = deepcopy(chess_board)
    #   execute_move(sim_board, move, player)

    #   score = self.evaluate_board(sim_board, chess_board, player, opponent)

    #   if (score > chosen_score):
    #     chosen_score = score
    #     chosen_move = move
    #   sim_board = deepcopy(chess_board)
    #   execute_move(sim_board, move, player)

    #   score = self.evaluate_board(sim_board, chess_board, player, opponent)

    #   if (score > chosen_score):
    #     chosen_score = score
    #     chosen_move = move

    # for move in legal_moves:
    #   sim_board = deepcopy(chess_board)
    #   execute_move(sim_board, move, player)

    #   score = self.evaluate_board(sim_board, chess_board, player, opponent)

    #   if (score > chosen_score):
    #     chosen_score = score
    #     chosen_move = move

    #   legal_moves2 = get_valid_moves(sim_board, opponent)
    #   for move2 in legal_moves2:
    #     sim_board2 = deepcopy(sim_board)
    #     execute_move(sim_board2, move2, opponent)

    #     legal_moves3 = get_valid_moves(sim_board2, player)
    #     for move3 in legal_moves3:
    #       sim_board3 = deepcopy(sim_board2)
    #       execute_move(sim_board3, move3, player)

    #       score = self.evaluate_board(sim_board3, sim_board2, player, opponent)

    #       if (score > chosen_score):
    #         chosen_score = score
    #         chosen_move = move


    """ Stone Counter Heuristic
    
    """

    # for move in legal_moves:
    #   sim_board = deepcopy(chess_board)
    #   execute_move(sim_board, move, player)

    #   # score = self.evaluate_board(sim_board, chess_board, player, opponent)

    #   is_jump = True if (move.col_dest - move.col_src == 2 or
    #                      move.row_dest - move.row_src == 2) else False
    #   score = self.stone_count_heuristic(sim_board, chess_board,
    #                                      player, opponent, is_jump, 
    #                                      move.row_src, move.row_dest, 
    #                                      move.col_src, move.col_dest)

    #   if (score > chosen_score):
    #     chosen_score = score
    #     chosen_move = move


    """ Branch=3 minimax
    
    """
    for move in legal_moves:
      sim_board = deepcopy(chess_board)
      execute_move(sim_board, move, player)

      score = self.doMin(sim_board, chess_board, player, opponent,
                         2, alpha, beta)

      if (score > chosen_score):
        chosen_score = score
        chosen_move = move
    

    


    """ MOVE EVALUATOR
    
    """

    # sim_board1 = deepcopy(chess_board)
    # sim_board2 = deepcopy(sim_board1)
    # chosen_board = None
    # for move in legal_moves:
    #   sim_board2 = deepcopy(sim_board1)
    #   execute_move(sim_board2, move, player)

    #   score = self.doMax(sim_board2, sim_board1, player, opponent)

    #   if (score > chosen_score):
    #     chosen_score = score
    #     chosen_move = move
    #     chosen_board = sim_board2





    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    return chosen_move

    # this is just to test the learner funcs lol
    # return valid_moves[0]


    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)

