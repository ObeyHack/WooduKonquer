from src.util import raiseNotDefined


class Agent:
  """
  An agent must define a getAction method, but may also define the
  following methods which will be called if they exist:
  """
  def __init__(self, action_space, observation_space, index=0):
    self.index = index
    self.action_space = action_space

  def getAction(self, state):
    """
    The Agent will receive a state and must return an action
    :param state: observation from the environment. It is a dictionary containing the following:
    - board: a 2D list representing the board state
    - block_1: a 2D list representing the block state
    - block_2: a 2D list representing the block state
    - block_3: a 2D list representing the block state
    :return: action to take from the action space: Discrete(243). It is an integer as follows:
    - 0~80 : use block_1
    -- Place block_1 at ((action-0) // 9, (action-0) % 9)
    - 81~161 : use block_2
    -- Place block_2 at ((action-81) // 9, (action-81) % 9)
    - 162~242 : use block_3
    -- Place block_3 at ((action-162) // 9, (action-162) % 9)
    """
    raiseNotDefined()