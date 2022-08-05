from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): & What actions could be? &
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position &Who to call this methods&
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions() #& how the out put look like?&

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current GameState (defined in pacman.py)
    and a proposed action and returns a rough estimate of the resulting successor
    GameState's value.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction. Terminal states can be found by one of the following:
            pacman won, pacman lost or there are no legal moves.

            Don't forget to limit the search depth using self.depth. Also, avoid modifying
            self.depth directly (e.g., when implementing depth-limited search) since it
            is a member variable that should stay fixed throughout runtime.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
                Returns a list of legal actions for an agent
                agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
                Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
                Returns the total number of agents in the game

            gameState.getScore():
                Returns the score corresponding to the current state of the game

            gameState.isWin():
                Returns True if it's a winning state

            gameState.isLose():
                Returns True if it's a losing state

            self.depth:
                The depth to which search should continue

            """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)


        pacman_legal_actions = gameState.getLegalActions(0)
        max_value = float('-inf')
        best_action  = None 

        for action in pacman_legal_actions:   
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0)
            if ((action_value) > max_value ): 
                max_value = action_value
                best_action = action

        return best_action #Returns the final action .

    def Max_Value (self, gameState, depth):
        """This funtion finds max value for max agent"""

        values = []

        if ((depth == self.depth)  or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState) # checks the game is ended or not

        for possible_actions in gameState.getLegalActions(0):
            
            successors =  gameState.generateSuccessor(0, possible_actions)
            values.append(self.Min_Value(successors, 1, depth))

        return max(values)


    def Min_Value (self, gameState, agent_index, depth):
        """ For the MIN Players or Agents  """

        if (len(gameState.getLegalActions(agent_index)) == 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        if (agent_index < gameState.getNumAgents() - 1):
            values2 = []
            for possible_actions in gameState.getLegalActions(agent_index):
                successors =  gameState.generateSuccessor(agent_index, possible_actions)
                values2.append(self.Min_Value(successors, agent_index + 1, depth)) 

            return min(values2)

        else:  
            values3 = []
            for possible_actions in gameState.getLegalActions(agent_index):
                successors =  gameState.generateSuccessor(agent_index, possible_actions)
                values3.append(self.Max_Value(successors, depth + 1)) 

            return min(values3)           
            
# END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """

    def getAction(self, gameState: GameState) -> str:
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
        pacmanIndex = 0
        return self.MyAlphaBeta(1, pacmanIndex, gameState, -99999, 99999)

    def MyAlphaBeta(self, depth, agentIndex, gameState, alpha, beta):

        if (gameState.isWin() or gameState.isLose() or depth > self.depth):
            return self.evaluationFunction(gameState)

        value = []  
        possible_actions = gameState.getLegalActions(agentIndex) 

        for action in possible_actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if((agentIndex+1) >= gameState.getNumAgents()):
                ret = self.MyAlphaBeta(depth+1, 0, successor, alpha, beta)
            else:
                ret = self.MyAlphaBeta(depth, agentIndex+1, successor, alpha, beta)

            if(agentIndex == 0 and ret > beta):
                return ret
            if (agentIndex > 0 and ret < alpha):
                return ret

            if (agentIndex == 0 and ret > alpha):
                alpha = ret

            if (agentIndex > 0 and ret < beta):
                beta = ret

            value += [ret]
        if agentIndex == 0:
            if(depth == 1): 
                maxscore = max(value)
                length = len(value)
                for i in range(length):
                    if (value[i] == maxscore):
                        return possible_actions[i]
            else:
                best_value = max(value)

        elif agentIndex > 0: 
            best_value = min(value)

        return best_value
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 3)
    """

    def getAction(self, gameState: GameState) -> str:
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        possibleActions = gameState.getLegalActions(0)
        action_scores = [self.expectimax(0, 0, gameState.generateSuccessor(0, action)) for action
                            in possibleActions]
        best_action = max(action_scores)
        max_index_list = []
        for index in range(len(action_scores)):
            if action_scores[index] == best_action:
                max_index_list.append(index)
        chosenIndex = random.choice(max_index_list)
        return possibleActions[chosenIndex]

    def expectimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            value4 = []
            for possible_actions in gameState.getLegalActions(0):
                value4.append(self.expectimax(1, depth, gameState.generateSuccessor(agent, possible_actions)))
            return max(value4)
        else:  # min for ghosts
            next_agent = agent + 1  # get the next agent
            if gameState.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:  # increase depth 
                depth += 1
            return sum(self.expectimax(next_agent, depth, gameState.generateSuccessor(agent, action)) for action in
                       gameState.getLegalActions(agent)) / len(gameState.getLegalActions(agent))
    
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
