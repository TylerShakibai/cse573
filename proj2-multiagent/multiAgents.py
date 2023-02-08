# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodMin = float('inf')
        nextFood = successorGameState.getFood().asList()
        for food in nextFood:
            foodMin = min(manhattanDistance(newPos, food), foodMin)
        
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        
        return successorGameState.getScore() + 1/foodMin

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agent, depth):
            if agent == 0:
                new_depth = depth - 1
            else:
                new_depth = depth
            
            if agent == 0:
                best_value, best = (-float('inf'), max)
            else:
                best_value, best = (float('inf'), min)
            
            if (new_depth == 0) or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            new_agent = (agent + 1) % state.getNumAgents()
            currAction = None
            actions = state.getLegalActions(agent)
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                valueOfAct, tempAction = minimax(successor, new_agent, new_depth)
                if best(valueOfAct, best_value) == valueOfAct:
                    best_value = valueOfAct
                    currAction = action
            return best_value, currAction

        tempValue, bestAction = minimax(gameState, self.index, self.depth + 1)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, agent, depth, alpha, beta):
            maxBool = (agent == 0)

            if maxBool:
                new_depth = depth - 1
            else:
                new_depth = depth
            
            if (new_depth == 0) or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            new_agent = (agent + 1) % state.getNumAgents()

            if maxBool:
                best_value = -float('inf')
                best = max
            else:
                best_value = float('inf')
                best = min
            
            actions = state.getLegalActions(agent)
            for action in actions:
                successorState = state.generateSuccessor(agent, action)
                valueOfAct, tempAction = alphaBeta(successorState, new_agent, new_depth, alpha, beta)
                if best(best_value, valueOfAct) == valueOfAct:
                    best_value, currAction = valueOfAct, action
                
                if maxBool:
                    if best_value > beta:
                        return best_value, currAction
                    alpha = max(alpha, best_value)
                else:
                    if best_value < alpha:
                        return best_value, currAction
                    beta = min(beta, best_value)

            return best_value, currAction

        tempValue, bestAction = alphaBeta(gameState, self.index, self.depth + 1, -float('inf'), float('inf'))
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(gameState, agent, depth):
            valAction = []

            if depth == self.depth:
                return self.evaluationFunction(gameState), 0
            
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState), 0
            
            if agent == (gameState.getNumAgents() - 1):
                depth += 1
                new_agent = self.index
            else:
                new_agent = agent + 1
            
            actions = gameState.getLegalActions(agent)
            for action in actions:
                if not valAction:
                    currValue = expectiMax(gameState.generateSuccessor(agent, action), new_agent, depth)
                    if(agent != self.index):
                        valAction.append((1/len(gameState.getLegalActions(agent))) * currValue[0])
                        valAction.append(action)
                    else:
                        valAction.append(currValue[0])
                        valAction.append(action)
                else:
                    preValue = valAction[0]
                    currValue = expectiMax(gameState.generateSuccessor(agent, action), new_agent, depth)

                    if agent == self.index:
                        if currValue[0] > preValue:
                            valAction[0] = currValue[0]
                            valAction[1] = action
                    else:
                        valAction[0] = valAction[0] + (1/len(gameState.getLegalActions(agent))) * currValue[0]
                        valAction[1] = action
            return valAction
        
        return expectiMax(gameState, self.index, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodMin = float('inf')
    for food in foodList:
        foodMin = min(foodMin, manhattanDistance(position, food))
    
    ghost_dist = 0
    for ghost in currentGameState.getGhostPositions():
        ghost_dist = manhattanDistance(position, ghost)
        if (ghost_dist < 2):
            return -float('inf')
    
    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    food_remaining = 999999
    caps_remaining = 9999
    food_dist = 99
    winLoss_Adjust = 0
    if currentGameState.isLose():
        winLoss_Adjust -= 9999
    else:
        winLoss_Adjust += 9999
    
    evaluation = 1/(foodLeft + 1) * food_remaining + ghost_dist + 1/(foodMin + 1) * food_dist + 1/(capsLeft + 1) * caps_remaining + winLoss_Adjust
    return evaluation

# Abbreviation
better = betterEvaluationFunction
