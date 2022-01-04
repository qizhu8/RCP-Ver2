import numpy as np
import logging
import random
import sys


class DecisionBrain(object):
    LOGLEVEL = logging.INFO

    def __init__(self, 
        convergeLossThresh:float=1.0, 
        epsilon=0.95,       
        epsilon_decay=0.99,
        loglevel=LOGLEVEL,
        createLogFile=False) -> None:
        self.initLogger(loglevel, createLogFile)

        self.globalEvalOn = False  # True: ignore greedy random policy

        self.loss = sys.maxsize
        self.convergeLossThresh = convergeLossThresh
        self.isConverge = False  # whether the network meets the convergence condition

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_default = epsilon

    def initLogger(self, loglevel, create_file=False):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)
        
        formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)

        if create_file:
            # create file handler for logger.
            fh = logging.FileHandler(type(self).__name__+'.log')
            fh.setLevel(loglevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def loadModel(self, modelFile):
        """load a previously save modelfile"""
        raise NotImplementedError

    def saveModel(self, modelFile):
        """save the current prediction model to a file"""
        raise NotImplementedError
    
    def chooseAction(self, state: np.ndarray, evalOn=False, baseline_Q0=None):
        """
        choose action based on maximum Q with probability 1-epsilon, and random action with probability epsilon
        """

        # epsilon greedy
        if evalOn or self.globalEvalOn or np.random.uniform() < self.epsilon:
            # actionRewards if of shape 1 x nAction
            action = self.chooseMaxQAction(state, baseline_Q0=baseline_Q0)
            self.logger.debug("Q-base action {action}.".format(action=action))
        else:
            action = random.randint(0, self.nActions-1)
            self.logger.debug("Random action {action}.".format(action=action))
        return action

    def resetEpsilon(self):
        """reset epsilon back to its original value."""
        self.epsilon = self.epsilon_default

    def decayEpsilon(self):
        """Increase epsilon by shrinking its gap to 1.0 by self.epsilon_decay"""
        # if self.epsilon > 0.99:
        #     return
        self.epsilon = 1 - (1-self.epsilon)*self.epsilon_decay

    def chooseMaxQAction(self, state) -> int:
        """
        Get the Q values of the given state for all actions, 
        and take the action that counts for the maximum Q.
        """
        raise NotImplementedError
    
    def _getTxAttempts(self, state):
        return int(state[0])
    
    def _getPktDelay(self, state):
        return int(state[1])
    
    def _getRTT(self, state):
        return float(state[2])

    def _getGamma(self, state):
        return float(state[3])
    
    def _getAvgDelay(self, state):
        return float(state[4])
    
    def _getRTTVar(self, state):
        return float(state[5])

    def _getRTO(self, state):
        return float(state[6])
    

    def digestExperience(self, prevState, action, reward, curState) -> None:
        """
        Digest one piece of experience. 
        Each implementation decide to learn immediately or store to the memory
        """
        raise NotImplementedError

    def learn(self):

        if self.loss < self.convergeLossThresh:
            self.isConverge = True

        if self.loss > 20*self.convergeLossThresh:
            self.isConverge = False
        
        # if self.loss > 100 * self.convergeLossThresh:
            # self.resetEpsilon()
        