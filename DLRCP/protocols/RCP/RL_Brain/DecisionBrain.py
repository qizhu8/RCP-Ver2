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
        loglevel=LOGLEVEL) -> None:
        self.initLogger(loglevel)

        self.globalEvalOn = False  # True: ignore greedy random policy

        self.loss = sys.maxsize
        self.convergeLossThresh = convergeLossThresh
        self.isConverge = False  # whether the network meets the convergence condition

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_default = epsilon

    def initLogger(self, loglevel):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
            sh.setFormatter(formatter)

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
        self.epsilon = 1 - (1-self.epsilon)*self.epsilon_decay

    def chooseMaxQAction(self, state) -> int:
        """
        Get the Q values of the given state for all actions, 
        and take the action that counts for the maximum Q.
        """
        raise NotImplementedError

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
        