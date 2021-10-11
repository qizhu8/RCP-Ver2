import sys
import time
import math
import random
import numpy as np
import logging
import csv
from .DecisionBrain import DecisionBrain


class ThompsonSampling(object):
    def __init__(self, norm: bool = True, mapfunc=None) -> None:
        """
        norm: bool. True, normalize pmf to sum = 1; False, no normalization
        map: a function that maps the pmf to another distribution. 
            Apply before the norm function. E.g. math.exp
        """

        super().__init__()
        self.norm=norm
        self.mapfunc=mapfunc

    def __call__(self, pmf) -> int:
        return self.randIntFromPMF(pmf)

    def randIntFromPMF(self, pmf) -> int:
        """
        Generate a random integer based on the input PMF.
        inputs:
            pmf: list or np.ndarray. The pmf of each category
            norm: bool. True, normalize pmf to sum = 1; False, no normalization
            map: a function that maps the pmf to another distribution. 
                Apply before the norm function
        output:
            int, random integer based on the given pmf
        """
        norm = self.norm
        mapfunc = self.mapfunc
        # copy
        pmf_clean = [p for p in pmf]
        if mapfunc is not None:
            norm = True
            for i in range(len(pmf_clean)):
                pmf_clean[i] = mapfunc(pmf_clean[i])

        if norm:
            s = sum(pmf_clean)
            for i in range(len(pmf_clean)):
                pmf_clean[i] = pmf_clean[i] / s

        if not math.isclose(sum(pmf_clean), 1.0, rel_tol=1e-6):
            raise Exception("Input PMF is not valid (sum={sum}). PMF={pmf}".format(
                sum=sum(pmf_clean), pmf=pmf_clean.__str__()))
            

        randNum = random.random()
        for idx, p in enumerate(pmf_clean):
            if randNum < p:
                return idx
            randNum -= p


class QTable(object):
    """
    A expandable table that stores the Q values for Q-Learning.

    Q tables is of size (nState, nActions), where nState is the number of states, and nActions is the size of the action space. 
    """
    INIT_QTABLE_SIZE = 20

    def __init__(self, nActions: int, nStates: int = INIT_QTABLE_SIZE, autoExpand: bool = True) -> None:
        """
        nActions: number of possible actions
        nStates: number of states
        autoExpand: bool: True, expand the table when the current table is too small.
            The expansion is done automatically when requested to access a Q value that has either state > nStates and/or action > nActions. 
        """
        self.nActions = nActions
        self.nStates = nStates
        self.autoExpand = autoExpand

        # _maxStates and _maxActions are the allocated space for the table.
        # the actual used space may be smaller than that.
        self._maxStates = self.nStates
        self._maxActions = self.nActions

        self.table = np.zeros((self._maxStates, self._maxActions))

    def _extendMoreStates2Hold(self, tgtStates):
        if tgtStates >= self.nStates:
            moreStates = tgtStates - self.nStates + 1
            self.table = np.vstack(
                [self.table, np.zeros((moreStates, self.nActions))])
            self.nStates = tgtStates + 1

    def _extendMoreActions2Hold(self, tgtActions):
        if tgtActions >= self.nActions:
            moreActions = tgtActions - self.nActions + 1
            self.table = np.hstack(
                [self.table, np.zeros((self.nStates, moreActions))])
            self.nActions = tgtActions + 1

    def getQ(self, state, action=None) -> np.ndarray:
        """
        Get Q[state, :]. 
        Return None if state >= self.nStates.

        output:
            Q[state, :]: ndarray(nActions, )
        """
        if state < self.nStates:
            if action:
                return self.table[state, action] if action < self.nActions else None
            else:
                return self.table[state, :]

        return np.zeros((self.nActions,))

    def setQ(self, state, action, Qval):
        if self.autoExpand:
            # extend table if needed
            self._extendMoreStates2Hold(state)
            self._extendMoreActions2Hold(action)
        if state < self.nStates and action < self.nActions:
            self.table[state, action] = Qval
            return True
        return False

    def saveToCSV(self, filename):
        np.savetxt(filename, self.table, delimiter=",")

    def loadFromCSV(self, filename):
        self.table = np.loadtxt(filename, delimiter=",")


class Q_Brain(DecisionBrain):
    """
    This is the traditional Q-Learning algorithm where state s is an integer.
    """

    def __init__(self,
                 nActions: int,                 # dimension of the action space
                 epsilon: float = 0.95,         # greedy policy parameter
                 eta: float = 0.9,              # reward discount
                 epsilon_decay: float = 0.99,   # the decay of greedy policy parameter, epsilon
                 # method to choose action. e.g. "argmax" or "ThompsonSampling"
                 decisionMethod: str = "argmax",
                 decisionMethodArgs: dict = {},  # support parameters
                 loglevel: int = DecisionBrain.LOGLEVEL,
                 ) -> None:

        super().__init__(loglevel)

        self.nActions = nActions

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.decisonMethodObj = self._parseDecisionMethod(decisionMethod, decisionMethodArgs)

        self.eta = eta

        self.QTable = QTable(nActions=self.nActions)

        self.loss = sys.maxsize

    def _parseDecisionMethod(self, decisionMethod:str, args:dict={}):
        def _initArgmax(args):
            return np.argmax
        
        def _initThompsonSampling(args):
            if "norm" not in args:
                norm = True
            if "mapfunc" not in args:
                mapfunc = None
            
            return ThompsonSampling(norm=norm, mapfunc=mapfunc)

        supportMethods = {"argmax": _initArgmax, "thompsonsampling": _initThompsonSampling}
        decisionMethod = decisionMethod.lower().strip()
        if decisionMethod not in supportMethods:
            self.logger.info("Input decision method", decisionMethod, " is not supported. We have changed it back to 'argmax'. Choose one from ", supportMethods, " if needed")
            decisionMethod = "argmax"

        return supportMethods[decisionMethod]


    def _parseState(self, state):
        """Extract only the number of retransmission attempts from a full packet state"""
        return int(state[0])

    def chooseMaxQAction(self, state):
        # state = [txAttempts, delay, RTT, packetLossHat, averageDelay]
        state = self._parseState(state)
        qVals = self.QTable.getQ(state)
        # method 1 - Pure Q-based method
        action = qVals.argmax()

        # method 2 - Thompson sampling based method
        # action = ThompsonSampling.randIntFromPMF(
        #     pmf=qVals, norm=True, map=None)
        return action

    def digestExperience(self, prevState, action, reward, curState) -> None:
        """
        For traditional Q learning, it learns from the experience immediately.
        No memory storage.
        """
        prevState, curState = self._parseState(
            prevState), self._parseState(curState)
        self.learnReward(prevState, action, reward, curState)

    def learnReward(self, prevState, action, reward, newState):
        """
        learnReward function is called when the packet reaches the final state (delivered or dropped). 
        """
        nextStateQ_max = max(self.QTable.getQ(state=newState))
        prevStateQ_new = (1-self.eta) * nextStateQ_max + reward

        self.loss = np.abs(self.QTable.getQ(prevState)
                           [action] - prevStateQ_new)

        self.QTable.setQ(prevState, action, prevStateQ_new)
        self.logger.debug("learning (S, A, r, S'): ({s_old}, {a}, {r}, {s_new})".format(
            s_old=prevState, a=action, r=reward, s_new=newState
        ))
        super().learn()

    def loadModel(self, modelFile):
        self.QTable.loadFromCSV(modelFile)
        self.logger.info("Load Q Table from csv file", modelFile)

    def saveModel(self, modelFile):
        self.QTable.saveToCSV(modelFile)
        self.logger.info("Save Q Table to csv file", modelFile)


if __name__ == "__main__":
    import math

    def testThompson(pmf, mapFunc, nSamples=100000):
        thomp = ThompsonSampling(norm=False, mapfunc=mapFunc)

        randomNumbers = [thomp(pmf) for _ in range(nSamples)]
        if mapFunc is not None:
            probs = [mapFunc(p) for p in pmf]
        else:
            probs = pmf
        probs_sum = sum(probs)
        probs = [p/probs_sum for p in probs]
        for category in range(len(pmf)):
            counts = randomNumbers.count(category)
            mean = counts / len(randomNumbers)

            print("int {category}: {counts} / {totalNum} : Prob: {mean} / {prob}".format(
                category=category, counts=counts, totalNum=len(randomNumbers),
                mean=mean, prob=probs[category]))

    print("=====Test our random number=====")
    print("=====No Mapping=====")
    random.seed(0)
    pmf = [.1, .4, .3, .2]
    testThompson(pmf=pmf, mapFunc=None, nSamples=100000)

    print("=====Exp Mapping=====")
    random.seed(0)
    pmf = [1, 4, 3, 2]
    def mapFunc(x): return math.exp(x)
    testThompson(pmf=pmf, mapFunc=math.exp, nSamples=100000)

    print("=====Wrong Mapping=====")
    print("Expect to see a non-valid pmf exception")
    random.seed(0)
    pmf = [.1, .4, .3, .9]
    testThompson(pmf=pmf, mapFunc=None, nSamples=100000)
