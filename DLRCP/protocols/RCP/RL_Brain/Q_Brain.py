import sys
import time
import math
import random
import numpy as np
import logging
import csv

from DLRCP.protocols.utils import AutoRegressEst
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
            additionalRows = np.ones((moreStates, 1)).dot([self.table[-1, :]])
            self.table = np.vstack(
                [self.table, additionalRows])
            # print("expand states from ", self.nStates, " to ", tgtStates+1)
            self.nStates = tgtStates + 1
            

    def _extendMoreActions2Hold(self, tgtActions):
        if tgtActions >= self.nActions:
            moreActions = tgtActions - self.nActions + 1
            minvalue = np.min(self.table) # treat it as initial value
            # minvalue = 0

            self.table = np.hstack(
                [self.table, minvalue * np.ones((self.nStates, moreActions))])
            self.nActions = tgtActions + 1

    def getQ(self, state, action=None) -> np.ndarray:
        """
        Get Q[state, :]. 
        Return None if state >= self.nStates.

        output:
            Q[state, :]: ndarray(nActions, )
        """
        if action is not None:
            return self.table[state, action] if action < self.nActions and state < self.nStates else 0
        return self.table[state, :] if state < self.nStates else np.zeros((self.nActions,))

    def setQ(self, state, action, Qval, setAColumn=False):
        """
        Set Q values. 
        if setAColumn is false (default), Q[state, action] = Qval
        else Q[:, action] = Qval
        """
        if self.autoExpand:
            # extend table if needed
            self._extendMoreStates2Hold(state)
            self._extendMoreActions2Hold(action)
        if state < self.nStates and action < self.nActions:
            if setAColumn:
                self.table[:, action] = Qval
            else:
                self.table[state, action] = Qval
            return True
        return False
    
    def __str__(self) -> str:
        msg = ""
        for state in range(self.nStates):
            msg += "{state} {q0}\t{q1}\n".format(state=state, q0=self.table[state, 0], q1=self.table[state, 1])
        return msg

    def saveToCSV(self, filename):
        np.savetxt(filename, self.table, delimiter=",", fmt='%f')

    def loadFromCSV(self, filename):
        self.table = np.loadtxt(filename, delimiter=",")


class Q_Brain(DecisionBrain):
    """
    This is the traditional Q-Learning algorithm where state s is an integer.
    """

    def __init__(self,
                 nActions: int,                 # dimension of the action space
                 epsilon: float = 0.95,         # greedy policy parameter
                 epsilon_decay: float = 0.99,   # the decay of greedy policy parameter, epsilon
                 eta: float = 0.9,              # reward discount # for SMDP, this discount is the Laplace-Stieltjes transform of state transition
                 # method to choose action. e.g. "argmax" or "ThompsonSampling"
                 convergeLossThresh=0.01,
                 updateFrequency: int = 1000,    # period to copying evalNet weights to tgtNet
                 decisionMethod: str = "argmax",
                 decisionMethodArgs: dict = {},  # support parameters
                 loglevel: int = DecisionBrain.LOGLEVEL,
                 calcBellmanTimeDiscountHandler=None, # the handler to calculate Bellman time discount
                 createLogFile:bool=False
                 ) -> None:

        # super().__init__(loglevel)
        super().__init__(convergeLossThresh=convergeLossThresh,
            epsilon=epsilon, epsilon_decay=epsilon_decay,loglevel=loglevel,createLogFile=createLogFile)

        self.nActions = nActions


        self.decisonMethodObj = self._parseDecisionMethod(decisionMethod, decisionMethodArgs)

        self.eta = eta

        self.learningCounter = 0
        self.updateFrequencyFinal = updateFrequency

        # nStates here is the number of different states, not the dimension of a state
        self.QTable = QTable(nActions=self.nActions, nStates=2)

        self.calcBellmanTimeDiscountHandler = calcBellmanTimeDiscountHandler
        # self.chRTOEst = AutoRegressEst(0.1)
        self.chRTTEst = AutoRegressEst(0.1)
        self.chRTTVarEst = AutoRegressEst(0.1)
        self.gamma = 0 # packet loss rate

        self.loss = sys.maxsize

        # self.thompsonSampler = ThompsonSampling() # not used 

        # for debug
        self.debugOn = True
        self.debugLog = []


        self.debugLogHead = "prevTxAttemp,prevDelay,prevRTT,prevChPktLoss,prevAvgDelay,prevRTTVar,prevRTO,prevDelvyRate,action,reward,curTxAttemp,curDelay,curRTT,curChPktLoss,curAvgDelay,curRTTVar,curRTO,curDelvyRate"

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
    

    def chooseMaxQAction(self, state, baseline_Q0=None):
        # state = [txAttempts, delay, RTT, packetLossHat, averageDelay, RTTVar, RTO]
        state = self._parseState(state)
        qVals = self.QTable.getQ(state)
        # if baseline_Q0 is not None:
        #     qVals[0] = baseline_Q0 # this operation will change QTable, because this is python, qVals is a pointer
        # method 1 - Pure Q-based method
        action = qVals.argmax()

        return action

    def digestExperience(self, prevState, action, reward, curState) -> None:
        """
        For traditional Q learning, it learns from the experience immediately.
        No memory storage.
        """
        if self.debugOn:
            record = prevState + [action, reward] + curState
            self.debugLog.append(record)

        # self.chRTOEst.update(self._getRTO(curState))
        self.chRTTEst.update(self._getRTT(curState))
        self.chRTTVarEst.update(self._getRTTVar(curState))
        self.gamma = self._getGamma(curState)

        prevState, curState = self._parseState(
            prevState), self._parseState(curState)

        self.learnReward(prevState, action, reward, curState)

        if self.learningCounter > self.updateFrequencyFinal:
            # transfer the weight of the evalNet to tgtNet
            self.decayEpsilon()
            self.learningCounter = 0
        self.learningCounter += 1

        

    def learnReward(self, prevState, action, reward, newState):
        """
        learnReward function is called when the packet reaches the final state (delivered or dropped). 
        """
        # print("learn: ", prevState, action, reward, newState)
        if self.calcBellmanTimeDiscountHandler is not None:
            timeDiscount = self.calcBellmanTimeDiscountHandler(self.chRTTEst.getEstVal(), self.chRTTVarEst.getEstVal(), prevState)
        else:
            timeDiscount = 1

        # debug use
        # if prevState == 2 and action == 1:
        #     print("prev: {prevState} act: {action} reward: {reward} new: {newState}".format(
        #       prevState=prevState, action=action, reward=reward, newState=newState  
        #     ))
       

        if action == 0: # we have ignored the packet, no more gain
            # we want to smooth the reward
            prevStateQ_old = self.QTable.getQ(prevState, action)
            prevStateQ_new = self.eta * prevStateQ_old + (1-self.eta) * reward
            setAColumn = True
            # setAColumn = False
        else:
            # we retransmit the packet and the packet got delivered
            nextAction = np.argmax(self.QTable.getQ(state=newState))
            setAColumn = False

            # """Original Bellman's Equation"""
            # if nextAction == 0: # the only option next time is to drop it, 
            #     prevStateQ_new = timeDiscount * self.QTable.getQ(state=newState, action=0) + reward
            # else:
            #     prevStateQ_new = timeDiscount * self.QTable.getQ(state=newState, action=0) + reward
            
            # gammaFactor = self.gamma

            """The modified Bellman's Equation in paper"""
            if nextAction == 0: # the only option next time is to drop it, 
                prevStateQ_new = self.gamma * self.QTable.getQ(state=newState, action=0) + reward #/ (1-self.gamma)
            else:
                prevStateQ_new = self.gamma * timeDiscount * self.QTable.getQ(state=newState, action=1) + reward # / (1-self.gamma)
            # if nextAction == 0: # the only option next time is to drop it, 
            #     prevStateQ_new = self.gamma * self.QTable.getQ(state=newState, action=0) + reward / (1-self.gamma)
            # else:
            #     prevStateQ_new = self.gamma * timeDiscount * self.QTable.getQ(state=newState, action=1) + reward / (1-self.gamma)

            # print(prevState, self.gamma, timeDiscount, nextStateQ_max, reward, prevStateQ_new)

        self.loss = np.abs(self.QTable.getQ(prevState, action) - prevStateQ_new)

        # print("update Qtable [{}-{}]({}) -> [{}-{}]({})".format(prevState, action, self.QTable.getQ(prevState, action), prevState, action, prevStateQ_new))
        self.QTable.setQ(prevState, action, prevStateQ_new, setAColumn=setAColumn)
        
        # print(self.QTable)

        super().learn()

    def loadModel(self, modelFile):
        self.QTable.loadFromCSV(modelFile)
        self.logger.info("Load Q Table from csv file", modelFile)

    def saveModel(self, modelFile):
        self.QTable.saveToCSV(modelFile)
        self.logger.info("Save Q Table to csv file", modelFile)

        if self.debugOn:
            np.savetxt(modelFile+"debug.csv", np.asarray(self.debugLog), header=self.debugLogHead, delimiter=",")


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
