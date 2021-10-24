import sys
import time
import math
import random
import numpy as np
import logging
import csv
from .DecisionBrain import DecisionBrain
from DLRCP.protocols.utils import AutoRegressEst

class RTQ_Brain(DecisionBrain):
    """
    This is the traditional Q-Learning algorithm where state s is an integer.
    """

    def __init__(self,
                 utilityCalcHandler,            # handler to calculate utility
                 retransMax,                    # maximum allowed retransmission attempts
                 # method to choose action. e.g. "argmax" or "ThompsonSampling"
                 updateFrequency: int = 8,    # period to learn the best s
                 loglevel: int = DecisionBrain.LOGLEVEL,
                 ) -> None:

        # super().__init__(loglevel)
        super().__init__(convergeLossThresh=100,
            epsilon=1, epsilon_decay=1,loglevel=loglevel)


        # nStates here is the number of different states, not the dimension of a state
        assert retransMax > 1, "For RTQ, retransMax must be > 1"
        self.retransMax = retransMax
        self.utilityList = np.zeros((retransMax,))
        self.utilityCalcHandler = utilityCalcHandler

        self.learningCounter = 0
        self.learningPeriod = updateFrequency

        # smoothed channel RTT and pktloss
        self.chRTTEst = AutoRegressEst(0.1)
        self.chPktLossEst = AutoRegressEst(0.1)
        self.s_star = 0

        self.loss = 0
    
    def calcDelvyRate(self, chPktLossRate, rx):
        return 1 - chPktLossRate**rx

    # def calcDelay(self, chPktLossRate, onewayDelay, rx):
    #     return onewayDelay * (-(chPktLossRate**rx)*rx + (1-chPktLossRate**rx)/(1-chPktLossRate)) / ( 1 - chPktLossRate**rx)
    
    def calcDelay(self, gamma, rtt, rxMax):
        rto_div_rtt = 3 # rto = 3 * rtt
        delay = rtt * (1 - (rto_div_rtt * ((rxMax-1)+1)* (gamma**rxMax) + (gamma - gamma ** rxMax)/(1-gamma)))
        return delay

    def _parseState(self, state):
        """Extract only txAttempts, pktLossHat, avgDelay"""
        return int(state[0]), state[2], state[3]

    def chooseMaxQAction(self, state, baseline_Q0=None):
        # state = [txAttempts, delay, RTT, packetLossHat, averageDelay]
        txAttempts, RTT, packetLossHat = self._parseState(state)
        self.chPktLossEst.update(packetLossHat)
        self.chRTTEst.update(RTT)

        if self.learningCounter == 0:
            self.calcBestS()
        self.learningCounter = (self.learningCounter+1) % self.learningPeriod


        # qVals = self.QTable.getQ(state)
        # if baseline_Q0 is not None:
        #     qVals[0] = baseline_Q0
        # # method 1 - Pure Q-based method
        # action = qVals.argmax()

        # # method 2 - Thompson sampling based method
        # # action = ThompsonSampling.randIntFromPMF(
        # #     pmf=qVals, norm=True, map=None)
        # return action

        return txAttempts < self.s_star
    
    def digestExperience(self, prevState, action, reward, curState) -> None:
        "no need for this function"
        return 

    def calcBestS(self):
        smoothedPktLossRate = self.chPktLossEst.getEstVal()
        smoothedDelay = self.chRTTEst.getEstVal()
        for rx in range(1, self.retransMax):
            avgDelay = self.calcDelay(smoothedPktLossRate, smoothedDelay, rx)
            avgDeliveryRate = self.calcDelvyRate(smoothedPktLossRate, rx)
            self.utilityList[rx] = self.utilityCalcHandler(
                delvyRate=avgDeliveryRate,
                avgDelay=avgDelay,
            )
        self.s_star = np.argmax(self.utilityList)
        
        
