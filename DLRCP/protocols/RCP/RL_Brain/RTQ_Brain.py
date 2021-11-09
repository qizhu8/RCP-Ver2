import sys
import time
import math
import random
import numpy as np
import logging
import csv
from .DecisionBrain import DecisionBrain
from DLRCP.protocols.utils import AutoRegressEst
import DLRCP.theoreticalAnalysis as theoTool


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
                 createLogFile: bool = False,
                 ) -> None:

        # super().__init__(loglevel)
        super().__init__(convergeLossThresh=100,
                         epsilon=1, epsilon_decay=1, loglevel=loglevel, createLogFile=createLogFile)

        # nStates here is the number of different states, not the dimension of a state
        assert retransMax > 1, "For RTQ, retransMax must be > 1"
        self.retransMax = retransMax
        self.utilityList = np.zeros((retransMax,))
        self.utilityCalcHandler = utilityCalcHandler

        self.learningCounter = 0
        self.learningPeriod = updateFrequency

        # smoothed channel RTT and pktloss
        self.chRTTEst = AutoRegressEst(0.1)
        self.chRTOEst = AutoRegressEst(0.1)
        self.chRTTVarEst = AutoRegressEst(0.1)
        self.chPktLossEst = AutoRegressEst(0.1)
        self.s_star = 0

        self.loss = 0

    # def calcDelvyRate(self, chPktLossRate, rx):
    #     return 1 - chPktLossRate**rx

    # def calcDelay(self, gamma, rtt, rto, rxMax):
    #     numerator = rtt + rto * (gamma * (1-gamma ** (rxMax-1)))/(1-gamma) - (rtt + (rxMax-1)*rto)*(gamma**rxMax)
    #     denominator = 1 - gamma ** rxMax
    #     return numerator / denominator

    def _parseState(self, state):
        """Extract only txAttempts, pktLossHat, avgDelay"""
        return int(state[0]), state[2], state[3], state[5], state[6]

    def chooseMaxQAction(self, state, baseline_Q0=None):
        # state = [txAttempts, delay, RTT, packetLossHat, averageDelay]
        txAttempts, RTT, packetLossHat, RTTVar, RTO = self._parseState(state)
        self.chPktLossEst.update(packetLossHat)
        self.chRTTEst.update(RTT)
        self.chRTTVarEst.update(RTTVar)
        self.chRTOEst.update(RTO)

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
        self.logger.debug("exp:{prevState},{action},{reward},{newState}".format(
            prevState=prevState, action=action, reward=reward, newState=curState
        ))
        return

    def calcBestS(self):
        smoothedPktLossRate = self.chPktLossEst.getEstVal()
        smoothedDelay = self.chRTTEst.getEstVal()
        smoothedRTO = self.chRTOEst.getEstVal()
        for rx in range(1, self.retransMax):
            avgDelay = theoTool.calc_delay_expect(smoothedPktLossRate, smoothedDelay, smoothedRTO, rx)
            avgDeliveryRate = theoTool.calc_delvy_rate_expect(smoothedPktLossRate, rx)
            self.utilityList[rx] = self.utilityCalcHandler(
                delvyRate=avgDeliveryRate,
                avgDelay=avgDelay,
            )
        self.s_star = np.argmax(self.utilityList)

    def saveModel(self, modelFile):
        with open(modelFile, 'w') as f:
            f.writelines("pktLossRate,{}\n".format(
                self.chPktLossEst.getEstVal()))
            f.writelines("delay,{}\n".format(self.chRTTEst.getEstVal()))
            f.writelines("delay_var,{}\n".format(self.chRTTVarEst.getEstVal()))
            f.writelines("s*,{}\n".format(self.s_star))

        self.logger.info("Save Q Table to csv file"+modelFile)
