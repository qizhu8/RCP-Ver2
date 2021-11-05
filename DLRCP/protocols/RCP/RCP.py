from copy import copy
import logging
import numpy as np
from typing import List

from DLRCP.common.packet import Packet, PacketType
from DLRCP.protocols.utils import AutoRegressEst, MovingAvgEst, RTTEst, Window
from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from .RL_Brain import DQN_Brain
from .RL_Brain import Q_Brain
from .RL_Brain import RTQ_Brain


class RCP(BaseTransportLayerProtocol):
    requiredKeys = {"RLEngine"}
    optionalKeys = {
        "maxTxAttempts": -1, "timeout": -1, "maxPktTxDDL": -1,
        "batchSize": 32,            #
        "learningRate": 1e-2,       #
        "memoryCapacity": 1e5,      #
        "updateFrequency": 100,     # period to replace target network with evaluation
        "gamma": 0.9,               # reward discount
        "epsilon": 0.7,             # greedy policy parameter
        "epsilon_decay": 0.7,
        "convergeLossThresh": 0.01,  # below which we consider the network as converged
        "learnRetransmissionOnly": False,
        # utility
        # "alpha": 2,  # shape of utility function
        # "beta1": 0.9, "beta2": 0.1,   # beta1: emphasis on delivery, beta2: emphasis on delay
        # time-discount delivery
        # "timeDiscount": 0.9,  # reward will be raised to timeDiscound^delay
        # "timeDivider": 100,
    }

    def __init__(self, suid: int, duid: int, params: dict = ..., loglevel=BaseTransportLayerProtocol.LOGLEVEL, create_file: bool = False) -> None:
        super().__init__(suid, duid, params=params,
                         loglevel=loglevel, create_file=create_file)

        self.protocolName = self.__class__.__name__ + self.RLEngine

        self.ACKMode = "SACK"

        self.window = Window(uid=suid, maxCwnd=-1, maxPktTxDDL=self.maxPktTxDDL,
                             maxTxAttempts=self.maxTxAttempts, ACKMode=self.ACKMode, loglevel=loglevel)

        self._initRLEngine(self.RLEngine)

        self.learnCounter = 0
        self.learnPeriod = 8  # number of new data before calling learn

        # performance estimator

        self.pktIgnoredCounter = []

    def _initRLEngine(self, RLEngine: str):
        def _initQLearningEngine():
            self.RL_Brain = Q_Brain(
                nActions=2,
                epsilon=self.epsilon,       # greedy policy parameter
                eta=self.gamma,             # reward discount
                epsilon_decay=self.epsilon_decay,
                updateFrequency=1000,
                # method to choose action. e.g. "argmax" or "ThompsonSampling"
                decisionMethod="argmax",
                decisionMethodArgs={},  # support parameters, e.g. mapfunc=np.exp
                loglevel=logging.INFO,
                # loglevel=logging.DEBUG,
                createLogFile=False,
            )

        def _initDQNEngine():
            self.RL_Brain = DQN_Brain(
                nActions=2, stateDim=5,
                batchSize=self.batchSize,   #
                memoryCapacity=self.memoryCapacity,  # maximum number of experiences to store
                learningRate=self.learningRate,     #
                # period to replace target network with evaluation network
                updateFrequency=self.updateFrequency,
                epsilon=self.epsilon,       # greedy policy parameter
                epsilon_decay=self.epsilon_decay,
                eta=self.gamma,             # reward discount
                # below which we consider the network as converged
                convergeLossThresh=self.convergeLossThresh,
                verbose=False
            )

        def _initRTQEngine():
            self.RL_Brain = RTQ_Brain(
                utilityCalcHandler=self.calcUtility,
                retransMax=self.maxTxAttempts,
                updateFrequency=8,
                loglevel=logging.INFO,
                # loglevel=logging.DEBUG,
                createLogFile=False,
            )

        _initRLEngineDict = {
            "Q_LEARNING": _initQLearningEngine,
            "DQN": _initDQNEngine,
            "RTQ": _initRTQEngine,
        }

        if RLEngine.upper() in _initRLEngineDict:
            _initRLEngineDict[RLEngine.upper()]()
        else:
            raise Exception(
                "RL Engine not recognized. Please choose from", _initRLEngineDict.keys())
        self.RLEngineMode = RLEngine

    def _handleACK(self, ACKPktList: List[Packet]):
        """handle ACK Packets"""
        ACKPidList = []
        for pkt in ACKPktList:
            # only process designated packets
            if pkt.duid == self.suid and pkt.pktType == PacketType.ACK:

                # update pktLoss, RTT and delay
                rtt = self.time-pkt.txTime
                delay = self.time - pkt.genTime

                self.RTTEst.Update(rtt, self.perfDict)
                self._delayUpdate(delay)
                self._pktLossUpdate(False)
                self.perfDict["receivedACK"] += 1

                if self.window.isPktInWindow(pkt.pid):
                    ACKPidList.append(pkt.pid)

        self._handleACK_SACK(SACKPidList=ACKPidList)

    def _handleACK_SACK(self, SACKPidList: List[int]):
        for pid in SACKPidList:
            # one packet is delivered
            delay = self.time-self.window.getPktGenTime(pid)
            self._deliveryRateUpdate(isDelivered=True)
            self.perfDict["deliveredPkts"] += 1
            pktTxAttempts = self.window.getPktTxAttempts(pid)

            if not self.learnRetransmissionOnly:
                reward = self.calcUtility(1, delay)

                finalState = [
                    pktTxAttempts,
                    delay,
                    self.RTTEst.getRTT(),
                    self.perfDict["pktLossHat"],
                    self.perfDict["avgDelay"],
                    self.RTTEst.getRTTVar(),
                ]
                # store the ACKed packet info
                self.RL_Brain.digestExperience(
                    prevState=self.window.getPktRLState(pid),
                    action=1,
                    reward=reward,
                    curState=finalState
                )

            self.window.PopPkt(pid)

    def ticking(self, ACKPktList: List[Packet] = []) -> List[Packet]:
        self.timeElapse()

        self.perfDict["maxWin"] = self.window.perfDict["maxWinCap"]

        self._RLLossUpdate(self.RL_Brain.loss)
        self.perfDict["epsilon"] = self.RL_Brain.epsilon
        if self.RL_Brain.isConverge and self.time < self.perfDict["convergeAt"]:
            self.perfDict["convergeAt"] = self.time

        # process ACK packets
        self._handleACK(ACKPktList)

        # handle timeout packets
        pktsToRetransmit = self._getRetransPkts()
        self.perfDict["retransAttempts"] += len(pktsToRetransmit)

        # fetch new packets based on cwnd and packets in buffer
        newPktList = self._getNewPktsToSend()
        self.perfDict["distinctPktsSent"] += len(newPktList)

        # print the progress if verbose=True
        self.logger.debug("[+] Client {suid}->{duid} @ {time} retx {nReTx} + newTx {newTx}".format(
            suid=self.suid,
            duid=self.duid,
            time=self.time,
            nReTx=len(pktsToRetransmit),
            newTx=len(newPktList)
        ))

        self.perfDict["maxWin"] = max(
            self.perfDict["maxWin"], self.window.bufferSize())
        self.pktIgnoredCounter.append(self.perfDict["ignorePkts"])

        return pktsToRetransmit + newPktList

    def _getRetransPkts(self) -> List[Packet]:
        # wipe out packets that exceed maxTxAttempts and/or maxPktTxDDL
        # self._cleanWindow()
        removedPktNum = self.window.cleanBuffer()
        self.perfDict["ignorePkts"] += removedPktNum

        # pkts to retransmit
        timeoutPktSet = self.window.getTimeoutPkts(updateTxAttempt=False,
                                                   curTime=self.time, RTO=self.RTTEst.getRTO(), pktLossEst=self._pktLossUpdate)

        # generate pkts and update buffer information
        retransPktList = []
        for pkt in timeoutPktSet:
            txAttempts = self.window.getPktTxAttempts(pkt.pid)
            delay = self.time - pkt.genTime
            # use RL to make a decision
            decesionState = [
                txAttempts,
                delay,
                self.RTTEst.getRTT(),
                self.perfDict["pktLossHat"],
                self.perfDict["avgDelay"],
                self.RTTEst.getRTTVar(),
            ]

            # action = self.RL_Brain.chooseAction(state=decesionState,baseline_Q0=None)
            action = self.RL_Brain.chooseAction(
                state=decesionState, baseline_Q0=self.getSysUtil())
            # if self.RLEngine.upper() in {"DQN"}:
            #     # for DQN, we shouldn't interfere its learning strategy.
            #     action = self.RL_Brain.chooseAction(state=decesionState,baseline_Q0=None)
            # else:
            #     action = self.RL_Brain.chooseAction(state=decesionState,baseline_Q0=self.getSysUtil())

            self._retransUpdate(action)

            if action == 0:
                # ignored
                self._ignorePktAndUpdateMemory(
                    pkt, decesionState=decesionState, popKey=True, firstTxAttempt=False)
            else:
                self.window.setPktRLState(pkt.pid, decesionState)
                self.window.updatePktInfo_retrans(pkt.pid, self.time)
                retransPktList += self.window.getPkts([pkt.pid])

        return retransPktList

    def _getNewPktsToSend(self):
        """transmit all packets in txBuffer"""
        newPktList = []
        txAttempts = 0
        delay = 0
        for _ in range(len(self.txBuffer)):
            pkt = self.txBuffer.popleft()

            pkt.txTime = self.time
            pkt.initTxTime = self.time

            decesionState = [
                txAttempts,
                delay,
                self.RTTEst.getRTT(),
                self.perfDict["pktLossHat"],
                self.perfDict["avgDelay"],
                self.RTTEst.getRTTVar(),
            ]
            # action = self.RL_Brain.chooseAction(state=decesionState,baseline_Q0=None)
            action = self.RL_Brain.chooseAction(
                state=decesionState, baseline_Q0=self.getSysUtil())

            if action == 0:
                # ignored
                self._ignorePktAndUpdateMemory(
                    pkt, decesionState=decesionState, popKey=True, firstTxAttempt=True)
            else:
                self._addNewPktAndUpdateMemory(pkt)
                newPktList.append(pkt)

        #

        return newPktList

    """RL related functions"""
    # def calcReward(self, isDelivered, retentionTime):

    def getSysUtil(self, delay=None):
        # get the current system utility

        if not delay:
            delay = self.perfDict["avgDelay"]

        return self.calcUtility(
            delvyRate=self.perfDict["deliveryRate"],
            avgDelay=delay,
        )

    def _ignorePktAndUpdateMemory(self, pkt, decesionState, popKey=True,  firstTxAttempt=False):
        """
        inputs:
            pkt: the packet that is ignored
            decesionState: list(float), the state before taking the action (being igored)
            popKey: remove the pkt from cwnd
            firstTxAttempt: whether the packet has never been transmitted. Since such a packet may not been added to the window.
        """
        # if we ignore a packet, even though we harm the delivery rate, but we contribute to delay
        self.perfDict["ignorePkts"] += 1

        # ignore a packet contributes to no delay penalty
        self._deliveryRateUpdate(isDelivered=False)  # update delivery rate

        # if pid in self.buffer:
        pid = pkt.pid
        if firstTxAttempt or self.window.isPktInWindow(pid):
            if firstTxAttempt:
                finalState = [
                    0,
                    0,
                    self.RTTEst.getRTT(),
                    self.perfDict["pktLossHat"],
                    self.perfDict["avgDelay"],
                    self.RTTEst.getRTTVar(),
                ]
            if self.window.isPktInWindow(pid):
                txAttemps = self.window.getPktTxAttempts(pid)
                delay = self.time - self.window.getPktGenTime(pid)
                finalState = [
                    txAttemps,
                    delay,
                    self.RTTEst.getRTT(),
                    self.perfDict["pktLossHat"],
                    self.perfDict["avgDelay"],
                    self.RTTEst.getRTTVar(),
                ]

            # ignore a packet results in zero changes of system utility, so getSysUtil
            reward = self.getSysUtil()

            self.RL_Brain.digestExperience(
                prevState=decesionState,
                action=0,
                reward=reward,
                curState=finalState
            )

        if popKey:
            self.window.PopPkt(pid)

        return

    def _addNewPktAndUpdateMemory(self, pkt: Packet):
        RLState = [
            0,  # once used, it must be transmitted once.
            self.time - pkt.genTime,
            self.RTTEst.getRTT(),
            self.perfDict["pktLossHat"],
            self.perfDict["avgDelay"],
            self.RTTEst.getRTTVar(),
        ]

        self.window.pushNewPkt(self.time, pkt, RLState)
