import numpy as np
from typing import List

from DLRCP.common.packet import Packet
from DLRCP.protocols.utils import Window
from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from .RL_Brain import DQN_Brain
from .RL_Brain import Q_Brain

class RCP(BaseTransportLayerProtocol):
    requiredKeys = {"RLEngine"}
    optionalKeys = {
        "maxTxAttempts":-1, "timeout":-1, "maxPktTxDDL":-1,
        "batchSize": 32,            #
        "learningRate": 1e-2,       #
        "memoryCapacity": 1e5,      # 
        "updateFrequency": 100,     # period to replace target network with evaluation
        "gamma": 0.9,               # reward discount
        "epsilon": 0.7,             # greedy policy parameter
        "epsilon_decay": 0.7,
        "convergeLossThresh": 0.01, # below which we consider the network as converged
        "learnRetransmissionOnly": False,
    }

    def __init__(self, suid: int, duid: int, params: dict = ..., loglevel=BaseTransportLayerProtocol.LOGLEVEL) -> None:
        super().__init__(suid, duid, params=params, loglevel=loglevel)

        self.ACKMode = "SACK"

        self.window = Window(uid=suid, maxCwnd=-1, maxPktTxDDL=self.maxPktTxDDL, maxTxAttempts=self.maxTxAttempts, ACKMode=self.ACKMode, loglevel=loglevel)

        self._initRLEngine(self.RLEngine)

        self.learnCounter = 0
        self.learnPeriod = 8 # number of new data before calling learn
        self.pktLossTrackingNum = 100
        self.pktLostNum = 0

        """
        A cyclic queue keeping track of the most recent $pktLossTrackingNum of packet's delivery info, which is used to estimate the channel packet loss probability.
        """
        self.pktLossInfoQueue = np.zeros(self.pktLossTrackingNum) # keep track of the most recent 100 packets
        self.pktLossInfoPtr = 0 

    
    def _initRLEngine(self, RLEngine:str):
        def _initQLearningEngine():
            self.RL_Brain = Q_Brain(nActions=2)
        
        def _initDQNEngine():
            self.RL_Brain = DQN_Brain(
                nActions=2, stateDim=5, 
                batchSize=self.batchSize,   # 
                memoryCapacity=self.memoryCapacity, # maximum number of experiences to store
                learningRate=self.learningRate,     # 
                updateFrequency=self.updateFrequency,# period to replace target network with evaluation network 
                epsilon=self.epsilon,       # greedy policy parameter 
                epsilon_decay=self.epsilon_decay, # 
                eta=self.gamma,             # reward discount
                convergeLossThresh=self.convergeLossThresh, # below which we consider the network as converged
                verbose=False
            )
        
        _initRLEngineDict = {
            "Q_LEARNING": _initQLearningEngine,
            "DQN": _initDQNEngine
        }
        
        if RLEngine.upper() in _initRLEngineDict:
            _initRLEngineDict[RLEngine.upper()]()
        else:
            raise Exception("RL Engine not recognized. Please choose from", _initRLEngineDict.keys())
        self.RLEngineMode = RLEngine


    def _handleACK(self, ACKPktList: List[Packet]):
        """handle ACK Packets"""
        ACKPidList = []
        for pkt in ACKPktList:
            # only process designated packets
            if pkt.duid == self.suid and pkt.packetType == Packet.ACK:
                self.perfDict["receivedACK"] += 1

                if self.window.isPktInWindow(pkt.pid):
                    ACKPidList.append(pkt.pid)

                    rtt = self.time-self.window.getPktTxTime(pkt.pid)
                    self.RTTEst.Update(rtt, self.perfDict)

        self._handleACK_SACK(SACKPidList=ACKPidList) 


    def _handleACK_SACK(self, SACKPidList):
        for pid in SACKPidList:
            
            # one packet is delivered
            delay = self.time-self.buffer[pid].genTime
            self._delayUpdate(delay=delay)
            self._pktLossUpdate(isLost=False)
            self._deliveryRateUpdate(isDelivered=True)
            self.perfDict["deliveredPkts"] += 1


            if self.buffer[pid].txAttempts > 1 or not self.learnRetransmissionOnly:
                reward = self.calcUtility(1, delay, self.alpha, self.beta1, self.beta2)

                # store the ACKed packet info
                self.RL_Brain.digestExperience(
                    prevState=self.buffer[pid].RLState,
                    action=1,
                    reward=reward,
                    curState=[
                        self.buffer[pid].txAttempts,
                        delay,
                        self.SRTT,
                        self.perfDict["pktLossHat"],
                        self.perfDict["avgDelay"]
                    ]
                )
                # self.learn() # this function is integrated into the RL_Brain.digestExperience

            self.buffer.pop(pid, None)


    def ticking(self, ACKPktList=[]):
        self.timeElapse()

        self._RL_lossUpdate(self.RL_Brain.loss)
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
        if self.verbose:
            self._printProgress(
                retransPkts=pktsToRetransmit,
                newPktList=newPktList
                )
        
        self.pktIgnoredCounter.append(self.perfDict["ignorePkts"])

        return pktsToRetransmit + newPktList


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

    def _pktLossUpdate(self, isLost):
        # channel state estimate
        isLost = int(isLost)
        self.pktLostNum -= self.pktLossInfoQueue[self.pktLossInfoPtr]
        self.pktLostNum += isLost
        self.pktLossInfoQueue[self.pktLossInfoPtr] = isLost
        
        self.pktLossInfoPtr = (self.pktLossInfoPtr+1) % self.pktLossTrackingNum

        self.perfDict["pktLossHat"] = self.pktLostNum / self.pktLossTrackingNum

    def _RL_lossUpdate(self, loss):
        # keep track of RL network
        self.perfDict["RL_loss"] = (7/8.0) * self.perfDict["RL_loss"] + (1/8.0) * loss

    def _RL_retransUpdate(self, isRetrans):
        self.perfDict["retranProb"] = 0.99 * self.perfDict["retranProb"] + 0.01 * int(isRetrans)

    def _delayUpdate(self, delay, update=True):
        """auto-regression to estimate averaged delay. only for performance check."""
        alpha = 0.01
        if update:
            self.perfDict["avgDelay"] = (1-alpha) * self.perfDict["avgDelay"] + alpha * delay
            return self.perfDict["avgDelay"]
        else:
            return (1-alpha) * self.perfDict["avgDelay"] + alpha * delay

    def _deliveryRateUpdate(self, isDelivered):
        alpha = 0.001
        self.perfDict["deliveryRate"] = (1-alpha) * self.perfDict["deliveryRate"] + alpha * int(isDelivered)

    def _pktLossUpdate(self, isLost):
        # channel state estimate
        isLost = int(isLost)
        self.pktLostNum -= self.pktLossInfoQueue[self.pktLossInfoPtr]
        self.pktLostNum += isLost
        self.pktLossInfoQueue[self.pktLossInfoPtr] = isLost
        
        self.pktLossInfoPtr = (self.pktLossInfoPtr+1) % self.pktLossTrackingNum

        self.perfDict["pktLossHat"] = self.pktLostNum / self.pktLossTrackingNum

    def learn(self):
        """
        Learn from the stored experiences.
        If converge, learn per every $self.learnPeriod,
        otherwise, learn per function call.

        Will be deprecated.
        """
        if not self.RL_Brain.isConverge:
            self.RL_Brain.learn()
        else:
            self.learnCounter += 1
            if self.learnCounter >= self.learnPeriod:
                self.learnCounter = 0
                self.RL_Brain.learn()

if __name__ == "__main__":
    suid = 0
    duid=1
    params = {"maxTxAttempts":-1, "timeout":30, "maxPktTxDDL":-1,
    "alpha":2,
    "beta1":0.8, "beta2":0.2, # beta1: emphasis on delivery, beta2: emphasis on delay
    "gamma":0.9,
    "learnRetransmissionOnly": True}, # whether only learn the data related to retransmission
    RCP(suid=suid, duid=duid, params=params)