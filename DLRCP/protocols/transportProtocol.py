import os
import sys
import logging
import numpy as np
from typing import List
from collections import deque

from .utils import Window, MovingAvgEst, AutoRegressEst, RTTEst
from DLRCP.common import Packet



class BaseTransportLayerProtocol(object):
    """
    Base class for all transport layer protocols
    """
    LOGLEVEL = logging.INFO
    requiredKeys = {"utilityMethod"}
    optionalKeys = {"maxTxAttempts": -1, "timeout": -1, "maxPktTxDDL": -1,
                    }

    utilityKeys = {
        # parameter for utility
        "alpha": 2,  # shape of utility function
        "timeDivider": 100,
        "beta": 0.9,  # beta: emphasis on delay
    }
    


    defaultPerfDict = {
            "distinctPktsRecv": 0,  # of packets received from the application layer
            "distinctPktsSent": 0,  # pkts transmitted, not necessarily delivered or dropped
            "deliveredPkts": 0,  # pkts that are confirmed delivered
            "receivedACK": 0,  # of ACK pkts received (include duplications)
            "retransAttempts": 0,  # of retranmission attempts
            "retransProb": 0,
            "ignorePkts": 0,
            # estimation of the network packet loss (autoregression)
            "pktLossHat": 0,
            "rttHat": 0,  # estimation of the RTT (autoregression)
            # estimation of the Retransmission Timeout (autoregression)
            "rto": 0,
            "avgDelay": 0, # average transmission delay (delivery time - pktGenTime)
            # estimation of the current delivery rate (autoregression)
            "deliveryRate": 0,
            "maxWin": 0,  # maximum # of pkts in Tx window so far
            "loss": 0,  # loss of the decision brain (if applicable)
            # when the RL_brain works relatively good (converge) if applicable
            "convergeAt": sys.maxsize,
        }

    def parseParamByMode(self, params: dict, requiredKeys: set, optionalKeys: dict, utilityKeys: dict) -> None:
        # required keys
        for key in requiredKeys:
            assert key in params, key + \
                " is required for " + type(self).__name__
            setattr(self, key, params[key])

        # optinal keys
        for key in optionalKeys:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, optionalKeys[key])
        
        # utilityKeys keys
        for key in utilityKeys:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, utilityKeys[key])


    def __init__(self, suid: int, duid: int, params: dict = {}, loglevel=LOGLEVEL, create_file:bool=False) -> None:
        """
        1. define protocolName                                          self.protocolName
        2. set the link information,                                    self.suid, self.duid
        3. parse the input params based on the class's requiredKeys and optionalKeys. Each key
        will be then turned to a class attribute
            requiredKeys = {}
            optionalKeys = {"maxTxAttempts": -1, "timeout": -1, "maxPktTxDDL": -1}
        4. initialize RTT estimator                                     self.RTTEst
        5. allocate Tx buffer for storing new packets                   self.txbuffer
        6. declare (but no memory allocation) the Tx window if needed   self.window
        7. declare a dictionary to store tranmission performance        self.perfDict
        """

        self.protocolName = self.__class__.__name__
        self.suid = suid
        self.duid = duid

        self.initLogger(loglevel, create_file)


        # assign values to
        self.parseParamByMode(
            params=params, 
            requiredKeys=self.__class__.requiredKeys.union(BaseTransportLayerProtocol.requiredKeys), # 
            optionalKeys={**self.__class__.optionalKeys, **BaseTransportLayerProtocol.optionalKeys}, 
            utilityKeys={**self.__class__.utilityKeys, **BaseTransportLayerProtocol.utilityKeys},
            )

        # assign utility calculator handler
        self.initUtilityCalculator()

        self.RTTEst = RTTEst()                  # rtt, rto estimator
        self.pktLossEst = MovingAvgEst(size=500)  # estimate pkt loss rate
        self.delvyRateEst = AutoRegressEst(alpha=0.01)
        self.delayEst = AutoRegressEst(alpha=0.01)
        self.retransProbEst = AutoRegressEst(alpha=0.01)
        self.RLLossEst = AutoRegressEst(alpha=0.01) # reserved for RL_Brain


        # the buffer that new packets enters
        self.txBuffer = deque(maxlen=None)      # infinite queue

        # window that store un-ACKed packets.
        # Used by protocols that need a retransmission tracking
        self.window = Window(uid=self.suid)

        # performance recording dictionary
        self.perfDict = {}
        self.initPerfDict()

        # local time at the client side
        self.time = -1
    
    def reset(self):
        self.time=-1
        self.initPerfDict()
        self.txBuffer.clear()
        self.window.reset()

    def initUtilityCalculator(self):
        calcUtilityHandlerDict={
            "sumpower":self.calcUtility_sumPower, 
            "timediscount":self.calcUtility_timeDiscount,
            }

        if self.utilityMethod.lower() in calcUtilityHandlerDict:
            self.calcUtilityHandler = calcUtilityHandlerDict[self.utilityMethod.lower()]

    def initPerfDict(self):
        """initialize perfDict to default values"""
        for key, val in BaseTransportLayerProtocol.defaultPerfDict.items():
            self.perfDict[key] = val

    def initLogger(self, loglevel: int, create_file: bool=False) -> None:
        """This function is implemented in multiple base classes. """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)
        formatter = logging.Formatter(
            '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
        
        if create_file:
            # create file handler for logger.
            fh = logging.FileHandler(self.protocolName+'.log')
            fh.setLevel(loglevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def acceptNewPkts(self, pktList: List[Packet]) -> None:
        """
        Accept packets from application layer. 
        """
        self.perfDict["distinctPktsRecv"] += len(pktList)
        self.txBuffer.extend(pktList)
        self.logger.debug("[+] Client {suid} @ {time} accept {nNewPkt} pkts".format(
            suid=self.suid,
            time=self.time,
            nNewPkt=len(pktList)
        ))

    def ticking(self, ACKPktList: list = []) -> List[Packet]:
        """
        1. process feedbacks based on ACKPktList
        2. prepare packets to (re)transmit
        """
        raise NotImplementedError

    def timeElapse(self) -> None:
        self.time += 1


    """Update performance estimator"""
    def _RLLossUpdate(self, loss):
        # keep track of RL network's loss
        self.perfDict["loss"] = self.RLLossEst.update(loss)

    def _retransUpdate(self, isRetrans):
        """The probability of retransmitting a packet."""
        self.perfDict["retransProb"] = self.retransProbEst.update(int(isRetrans))

    def _delayUpdate(self, delay:float, update=True):
        """auto-regression to estimate averaged transmission delay = curTime-pkt.genTime. only for performance check."""
        newDelay = self.delayEst.update(delay, update=update)
        if update:
            self.perfDict["avgDelay"] = newDelay
        return newDelay

    def _deliveryRateUpdate(self, isDelivered):
        """
        The probability of a packet being delivered after multiple retransmission
        """
        self.perfDict["deliveryRate"] = self.delvyRateEst.update(int(isDelivered))
        

    def _pktLossUpdate(self, isLost:bool):
        """The channel packet loss probability"""
        self.perfDict["pktLossHat"] = self.pktLossEst.update(int(isLost))

    """Get performance metric"""
    def getPerf(self, verbose: bool = False) -> dict:
        if verbose:
            for key in self.perfDict:
                print("{key}:{val}".format(key=key, val=self.perfDict[key]))
        return self.perfDict

    def getRTT(self) -> float:
        return self.RTTEst.getRTT()

    def getRTO(self) -> float:
        return self.RTTEst.getRTO()

    """Calculate the protocol performance"""

    def calcUtility(self, delvyRate: float, avgDelay: float) -> float:
        return self.calcUtilityHandler(delvyRate, avgDelay)

    def calcUtility_sumPower(self, delvyRate: float, avgDelay: float) -> float:
        # utility used by ICCCN 2021. A little bit tricky. We changed it to be a general form.
        """
        UDP_dlvy, UDP_dly = 0.58306*0.9, 261.415*0.9
        ARQ_dlvy, ARQ_dly = 0.86000*1.1, 1204.294*1.1

        
        dlvy = (deliveryRate - UDP_dlvy) / (ARQ_dlvy - UDP_dlvy)
        q = (avgDelay - UDP_dly) / (ARQ_dly-UDP_dly)
        r = -beta1*((1-dlvy)**alpha) - beta2*(q**alpha)
        #"""
        # sigmoid = lambda x : 1/(1 + np.exp(-x))
        

        # sum of power
        r = -self.beta * ((1-delvyRate)**self.alpha) - (1-self.beta) * ((avgDelay/self.timeDivider)**self.alpha)
        return r

    def calcUtility_timeDiscount(self, delvyRate: float, avgDelay: float) -> float:
        
        # exponential
        r = (self.beta**(avgDelay/self.timeDivider)) * (delvyRate**self.alpha)
        return r
        
    def clientSidePerf(self, verbose=False):
        if verbose:
            for key in self.perfDict:
                print("{key}:{val}".format(key=key, val=self.perfDict[key]))
        return self.perfDict