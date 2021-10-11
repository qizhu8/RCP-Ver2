import os
import sys
import logging
from typing import List
from collections import deque

from .utils import Window
from DLRCP.common import Packet


class RTTEst(object):
    """
    RTT Estimator follows RFC 6298
    """

    def __init__(self):
        self.SRTT = 1    # mean of RTT
        self.RTTVAR = 1  # variance of RTT
        self.RTO = 0

    def Update(self, rtt, perfDict=None) -> None:
        """
        Same as RFC 6298, using auto-regression. But the true rtt estimation, or RTO 
        contains two more variables, RTTVAR (rtt variance) and SRTT (smoothed rtt).
        R' is the rtt for a packet.
        RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
        SRTT <- (1 - alpha) * SRTT + alpha * R'

        The values recommended by RFC are alpha=1/8 and beta=1/4.


        RTO <- SRTT + max (G, K*RTTVAR) where K =4 is a constant, 
        G is a clock granularity in seconds, the number of ticks per second.
        We temporarily simulate our network as a 1 tick per second, so G=1 here

        http://sgros.blogspot.com/2012/02/calculating-tcp-rto.html
        """
        self.RTTVAR = self.RTTVAR * 0.75 + abs(self.RTTVAR-rtt) * 0.25
        self.SRTT = self.SRTT * 0.875 + rtt * (0.125)
        self.RTO = self.SRTT + max(1, 4*self.RTTVAR)

        if perfDict:
            perfDict["rttHat"] = self.SRTT
            perfDict["rto"] = self.RTO

    def getRTT(self) -> float:
        return self.SRTT

    def getRTO(self) -> float:
        return self.RTO

    def multiplyRTO(self, multiplier: float) -> None:
        self.RTO *= multiplier
        return


class BaseTransportLayerProtocol(object):
    """
    Base class for all transport layer protocols
    """
    LOGLEVEL = logging.INFO
    requiredKeys = {}
    optionalKeys = {"maxTxAttempts": -1, "timeout": -1, "maxPktTxDDL": -1,
                    # sum of power utility
                    "alpha": 2,  # shape of utility function
                    "beta1": 0.9, "beta2": 0.1,   # beta1: emphasis on delivery, beta2: emphasis on delay
                    # time-discount delivery
                    "timeDiscount": 0.9,  # reward will be raised to timeDiscound^delay
                    "timeDivider": 100,
                    }

    def parseParamByMode(self, params: dict, requiredKeys: set, optionalKeys: dict) -> None:
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

    def __init__(self, suid: int, duid: int, params: dict = {}, loglevel=LOGLEVEL) -> None:
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
        self.initLogger(loglevel)

        self.protocolName = self.__class__.__name__
        self.suid = suid
        self.duid = duid

        # assign values to
        self.parseParamByMode(params=params, requiredKeys=self.__class__.requiredKeys,
                              optionalKeys=self.__class__.optionalKeys)

        self.RTTEst = RTTEst()  # rtt, rto estimator

        # the buffer that new packets enters
        self.txBuffer = deque(maxlen=None)  # infinite queue

        # window that store un-ACKed packets.
        # Used by protocols that need a retransmission tracking
        self.window = Window(uid=self.suid)

        # performance recording dictionary
        self.perfDict = {
            "distinctPktsRecv": 0,  # of packets received from the application layer
            "distinctPktsSent": 0,  # pkts transmitted, not necessarily delivered or dropped
            "deliveredPkts": 0,  # pkts that are confirmed delivered
            "receivedACK": 0,  # of ACK pkts received (include duplications)
            "retransAttempts": 0,  # of retranmission attempts
            # estimation of the network packet loss (autoregression)
            "pktLossHat": 0,
            "rttHat": 0,  # estimation of the RTT (autoregression)
            # estimation of the Retransmission Timeout (autoregression)
            "rto": 0,
            # estimation of the current delivery rate (autoregression)
            "deliveryRateHat": 0,
            "maxWin": 0,  # maximum # of pkts in Tx window so far
            "loss": sys.maxsize,  # loss of the decision brain (if applicable)
            # when the RL_brain works relatively good (converge) if applicable
            "convergeAt": sys.maxsize,
        }

        # local time at the client side
        self.time = -1

    def initLogger(self, loglevel: int) -> None:
        """This function is implemented in multiple base classes. """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
            sh.setFormatter(formatter)

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

    def getPerf(self, verbose: bool = False) -> dict:
        if verbose:
            for key in self.perfDict:
                print("{key}:{val}".format(key=key, val=self.perfDict[key]))
        return self.perfDict

    def getRTT(self) -> float:
        return self.RTTEst.getRTT()

    def getRTO(self) -> float:
        return self.RTTEst.getRTO()

    def calcUtility(self, delvyRate: float, avgDelay: float) -> float:
        delvyRate_norm = delvyRate
        avgDelay_norm = avgDelay

        # utility used by ICCCN
        """
        UDP_dlvy, UDP_dly = 0.58306*0.9, 261.415*0.9
        ARQ_dlvy, ARQ_dly = 0.86000*1.1, 1204.294*1.1

        
        dlvy = (deliveryRate - UDP_dlvy) / (ARQ_dlvy - UDP_dlvy)
        q = (avgDelay - UDP_dly) / (ARQ_dly-UDP_dly)
        r = -beta1*((1-dlvy)**alpha) - beta2*(q**alpha)
        #"""

        # sum of power
        # r = -self.beta1*((1-delvyRate_norm)**self.alpha) - self.beta2*(avgDelay_norm**self.alpha)

        # exponential
        # print("timeDiscount", self.timeDiscount)
        # print("avgDelay_norm", avgDelay_norm)
        # print("delvyRate_norm", delvyRate_norm)
        # print("alpha", self.alpha)
        # print("timeDivider", self.timeDivider)
        r = (self.timeDiscount**(avgDelay_norm/self.timeDivider)) * (delvyRate_norm**self.alpha)

        return r
