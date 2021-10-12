import logging
import os
import sys
from typing import List
from DLRCP.common import Packet, PacketType


class Window(object):
    """
    This class manages the cwnd. The client no longer needs to manually monitor packets.

    cwnd: the expected maximum number of packets to send without being acked.

    """

    perfDictDefault = {
        "maxWinCap": 0,  # maximum number of pkts in buffer
    }

    def __init__(self, uid, maxCwnd: int = -1, maxPktTxDDL=-1, maxTxAttempts=-1, ACKMode: str = "lc", loglevel=logging.DEBUG):

        self.initLogger(loglevel)

        self.uid = uid
        self.defaultValue = {
            "maxPktTxDDL": 0,
            "maxTxAttempts": 0,     # maximum allowed transmission attempts
            "cwnd": 0
        }

        if maxCwnd < 0:  # cwnd=-1 means unlimited
            self.maxCwnd = sys.maxsize
        else:
            self.maxCwnd = maxCwnd

        self.buffer = dict()  # used to store packets and packet information

        self.maxPktTxDDL = maxPktTxDDL
        self.maxTxAttempts = maxTxAttempts

        self.ACKPktProcessorDict = {
            'lc': self._ACKPkts_LC,
            'sack': self._ACKPkts_SACK,
            'other': self._ACKPkts_error
        }
        if ACKMode.lower() in self.ACKPktProcessorDict:
            self.ACKPktProcessor = self.ACKPktProcessorDict[ACKMode.lower()]
        else:
            raise Exception("Input ACKMode not recognized. Accept only ",
                            self.ACKPktProcessorDict.keys().__str__())

        """For LCACK"""
        self.LastLC = -1  # last seen LC pid
        """For SACK"""
        # nothing

        # set default value
        self.defaultValue["maxPktTxDDL"] = self.maxPktTxDDL
        self.defaultValue["maxTxAttempts"] = self.maxTxAttempts
        self.defaultValue["cwnd"] = self.maxCwnd

        # performance check
        self.perfDict = Window.perfDictDefault

    def initLogger(self, loglevel):
        """This function is implemented in multiple base classes. """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
            sh.setFormatter(formatter)

    def isPktInWindow(self, pid: int) -> bool:
        return pid in self.buffer

    def reset(self) -> None:
        self.perfDict = Window.perfDictDefault
        self.buffer.clear()

        self.maxPktTxDDL = self.defaultValue["maxPktTxDDL"]
        self.maxTxAttempts = self.defaultValue["maxTxAttempts"]
        self.maxCwnd = self.defaultValue["cwnd"]

    def bufferSize(self) -> int:
        """
        Return the number of packets in buffer
        """
        return len(self.buffer)

    def availSpace(self) -> int:
        """
        Return the number of free spaces
        """
        return max(0, self.maxCwnd - self.bufferSize())

    def _hasSpace(self):
        """check whether the buffer can hold another new packet"""
        return self.availSpace() > 0

    def getPidList(self) -> List[int]:
        return list(self.buffer.keys())
        # return [pid for pid in self.buffer]

    def pushNewPkt(self, curTime, pkt, RLState=[]):
        """Push a packet into the window"""
        pid = pkt.pid
        if pid not in self.buffer:
            if self._hasSpace():
                # store the packet info in cwnd
                self.buffer[pid] = self._genNewPktInfoFromPkt(pkt, RLState)
                self.logger.debug("included pkt {pid}. {bufferSize} pkts in buffer (size={cwnd})".format(
                    pid=pid, bufferSize=self.bufferSize(), cwnd=self.maxCwnd))

                # performance update
                self.perfDict["maxWinCap"] = max(
                    self.perfDict["maxWinCap"], self.bufferSize())

            else:  # no room for new packets

                self.logger.debug("no room for pkt {pid}. {bufferSize} pkts in buffer (size={cwnd})".format(
                    pid=pid, bufferSize=self.bufferSize(), cwnd=self.maxCwnd))
        else:
            # one packet showed up twice.
            self.logger.debug(
                "pkt {pid} already in buffer. What happened?".format(pid=pid))
            
    def pushNewPkts(self, curTime, pktList):
        """
        push new packets to the congestion window. If you have RLState, plase push manually using 
        pushNewPkt
        """
        if not isinstance(pktList, list):
            pktList = [pktList]

        for pkt in pktList:
            pid = pkt.pid
            self.pushNewPkt(curTime, pkt, [])
        return

    """Default ACK Processing Function"""

    def _ACKPkts_SACK(self, SACKPidList: List[int], perfEstimator=None) -> int:
        """
        Process ACK based on ideal Selective ACK
        """

        if isinstance(SACKPidList, int):
            LCPidList = [SACKPidList]

        if not SACKPidList:  # nothing to do
            return 0

        deliveredPkts = 0
        for pid in SACKPidList:
            if pid in self.buffer:
                self.buffer.pop(pid, None)
                deliveredPkts += 1

                if perfEstimator is not None:
                    perfEstimator(1)

                self.logger.debug("SACK: ACK pkt {pid}".format(pid=pid))

        return deliveredPkts

    def _ACKPkts_LC(self, LCPidList: List[int], perfEstimator=None) -> None:
        """
        Process ACK based on largest Consecutive PID
        """
        if isinstance(LCPidList, int):
            LCPidList = [LCPidList]

        if not LCPidList:  # nothing to do
            return 0

        # the server will return the largest consecutive PID.
        # any pkt whose pid <= the LCPid are delivered
        LCPid = max(LCPidList)
        deliveredPkts = 0
        for pid in range(self.LastLC+1, LCPid+1):
            if pid in self.buffer:
                self.buffer.pop(pid, None)
                deliveredPkts += 1

                if perfEstimator is not None:
                    perfEstimator(1)

                self.logger.debug("LC: ACK pkt {pid}".format(pid=pid))

        self.LastLC = LCPid+1
        return deliveredPkts

    def _ACKPkts_error(self, LCPidList: List[int], perfEstimator=None):
        raise Exception("The ACK function is not implemented.")

    def ACKPkts(self, pidList: List[int], perfEstimator=None) -> int:
        """Process ACK Packet id list"""
        return self.ACKPktProcessor(pidList, perfEstimator)

    def PopPkt(self, pid: int) -> Packet:
        """
        This function is for customized protocol design.
        """
        pkt = self.buffer.pop(pid, None)
        return pkt

    def getPktTxTime(self, pid: int) -> int:
        if pid in self.buffer:
            return self.buffer[pid].txTime
        else:
            return None

    def getPktGenTime(self, pid: int) -> int:
        if pid in self.buffer:
            return self.buffer[pid].genTime
        else:
            return None
    
    def getPktTxAttempts(self, pid: int) -> int:
        if pid in self.buffer:
            return self.buffer[pid].txAttempts
        else:
            return None
    
    def getPktRLState(self, pid: int) -> int:
        if pid in self.buffer:
            return self.buffer[pid].RLState
        else:
            return None
    def setPktRLState(self, pid, RLState: List):
        if pid in self.buffer and RLState:
            self.buffer[pid].RLState = RLState
        else:
            raise Exception("Fail to set RLState for pid=", pid, ". RLState=", RLState)

    def getPkts(self, pidList: List[int]) -> List[Packet]:
        pktList = []
        for pid in pidList:
            if pid in self.buffer:
                pktList.append(self.buffer[pid].toPacket())
        return pktList

    def cleanBuffer(self, curTime: int = -1) -> int:
        """
        wipe out packets that exceed maxTxAttempts or maxPktTxDDL
        """
        removedPktNum = 0 
        if self.maxTxAttempts > -1:
            for pid in self.buffer:
                if self.buffer[pid].txAttempts >= self.maxTxAttempts:
                    removedPktNum += 1
                    self.buffer.pop(pid, None)

                    self.logger.debug("Pkt {pid} exceeds max Tx attempts ({txAttempts} >= {maxTxAttempts}) Give up".format(
                        pid=pid, txAttempts=self.buffer[pid].txAttempts, maxTxAttempts=self.maxTxAttempts
                    ))

        if self.maxPktTxDDL > -1 and curTime > -1:
            timeDDL = curTime - self.maxPktTxDDL
            for pid in self.buffer:
                if self.buffer[pid].genTime < timeDDL:
                    removedPktNum += 1
                    self.buffer.pop(pid, None)

                    self.logger.debug("Pkt {pid} exceeds max queuing delay ({delay} >= {maxPktTxDDL}) Give up".format(
                        pid=pid, delay=curTime-self.buffer[pid].initTxTime, maxPktTxDDL=self.maxPktTxDDL
                    ))
        return removedPktNum

    def getTimeoutPkts(self, curTime: int, RTO: int = -1, perfEstimator=None) -> list:
        """
        Collect packets that are regarded as timeout to be retransmitted.
        Update the packet info once a packet is decided to be retransmitted.

        inputs:
            curTime: int, current time
            RTO: int, current retransmission timeout

        outputs:
            pktList: list(Packet), packets to be retransmitted
        """

        # clean packets in buffer
        # self.cleanBuffer(curTime) # not needed

        pktList = []
        # search for packets that exceeds RTO
        for pid in self.buffer:
            timeDDL = curTime - RTO
            if self.buffer[pid].txTime <= timeDDL:

                self.logger.debug("Pkt {pid} exceeds RTO ({retention} >= {RTO}) Retransmitted".format(
                    pid=pid, retention=curTime-self.buffer[pid].txTime, RTO=RTO
                ))

                pktList.append(self.buffer[pid].toPacket())

                # update packet info
                self.updatePktInfo_retrans(pid, curTime)

                # update performance estimator if any
                if perfEstimator:
                    perfEstimator(True)
        return pktList

    def updatePktInfo_retrans(self, pid: int, curTime: int) -> None:
        """Decide to retransmit the packet. So update the packet information."""
        self.buffer[pid].txTime = curTime
        self.buffer[pid].txAttempts += 1

    def _genNewPktInfoFromPkt(self, pkt: Packet, RLState=[]) -> None:
        return pkt.toPktInfo(
            initTxTime=pkt.genTime, 
            txAttempts=1, # in buffer means it will be transmitted 
            isFlying=True,
            RLState=RLState
            )

    def __str__(self):
        rst = ""
        for pid in self.buffer:
            if rst != "":
                rst += "\n"
            rst += self.buffer[pid].__str__()

        return rst


class MovingAvgEst(object):
    """
    This class implements a memory-based estimator.

    The idea is to use a ring buffer to store the most recent records. Estimation is achieved by taking the average of the stored history
    """
    def __init__(self, size:int=200) -> None:
        self.size = size
        self.ring = [0]*self.size
        self.ptr = 0
        self.positiveNum = 0
    
    def update(self, newVal:float)->float:
        # remove the last record
        self.positiveNum -= self.ring[self.ptr]
        self.positiveNum += newVal
        # record the current state
        self.ring[self.ptr] = newVal
        # move pointer
        self.ptr = (self.ptr+1) % self.size

        return self.positiveNum / self.size
    
    def getPktLossRate(self):
        return self.positiveNum / self.size


class AutoRegressEst(object):
    """
    An estimator implementing auto-regression.

    estimator = alpha * newVal + (1-alpha) * estimator
    """
    def __init__(self, alpha:float, initVal:float=0) -> None:
        assert alpha < 1 and alpha > 0, "alpha should be in range (0, 1)"
        self.alpha = alpha
        self.estVal = initVal
    
    def update(self, newVal:float, update=True) -> float:
        newVal = self.alpha * newVal + (1-self.alpha) * self.estVal
        if update:
            self.estVal = newVal
        return newVal
    
    def getEstVal(self) -> float:
        return self.estVal


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
