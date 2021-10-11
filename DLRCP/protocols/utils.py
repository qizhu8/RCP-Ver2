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
        return [pkt.pid for pkt in self.buffer]

    def pushNewPkts(self, curTime, pktList):
        """
        push new packets to the congestion window.
        """
        if not isinstance(pktList, list):
            pktList = [pktList]

        for pkt in pktList:
            pid = pkt.pid
            if pid not in self.buffer:
                if self._hasSpace():
                    # store the packet info in cwnd
                    self.buffer[pid] = self._genNewPktInfoFromPkt(pkt)

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

        return

    """Default ACK Processing Function"""

    def _ACKPkts_SACK(self, SACKPidList: List[int], perfDict: dict = None) -> None:
        """
        Process ACK based on ideal Selective ACK
        """

        if isinstance(SACKPidList, int):
            LCPidList = [SACKPidList]

        if not SACKPidList:  # nothing to do
            return

        for pid in SACKPidList:
            if pid in self.buffer:
                self.buffer.pop(pid, None)
                perfDict["deliveredPkts"] += 1

                self.logger.debug("SACK: ACK pkt {pid}".format(pid=pid))

        return

    def _ACKPkts_LC(self, LCPidList: List[int], perfDict: dict = None) -> None:
        """
        Process ACK based on largest Consecutive PID
        """
        if isinstance(LCPidList, int):
            LCPidList = [LCPidList]

        if not LCPidList:  # nothing to do
            return

        # the server will return the largest consecutive PID.
        # any pkt whose pid <= the LCPid are delivered
        LCPid = max(LCPidList)

        for pid in range(self.LastLC+1, LCPid+1):
            if pid in self.buffer:
                self.buffer.pop(pid, None)
                perfDict["deliveredPkts"] += 1

                self.logger.debug("LC: ACK pkt {pid}".format(pid=pid))

        self.LastLC = LCPid+1

    def _ACKPkts_error(self, LCPidList: List[int], perfDict: dict = None):
        raise Exception("The ACK function is not implemented.")

    def ACKPkts(self, pidList: List[int], perfDict: dict = None) -> None:
        """Process ACK Packet id list"""
        self.ACKPktProcessor(pidList, perfDict)

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

    def getPkts(self, pidList: List[int]) -> List[Packet]:
        pktList = []
        for pid in pidList:
            if pid in self.buffer:
                pktList.append(self.buffer[pid].toPacket())
        return pktList

    def cleanBuffer(self, curTime: int = -1) -> None:
        """
        wipe out packets that exceed maxTxAttempts or maxPktTxDDL
        """

        if self.maxTxAttempts > -1:
            for pid in self.buffer:
                if self.buffer[pid].txAttempts >= self.maxTxAttempts:
                    self.buffer.pop(pid, None)

                    self.logger.debug("Pkt {pid} exceeds max Tx attempts ({txAttempts} >= {maxTxAttempts}) Give up".format(
                        pid=pid, txAttempts=self.buffer[pid].txAttempts, maxTxAttempts=self.maxTxAttempts
                    ))

        if self.maxPktTxDDL > -1 and curTime > -1:
            timeDDL = curTime - self.maxPktTxDDL
            for pid in self.buffer:
                if self.buffer[pid].genTime < timeDDL:
                    self.buffer.pop(pid, None)

                    self.logger.debug("Pkt {pid} exceeds max queuing delay ({delay} >= {maxPktTxDDL}) Give up".format(
                        pid=pid, delay=curTime-self.buffer[pid].initTxTime, maxPktTxDDL=self.maxPktTxDDL
                    ))

    def getTimeoutPkts(self, curTime: int, RTO: int = -1) -> list:
        """
        Collect packets that are regarded as timeout to be retransmitted.

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

                # update packet info
                self.updatePktInfo_retrans(pid, curTime)

                pktList.append(self.buffer[pid].toPacket())

        return pktList

    def updatePktInfo_retrans(self, pid: int, curTime: int) -> None:
        self.buffer[pid].txTime = curTime
        self.buffer[pid].txAttempts += 1

    def _genNewPktInfoFromPkt(self, pkt: Packet) -> None:
        return pkt.toPktInfo(initTxTime=pkt.genTime, txAttempts=1, isFlying=True)

    def __str__(self):
        rst = ""
        for pid in self.buffer:
            if rst != "":
                rst += "\n"
            rst += self.buffer[pid].__str__()

        return rst
