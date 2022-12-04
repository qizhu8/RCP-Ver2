import sys
import os
import copy
from typing import List

from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from DLRCP.protocols.utils import Window
from DLRCP.common import Packet, PacketType


class TCP_CTCP(BaseTransportLayerProtocol):
    requiredKeys = {"timeout"}
    optionalKeys = {"IW": 4,
                    "maxTxAttempts": -1,  # no maximum retransmission
                    "maxPktTxDDL": -1,
                    "ctcp_alpha": 0.125,
                    "ctcp_beta": 0.5,
                    "ctcp_eta": 1.0,
                    "ctcp_k": 0.75,
                    "ctcp_lambda": 0.8,
                    "ctcp_gammaLow": 5,
                    "ctcp_gammaHigh": 30,
                    "ctcp_gamma": 30,
                    # utility
                    # "alpha": 2,  # shape of utility function
                    # "beta1": 0.9, "beta2": 0.1,   # beta1: emphasis on delivery, beta2: emphasis on delay
                    # time-discount delivery
                    # "timeDiscount": 0.9,  # reward will be raised to timeDiscound^delay
                    # "timeDivider": 100,
                    }

    # TCP state
    SLOW_START = 0
    RETRANSMISSION = 1
    FAST_RETRANSMISSION = 2
    CONGESTION_AVOIDANCE = 3
    FAST_RECOVERY = 4

    def __init__(self, suid, duid, params, loglevel=BaseTransportLayerProtocol.LOGLEVEL, create_file: bool = False):
        super().__init__(suid=suid, duid=duid, params=params,
                         loglevel=loglevel, create_file=create_file)

        self.window = Window(uid=suid, maxCwnd=-1, maxPktTxDDL=self.maxPktTxDDL,
                             maxTxAttempts=self.maxTxAttempts, ACKMode="other", loglevel=loglevel)

        # TCP NewReno Parameter

        # store the packet to retransmit after processing
        self.IW = 4  # initial cwnd window

        self.maxPidSent = -1
        self.high_water = -1

        self.cwnd = self.IW
        # for congestion avoidance mode. +1 for one ACK. cwnd+1 when cwnd_inc_counter >= cwnd
        self.cwnd_inc_counter = 0
        self.curTxMode = TCP_CTCP.SLOW_START
        self.ssthresh = sys.maxsize  # slow start threshold (init value: inf)
        self.lastACKCounter = [-1, 0]  # [ACK.pid, showup times]
        
        # ctcp parameters
        self.baseRtt = sys.maxsize
        self.minRtt = sys.maxsize
        self.cntRtt = 0
        self.srtt = 0
        self.lwnd = self.cwnd
        self.dwnd = 0

    def ticking(self, ACKPktList: List[Packet] = ...) -> List[Packet]:
        """
        Decide the number of packets to transmit (return) based on protocol implementation.
        Add new pkts to transmit
        """
        self.timeElapse()

        self.perfDict["maxWin"] = self.window.perfDict["maxWinCap"]

        self.logger.info("host-{uid}@{time}: before processing, {windowSize} pkts in cwnd".format(
            uid=self.suid, time=self.time, windowSize=self.window.bufferSize()))

        # process ACK packets
        pktsToRetransmit = self._handleACK_and_retrans(ACKPktList)

        # handle timeout packets if any
        pktsToRetransmit += self._handleTimeoutPkts()
        self.perfDict["retransAttempts"] += len(pktsToRetransmit)

        # generate new packets
        newPktList = self._getNewPktsToSend()
        self.perfDict["distinctPktsSent"] += len(newPktList)

        self.logger.debug("[+] Client {suid}->{duid} @ {time} retx {nReTx} + newTx {newTx}".format(
            suid=self.suid,
            duid=self.duid,
            time=self.time,
            nReTx=len(pktsToRetransmit),
            newTx=len(newPktList)
        ))

        self.perfDict["maxWin"] = max(
            self.perfDict["maxWin"], self.window.bufferSize())

        return pktsToRetransmit + newPktList

    def _handleACK_and_retrans(self, ACKPktList: List[Packet]) -> List[Packet]:
        """
        Filter out ACK and feed the pid list to reno's processing function.
        Generate retransmission packets based on TCP Reno's state machine
        """
        ACKPidList = []
        for pkt in ACKPktList:
            if pkt.duid == self.suid and pkt.pktType == PacketType.ACK:

                self._pktLossUpdate(False)
                self.perfDict["receivedACK"] += 1

                if self.window.isPktInWindow(pkt.pid):
                    ACKPidList.append(pkt.pid)

                    # update RTT and delay
                    txTime = self.window.getPktTxTime(pkt.pid)
                    genTime = self.window.getPktGenTime(pkt.pid)
                    rtt = self.time-txTime
                    
                    self.minRtt = min(self.minRtt, rtt)
                    self.baseRtt = min(self.baseRtt, rtt)
                    self.srtt = (1-self.ctcp_alpha) * self.srtt + self.ctcp_alpha * rtt
                    self.cntRtt += 1
                    
                    self.RTTEst.Update(rtt, self.perfDict)
                    delay = self.time - genTime
                    self._delayUpdate(delay, update=True)

        retransmitPktList = self._handleACK_ctcp(ACKPidList)

        return retransmitPktList

    def _handleACK_ctcp(self, ACKPidList):
        """
        cwnd adjustment
        """
        def cwndIncrement_SS():
            self.cwnd += 1
            if self.cwnd >= self.ssthresh:
                self.curTxMode = TCP_CTCP.CONGESTION_AVOIDANCE
                self.cwnd_inc_counter = 0

        def cwndIncrement_CA():
            self.cwnd_inc_counter += 1
            if self.cwnd_inc_counter >= self.cwnd:
                self.cwnd += 1
                self.cwnd_inc_counter -= self.cwnd

        cwndIncrementFuncDict = {
            TCP_CTCP.SLOW_START: cwndIncrement_SS,
            TCP_CTCP.CONGESTION_AVOIDANCE: cwndIncrement_CA
        }
        pktToRetransmit = []

        for pid in ACKPidList:            
            if pid < self.lastACKCounter[0]:  # delayed ACK
                continue
            if pid == self.lastACKCounter[0]:  # maybe dup ACK
                self.lastACKCounter[1] += 1
            else:  # new ack
                
                if self.cntRtt <= 2:
                    if self.cwnd < self.ssthresh:
                        cwndIncrement_SS()
                    if self.cwnd >= self.ssthresh:
                        cwndIncrement_CA()
                else:
                    tmp = self.baseRtt / self.minRtt
                    expectedCwnd = self.cwnd * tmp
                    
                    diff = self.cwnd - expectedCwnd
                    
                    if self.cwnd < self.ssthresh and diff < self.ctcp_gamma:
                        cwndIncrement_SS()
                    elif diff < self.ctcp_gamma:
                        self.cwnd += int(self.ctcp_alpha * (self.cwnd ** self.ctcp_k))
                        
                        adder = 1
                        self.lwnd += adder
                        self.dwnd = self.cwnd - self.lwnd
                    
                    elif diff >= self.ctcp_gamma:
                        adder = 1
                        self.m_lwnd += adder
                        dwndInPackets = self.dwnd / self.cwnd
                        dwndInPackets = max(0, dwndInPackets - self.ctcp_eta*diff)
                        self.dwnd = int(dwndInPackets * self.cwnd)
                        self.cwnd = self.lwnd + self.dwnd
                
                
                # ACKed the packets in range
                # [self.lastACKCounter[0]+1, self.lastACKCounter+2, ..., pid]
                for _ in range(pid-self.lastACKCounter[0]):
                    self._deliveryRateUpdate(True)

                self.lastACKCounter = [pid, 1]

                if self.curTxMode in {TCP_CTCP.SLOW_START, TCP_CTCP.CONGESTION_AVOIDANCE}:
                    # taken cared by previous code
                    pass

                elif self.curTxMode == TCP_CTCP.RETRANSMISSION:
                    self.curTxMode = TCP_CTCP.SLOW_START

                elif self.curTxMode == TCP_CTCP.FAST_RECOVERY:
                    if pid >= self.high_water:
                        # our retransmission works. Go back to CA
                        self.cwnd = self.ssthresh
                        self.curTxMode = TCP_CTCP.CONGESTION_AVOIDANCE
                        self.cwnd_inc_counter = 0
                    else:
                        # there are still packets missing
                        # decrease cwnd
                        # retransmit the last unACKed packet
                        self.cwnd -= 1
                        self.cwnd = max(self.cwnd, 0)  # TODO
                        pktToRetransmit += self.window.getPkts([pid+1])
                        self.window.updatePktInfo_retrans(pid+1, self.time)

                        self._pktLossUpdate(False)
                        # self._deliveryRateUpdate
                    # self._timeoutUpdate()

                for oldPid in self.window.getPidList():
                    if oldPid <= pid:
                        # self.window.pop(oldPid, None)
                        self.window.PopPkt(oldPid)
                        if self.curTxMode == TCP_CTCP.FAST_RECOVERY:
                            self.cwnd -= 1

            if self.lastACKCounter[1] >= 3:
                # triple dup ack
                # go to fast retransmission mode, and stay at Fast Recovery
                pktToRetransmit += self._handleTripleDupACK(
                    lastACKPid=self.lastACKCounter[0],
                    dupACKNum=self.lastACKCounter[1])

            # print("LastACKCounter", self.lastACKCounter)
            self.perfDict["retransAttempts"] += len(pktToRetransmit)

        self.cntRTT = 0
        self.minRTT = sys.maxsize
        return pktToRetransmit

    def _handleTripleDupACK(self, lastACKPid: int, dupACKNum: int) -> List[Packet]:
        """
        Add the (potential) missiong packet to retransmit if a trip-dup-ack is detected.
        """
        retransPktList = []

        # the missing packet is lastACKPid+1
        missPid = lastACKPid+1
        if self.window.isPktInWindow(missPid):
            # switch to Fast Retransmission mode
            # send the missing packet
            # update cwnd and ssthresh
            if self.curTxMode == TCP_CTCP.FAST_RECOVERY:
                pass
            else:
                self.ssthresh = max(self.cwnd//2, 1)
                self.cwnd = self.ssthresh+3
                self.high_water = self.maxPidSent
                # self._timeoutUpdate()
                # switch to Fast Recovery
                self.curTxMode = TCP_CTCP.FAST_RECOVERY

                retransPktList += self.window.getPkts([missPid])
                self.window.updatePktInfo_retrans(missPid, self.time)

                self._pktLossUpdate(False)

            self.logger.debug("fast recovery\n" +
                              "cwnd={}, sshthresh={}, high_water={}".format(
                                  self.cwnd, self.ssthresh, self.high_water)
                              )

        return retransPktList

    def _getNewPktsToSend(self):
        """
        Get new pkts from the TxBuffer.
        """
        pktList = []

        # number of packets to transfer from txbuffer to TCP's tx window
        numOfNewPackets = min(
            self.cwnd - self.window.bufferSize(), len(self.txBuffer))
        numOfNewPackets = max(numOfNewPackets, 0)

        for _ in range(numOfNewPackets):
            # add packets to window
            pkt = self.txBuffer.popleft()
            pkt.txTime = self.time
            pkt.initTxTime = self.time
            pkt.txAttempts = 1

            pktList.append(pkt)
        self.window.pushNewPkts(self.time, pktList)

        return pktList

    def _handleTimeoutPkts(self) -> List[Packet]:
        """
        Retransmission packets that exceed RTO. Switch to Retransmission mode if there is at least one.
        """
        timeoutPidList = []

        pidList = self.window.getPidList()
        pidList.sort()

        # print("cur timeout is ", self.timeout)
        for pid in pidList:
            if (self.time-self.window.getPktTxTime(pid)) > self.RTTEst.getRTO():
                self.logger.debug("[-]Client {uid} @ {time} Pkt {pid} is timeout {queuingTime} >= {RTO}".format(
                    uid=self.suid, time=self.time, pid=pid, queuingTime=self.time-self.window.getPktTxTime(pid), RTO=self.RTTEst.getRTO()))

                # switch to Retransmission mode
                # push the packet to buffer (retransmit)
                # update timeout, cwnd, ssthresh
                if self.curTxMode != TCP_CTCP.RETRANSMISSION:  # shrink ssthresh only once
                    self.ssthresh = self.cwnd // 2

                self.logger.debug("retransmission mode")
                self.curTxMode = TCP_CTCP.RETRANSMISSION

                timeoutPidList.append(pid)

                self.cwnd = 1
                self.RTTEst.multiplyRTO(2)

                # update packet info
                self.window.updatePktInfo_retrans(pid, self.time)

        return self.window.getPkts(timeoutPidList)
