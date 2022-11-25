import sys
import os
import copy
from typing import List

from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from DLRCP.protocols.utils import Window
from DLRCP.common import Packet, PacketType


class TCP_Vegas(BaseTransportLayerProtocol):
    requiredKeys = {"timeout"}
    optionalKeys = {"IW": 4,
                    "maxTxAttempts": -1,  # no maximum retransmission
                    "maxPktTxDDL": -1,
                    "vegas_alpha": 2, 
                    "vegas_beta": 4, 
                    "vegas_gamma": 1
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
        self.curTxMode = TCP_Vegas.SLOW_START
        self.ssthresh = sys.maxsize  # slow start threshold (init value: inf)
        self.lastACKCounter = [-1, 0]  # [ACK.pid, showup times]
        
        # TCP Vegas parameters
        self.minRTT = sys.maxsize
        self.baseRTT = sys.maxsize
        self.cntRTT = 0

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
                    # vegas related
                    self.cntRTT += 1
                    self.minRTT = min(self.minRTT, rtt)
                    self.baseRTT = min(self.baseRTT, rtt)
                    
                    self.RTTEst.Update(rtt, self.perfDict)
                    delay = self.time - genTime
                    self._delayUpdate(delay, update=True)

        retransmitPktList = self._handleACK_vegas(ACKPidList)

        return retransmitPktList

    def _handleACK_reno(self, ACKPidList):
        """
        cwnd adjustment
        """
        def cwndIncrement_SS():
            self.cwnd += 1
            if self.cwnd >= self.ssthresh:
                self.curTxMode = TCP_Vegas.CONGESTION_AVOIDANCE
                self.cwnd_inc_counter = 0

        def cwndIncrement_CA():
            self.cwnd_inc_counter += 1
            if self.cwnd_inc_counter >= self.cwnd:
                self.cwnd += 1
                self.cwnd_inc_counter -= self.cwnd

        cwndIncrementFuncDict = {
            TCP_Vegas.SLOW_START: cwndIncrement_SS,
            TCP_Vegas.CONGESTION_AVOIDANCE: cwndIncrement_CA
        }
        pktToRetransmit = []

        for pid in ACKPidList:
            # Vegas works different from NewReno only when there are enough packets
            if self.cntRTT > 2: # collect > 2 pkts
                tmp = self.baseRTT / self.minRTT
                tgtCwnd = self.cwnd * tmp
                diff = self.cwnd - tgtCwnd
                if diff > self.vegas_gamma and self.cwnd < self.ssthresh:
                    # We are going too fast. 
                    self.cwnd = min(self.cwnd, tgtCwnd + 1)
                    # assume segment size is 1
                    self.ssthresh = max(min(self.ssthresh, self.cwnd - 1), 2)
                elif self.cwnd < self.ssthresh:
                    # we are in slow start, and diff <= gamma
                    # follow NewReno slowstart
                    self.cwnd += 1
                else:
                    # linearly increase/decrease cwnd
                    if diff > self.vegas_beta:
                        # too fast
                        self.cwnd -= 1
                        self.ssthresh = max(min(self.ssthresh, self.cwnd - 1), 2)
                    elif diff < self.vegas_alpha:
                        # too slow
                        self.cwnd += 1
                    else:
                        # right speed
                        pass

                continue
            
            if pid < self.lastACKCounter[0]:  # delayed ACK
                continue
            if pid == self.lastACKCounter[0]:  # maybe dup ACK
                self.lastACKCounter[1] += 1
            else:  # new ack

                # ACKed the packets in range
                # [self.lastACKCounter[0]+1, self.lastACKCounter+2, ..., pid]
                for _ in range(pid-self.lastACKCounter[0]):
                    self._deliveryRateUpdate(True)

                self.lastACKCounter = [pid, 1]

                if self.curTxMode in {TCP_Vegas.SLOW_START, TCP_Vegas.CONGESTION_AVOIDANCE}:
                    cwndIncrementFuncDict[self.curTxMode]()

                elif self.curTxMode == TCP_Vegas.RETRANSMISSION:
                    self.curTxMode = TCP_Vegas.SLOW_START

                elif self.curTxMode == TCP_Vegas.FAST_RECOVERY:
                    if pid >= self.high_water:
                        # our retransmission works. Go back to CA
                        self.cwnd = self.ssthresh
                        self.curTxMode = TCP_Vegas.CONGESTION_AVOIDANCE
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
                        if self.curTxMode == TCP_Vegas.FAST_RECOVERY:
                            self.cwnd -= 1

            if self.lastACKCounter[1] >= 3:
                # triple dup ack
                # go to fast retransmission mode, and stay at Fast Recovery
                pktToRetransmit += self._handleTripleDupACK(
                    lastACKPid=self.lastACKCounter[0],
                    dupACKNum=self.lastACKCounter[1])

            # print("LastACKCounter", self.lastACKCounter)
            self.perfDict["retransAttempts"] += len(pktToRetransmit)

        # reset rtt pkts number for next round
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
            if self.curTxMode == TCP_Vegas.FAST_RECOVERY:
                pass
            else:
                self.ssthresh = max(self.cwnd//2, 1)
                self.cwnd = self.ssthresh+3
                self.high_water = self.maxPidSent
                # self._timeoutUpdate()
                # switch to Fast Recovery
                self.curTxMode = TCP_Vegas.FAST_RECOVERY

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
                if self.curTxMode != TCP_Vegas.RETRANSMISSION:  # shrink ssthresh only once
                    self.ssthresh = self.cwnd // 2

                self.logger.debug("retransmission mode")
                self.curTxMode = TCP_Vegas.RETRANSMISSION

                timeoutPidList.append(pid)

                self.cwnd = 1
                self.RTTEst.multiplyRTO(2)

                # update packet info
                self.window.updatePktInfo_retrans(pid, self.time)

        return self.window.getPkts(timeoutPidList)
