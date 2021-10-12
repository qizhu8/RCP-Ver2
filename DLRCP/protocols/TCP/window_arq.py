import sys
from typing import List

from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from DLRCP.protocols.utils import Window
from DLRCP.common import Packet, PacketType


class Window_ARQ(BaseTransportLayerProtocol):
    """
    General working procedure:

    In each time slot, sends at most cwnd packets from the top of the txBuffer. Each packet remains in pktInfo dictionary unless
    1. ACKed
    2. exceeds maximum transmission attempts (maxTxAttempts) or exceeds maximum pkt retention time (maxPktTxDDL)

    A transmitted packet will trigger the retransmission process if :
    1. NACKed (sent by channel)
    2. not ACKed after timeout

    Once a packet needs retransmission, it is pushed at the top of the transmission buffer (txBuffer)
    """

    requiredKeys = {"cwnd", "ACKMode"}
    optionalKeys = {
        "maxTxAttempts": -1,  # maximum allowed tx attempts
        # "timeout":30, # initial rto
        "maxPktTxDDL": -1,
        # sum of power utility
        "alpha": 2,  # shape of utility function
        "beta1": 0.9, "beta2": 0.1,   # beta1: emphasis on delivery, beta2: emphasis on delay
        # time-discount delivery
        "timeDiscount": 0.9,  # reward will be raised to timeDiscound^delay
        "timeDivider": 100,
    }

    def __init__(self, suid, duid, params, loglevel=BaseTransportLayerProtocol.LOGLEVEL):
        super().__init__(suid=suid, duid=duid, params=params, loglevel=loglevel)

        if self.cwnd <= 0:
            self.protocolName = "ARQ_inf_wind"
        else:
            self.protocolName = "ARQ_finit_wind"

        # initialize the congestion window
        self.window = Window(uid=suid, maxCwnd=self.cwnd, maxPktTxDDL=self.maxPktTxDDL,
                             maxTxAttempts=self.maxTxAttempts, ACKMode=self.ACKMode, loglevel=loglevel)

    def ticking(self, ACKPktList: List[Packet]) -> List[Packet]:
        """
        Decide the number of packets to transmit (return) based on protocol implementation.
        Add new pkts to transmit
        """

        self.timeElapse()

        self.perfDict["maxWin"] = self.window.perfDict["maxWinCap"]

        self.logger.info("host-{uid}@{time}: before processing, {windowSize} pkts in cwnd".format(
            uid=self.suid, time=self.time, windowSize=self.window.bufferSize()))

        # process ACK packets
        self._handleACK(ACKPktList)

        # handle timeout packets
        pktsToRetransmit = self.window.getTimeoutPkts(
            curTime=self.time, RTO=self.RTTEst.getRTO(), perfEstimator=self._pktLossUpdate)
        self.perfDict["retransAttempts"] += len(pktsToRetransmit)


        # fetch new packets based on cwnd and packets in buffer
        newPktList = self._getNewPktsToSend()
        self.perfDict["distinctPktsSent"] += len(newPktList)

        self.logger.debug("[+] Client {suid}->{duid} @ {time} retx {nReTx} + newTx {newTx}".format(
            suid=self.suid,
            duid=self.duid,
            time=self.time,
            nReTx=len(pktsToRetransmit),
            newTx=len(newPktList)
        ))

        self.perfDict["maxWin"] = max(self.perfDict["maxWin"], self.window.bufferSize())

        return pktsToRetransmit + newPktList

    def _handleACK(self, ACKPktList):
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
                    self.RTTEst.Update(rtt, self.perfDict)
                    delay = self.time - genTime
                    self._delayUpdate(delay, update=True)
                    
        self.perfDict["deliveredPkts"] += self.window.ACKPkts(ACKPidList, self._deliveryRateUpdate)

    def _getNewPktsToSend(self):

        newPktNum = min(self.window.availSpace(), len(self.txBuffer))

        newPktList = []

        for _ in range(newPktNum):
            newpkt = self.txBuffer.popleft()
            newpkt.txTime = self.time
            newpkt.initTxTime = self.time

            newPktList.append(newpkt)

        self.window.pushNewPkts(self.time, newPktList)

        return newPktList

    def clientSidePerf(self):
        for key in self.perfDict:
            print("{key}:{val}".format(key=key, val=self.perfDict[key]))
        return self.perfDict
