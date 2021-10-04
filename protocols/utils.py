import logging
import os, sys

# add parent directory to path
sys.path.append(os.path.normpath(os.path.join(
    os.path.abspath(__file__), '..', '..')))

from common.packet import Packet

class Window(object):
    """
    This class manages the cwnd. The client no longer needs to manually monitor packets.

    cwnd: the expected maximum number of packets to send without being acked.
    
    """

    perfDictDefault = {
        "maxWinCap": 0, # maximum number of pkts in buffer
    }
    
    def __init__(self, uid, cwnd:int=-1, maxPktTxDDL=-1, maxTxAttempts=-1, logLevel=logging.DEBUG):

        self.uid = uid
        self.defaultValue = {
            "maxPktTxDDL": 0,
            "maxTxAttempts": 0,
            "cwnd": 0
        }

        if cwnd < 0: # cwnd=-1 means unlimited
            self.cwnd = sys.maxsize
        else:
            self.cwnd = cwnd

        self.buffer = dict() # used to store packets and packet information

        self.maxPktTxDDL = maxPktTxDDL
        self.maxTxAttempts = maxTxAttempts

        # store default value
        self.defaultValue["maxPktTxDDL"] = self.maxPktTxDDL
        self.defaultValue["maxTxAttempts"] = self.maxTxAttempts
        self.defaultValue["cwnd"] = self.cwnd

        # performance check
        self.perfDict = Window.perfDictDefault
    
    def isPktInWindow(self, pkt: Packet)->bool:
        return pkt.pid in self.buffer

    def reset(self):
        self.perfDict = Window.perfDictDefault
        self.buffer.clear()

        self.maxPktTxDDL = self.defaultValue["maxPktTxDDL"]
        self.maxTxAttempts = self.defaultValue["maxTxAttempts"]
        self.cwnd = self.defaultValue["cwnd"]

    def bufferSize(self):
        """
        Return the number of packets in buffer
        """
        return len(self.buffer)
    
    def availSpace(self):
        """
        Return the number of free spaces
        """
        return max(0, self.cwnd - self.bufferSize())
    
    def _hasSpace(self):
        """check whether the buffer can hold another new packet"""
        return self.availSpace() > 0

    def pushPkts(self, curTime, pktList):
        if not isinstance(pktList, list):
            pktList = [pktList]
        
        for pkt in pktList:
            pid = pkt.pid
            if pid not in self.buffer:
                if self._hasSpace():
                    self.buffer[pid] = self._genNewPktInfoFromPkt(pkt)

                    logging.info("included pkt {pid}. {bufferSize} pkts in buffer (size={cwnd})".format(pid=pid, bufferSize=self.bufferSize(), cwnd=self.cwnd))

                    # performance update
                    self.perfDict["maxWinCap"] = max(self.perfDict["maxWinCap"], self.bufferSize())

                else: # no room for new packets

                    logging.info("no room for pkt {pid}. {bufferSize} pkts in buffer (size={cwnd})".format(pid=pid, bufferSize=self.bufferSize(), cwnd=self.cwnd))
            else:
                # TODO when we push another packet? Should we just ignore it?
                # self.buffer[pid].txAttempts += 1
                # self.buffer[pid].isFlying = True

                logging.debug("pkt {pid} already in buffer. What happened?".format(pid=pid))
        
        return
    
    def ACKPkts_SACK(self, SACKPidList):
        if not SACKPidList: # nothing to do
            return 

        for pid in SACKPidList:
            if pid in self.buffer:
                self.buffer.pop(pid, None)
                # add performance counting related codes

                logging.info("SACK: ACK pkt {pid}".format(pid=pid))
            
        return
    
    def ACKPkts_LC(self, LCPidList):       
        if isinstance(LCPidList, int):
            LCPidList = [LCPidList]
        
        if not LCPidList: # nothing to do
            return 

        LCPid = max(LCPidList)

        pidsInBuffer = list(self.buffer.keys())

        for pid in pidsInBuffer:
            if pid <= LCPid:
                self.buffer.pop(pid, None)
                # add performance counting related codes

                logging.info("LC: ACK pkt {pid}".format(pid=pid))
    
    def cleanBuffer(self, curTime=-1):
        # wipe out packets that exceed maxTxAttempts and/or maxPktTxDDL
        
        if self.maxTxAttempts > -1:
            pktsToConsider = set(self.buffer.keys())
            for pid in pktsToConsider:
                if self.buffer[pid].txAttempts >= self.maxTxAttempts:
                    self.buffer.pop(pid, None)

                    logging.info("Pkt {pid} exceeds max Tx attempts ({txAttempts} >= {maxTxAttempts}) Give up".format(
                        pid=pid, txAttempts=self.buffer[pid].txAttempts , maxTxAttempts=self.maxTxAttempts
                    ))
        
        if self.maxPktTxDDL > -1 and curTime > -1:
            timeDDL = curTime - self.maxPktTxDDL
            pktsToConsider = set(self.buffer.keys())
            for pid in pktsToConsider:
                if self.buffer[pid].genTime < timeDDL:
                    self.buffer.pop(pid, None)

                    logging.info("Pkt {pid} exceeds max queuing delay ({delay} >= {maxPktTxDDL}) Give up".format(
                        pid=pid, delay=curTime-self.buffer[pid].initTxTime, maxPktTxDDL=self.maxPktTxDDL
                    ))

    def getRetransPkts(self, curTime, RTO=-1):
        # clean packets in buffer
        self.cleanBuffer(curTime)

        pktList = []
        # search for packets that exceeds RTO
        for pid in self.buffer:
            timeDDL = curTime - RTO
            if self.buffer[pid].txTime <= timeDDL:

                logging.info("Pkt {pid} exceeds RTO ({retention} >= {RTO}) Retransmitted".format(
                    pid=pid, retention=curTime-self.buffer[pid].txTime, RTO=RTO
                ))

                pktList.append(self.buffer[pid].toPacket())

                self.buffer[pid].txTime = curTime
                self.buffer[pid].txAttempts += 1
        
        return pktList
    
    def _genNewPktInfoFromPkt(self, pkt):
        return PacketInfo(
            pid=pkt.pid, 
            suid=pkt.suid, 
            duid=pkt.duid,
            txTime=pkt.txTime,
            genTime=pkt.genTime,
            initTxTime=pkt.txTime, 
            txAttempts=1,
            isFlying=True,
            )

    def __str__(self):
        rst = ""
        for pid in self.buffer:
            if rst != "": rst += "\n"
            rst += self.buffer[pid].__str__()
        
        return rst