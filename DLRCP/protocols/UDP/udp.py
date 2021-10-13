from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol

class UDP(BaseTransportLayerProtocol):
    """
    docstring
    """
    def __init__(self, suid, duid, params={}, loglevel=BaseTransportLayerProtocol.LOGLEVEL):
        super().__init__(suid=suid, duid=duid, params=params, loglevel=loglevel)
    
    def ticking(self, ACKPktList=[]):
        """
        Add new pkts to transmit
        """

        self.timeElapse() # time +1

        pktToSendList = []

        # udp sends out all packets in the buffer
        self.perfDict["distinctPktsSent"] += len(self.txBuffer)
        
        while self.txBuffer:
            pkt = self.txBuffer.popleft()
            pkt.txTime = self.time # label the txTime
            pkt.initTxTime = self.time

            pktToSendList.append(pkt)
        

        self.logger.debug("[+] Client {suid}->{duid} @ {time} transmits {nNewPkt} pkts {pids}".format(
            suid = self.suid,
            duid = self.duid,
            time = self.time,
            nNewPkt = len(pktToSendList),
            pids = " ".join([str(pkt.pid) for pkt in pktToSendList])
        ))
        
        
        return pktToSendList