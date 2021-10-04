from enum import Enum


class PacketType(Enum):
    MSG = 0
    ACK = 1
    NACK = 2

class Packet(object):
    """
    Packet:
        A class that defines the packets used in the simulation
    """

    # Packet Type

    def __init__(self, pid=0, suid=0, duid=0, genTime=0, txTime=0, pktType=PacketType.MSG):
        self.pid = pid
        self.suid = suid
        self.duid = duid
        self.genTime = genTime
        self.txTime = txTime
        self.pktType = pktType

    def __str__(self):
        s = "{suid} -> {duid} pid:{pid} tx@:{txTime} gen@:{genTime} type: {type}".format(
            suid=self.suid, duid=self.duid,
            pid=self.pid,
            txTime=self.txTime, genTime=self.genTime,
            type=self.pktType.name)
        return s
    
    def __lt__(self, other):
        return (self.txTime < other.txTime)
    
if __name__ == "__main__":
    pkt1 = Packet(pid=1, suid=101, duid=111, genTime=500,
                  txTime=1000, pktType=PacketType.MSG)
    pkt2 = Packet(pid=2, suid=102, duid=112, genTime=501,
                  txTime=1001, pktType=PacketType.ACK)
    pkt3 = Packet(pid=3, suid=103, duid=113, genTime=502,
                  txTime=1002, pktType=PacketType.NACK)

    print(pkt1)
    print(pkt2)
    print(pkt3)
