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
        self.pid = int(pid)
        self.suid = int(suid)
        self.duid = int(duid)
        self.genTime = int(genTime)
        self.txTime = int(txTime)
        self.pktType = PacketType(pktType)

    def __str__(self):
        s = "{suid} -> {duid} pid:{pid} tx@:{txTime} gen@:{genTime} type: {type}".format(
            suid=self.suid, duid=self.duid,
            pid=self.pid,
            txTime=self.txTime, genTime=self.genTime,
            type=self.pktType.name)
        return s

    def toPktInfo(self, initTxTime=0, txAttempts=0, isFlying=True, RLState=[]):
        return PacketInfo(
            pid=self.pid, suid=self.suid, duid=self.duid, genTime=self.genTime, txTime=self.txTime, initTxTime=initTxTime, txAttempts=txAttempts, isFlying=isFlying, RLState=RLState, pktType=self.pktType
        )

    def __lt__(self, other) -> bool:
        return (self.txTime < other.txTime)


class PacketInfo(Packet):
    """
    A data structure storing more details about a packet, except its payload.
    """

    def __init__(self, pid=0, suid=0, duid=0, genTime=0, txTime=0, initTxTime=0, txAttempts=0, isFlying=True, RLState=[], pktType=PacketType.MSG):
        super().__init__(pid=pid, suid=suid, duid=duid,
                         genTime=genTime, txTime=txTime, pktType=pktType)

        self.initTxTime = int(initTxTime)
        self.txAttempts = int(txAttempts)
        self.isFlying = isFlying
        self.RLState = RLState

    def toPacket(self):
        return Packet(
            pid=self.pid, suid=self.suid, duid=self.duid, genTime=self.genTime, txTime=self.txTime, pktType=self.pktType
        )


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
