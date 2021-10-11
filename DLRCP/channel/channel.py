"""
This class simulates the channel behavior. 

Channel Model:
    We model Channel as a drop tail queue that has a process speed and also a propagation delay.
    If queue is full, drop the packet that would like to be added to the channel. 

    Process time is discretized into time slots.
"""
import random
import os
import sys
import logging


from DLRCP.common import Packet
from DLRCP.common import BaseRNG, RangeUniform
from .buffer import FIFOBuffer, PriorityBuffer

class BaseChannel(object):
    """
    BaseChannel:
        The base class of specific channel. 
    """

    LOGLEVEL = logging.INFO

    # Sanity check
    @classmethod
    def parseQueueSize(cls, _bufferSize: int):
        assert isinstance(_bufferSize, int), "bufferSize must an integer"

        if _bufferSize < 0:
            return -1
        return _bufferSize

    @classmethod
    def checkNonNegInt(cls, _val: int, _desc: str):
        assert isinstance(
            _val, int) and _val >= 0, _desc+" must be an integer >= 0"
        return _val

    @classmethod
    def checkPosInt(cls, _val: int, _desc: str):
        assert isinstance(
            _val, int) and _val > 0, _desc+" must be an integer > 0"
        return _val

    @classmethod
    def checkProb(cls, _val: float, _desc: str):
        assert _val >= 0 and _val <= 1, _desc+" must be an integer > 0"
        return _val

    def __init__(self, serviceRate: int = 1, bufferSize: int = 0, pktDropProb: float = 0, loglevel: int = LOGLEVEL):
        self.serviceRate = BaseChannel.checkPosInt(serviceRate, "serviceRate")
        self.bufferSize = BaseChannel.parseQueueSize(bufferSize)
        self.pktDropProb = BaseChannel.checkProb(pktDropProb, "pktDropProb")
        self.loglevel = loglevel

        self.time = 0
        self.channelBuffer = None

    def initBuffer(self):
        self.time = 0
        if self.channelBuffer:
            self.channelBuffer.clearBuffer()

    def initLogger(self, loglevel):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
            sh.setFormatter(formatter)
            # self.logger.addHandler(sh)

    def ifKeepPkt(self) -> bool:
        return random.random() >= self.pktDropProb

    def isFull(self) -> bool:
        return self.channelBuffer.isFull()

    def isEmpty(self) -> bool:
        return self.channelBuffer.isEmpty()

    def size(self) -> int:
        return self.channelBuffer.size()

    def getChannelTime(self) -> int:
        return self.time

    def setChannelTime(self, time: int):
        self.time = time

    def timeElapse(self):
        self.time += 1

    def __str__(self):
        s = "channel:{channelClass}\n\tserviceRate:{serviceRate}\n\tpktDropProb:{pktDropProb}\n{bufferState}".format(
            channelClass=type(
                self).__name__, serviceRate=self.serviceRate, pktDropProb=self.pktDropProb, bufferState=self.channelBuffer.__str__()
        )
        return s

    """
    channel operations
    """

    def acceptPkts(self, pktList: list = []) -> None:
        """
        Enqueue packets to the channel buffer with the current time. Each packet is randomly dropped with a packet loss probability, and will futher be dropped if the channel buffer is full.
        """
        raise NotImplementedError

    def getPkts(self):
        """
        Return a list of packets that have undergone the propagation delay (self.delay)
        """
        pktList = self.channelBuffer.dequeue(self.time)
        self.logger.debug("[-] @{time} Channel: get {nPktGet}".format(
            time=self.time, nPktGet=len(pktList)))
        return pktList


class ConstDelayChannel(BaseChannel):
    """
    ConstDelayChannel:
        A channel that introduces constant delay
    """

    def __init__(self, serviceRate: int = 1, delay: int = 0, bufferSize: int = 0, pktDropProb: float = 0, loglevel: int = BaseChannel.LOGLEVEL):
        super().__init__(serviceRate=serviceRate, bufferSize=bufferSize,
                         pktDropProb=pktDropProb, loglevel=loglevel)
        self.delay = BaseChannel.checkNonNegInt(delay, "serviceRate")

        self.channelBuffer = FIFOBuffer(
            bufferSize=self.bufferSize, delay=self.delay)

        self.initLogger(loglevel)

    def __str__(self):
        s = "channel:{channelClass}\n\tserviceRate:{serviceRate}\n\tdelay:{delay}\n\tpktDropProb:{pktDropProb}\n{bufferState}".format(
            channelClass=type(
                self).__name__, serviceRate=self.serviceRate, delay=self.delay, pktDropProb=self.pktDropProb, bufferState=self.channelBuffer.__str__()
        )
        return s

    def acceptPkts(self, pktList: list = []) -> None:
        """
        Enqueue packets to the channel buffer with the current time. Each packet is randomly dropped with a packet loss probability, and will futher be dropped if the channel buffer is full.
        """

        nPktAccpt = 0
        nPktFullBufDrop = 0
        availProsPower = self.serviceRate  # number of packets still can be processed
        while availProsPower:
            for pkt in pktList:
                if self.ifKeepPkt():
                    if self.channelBuffer.enqueue(pkt, self.time):
                        nPktAccpt += 1
                        availProsPower -= 1
                    else:
                        nPktFullBufDrop += 1
            break

        nPktDrop = len(pktList) - nPktAccpt - nPktFullBufDrop
        self.logger.debug("[+] @{time} Channel: accept {nPktAccpt}, drop {nPktDrop}".format(
            time=self.time, nPktAccpt=nPktAccpt, nPktDrop=nPktDrop+nPktFullBufDrop))
        self.logger.debug("[-] @{time} Channel: {nPktFullBufDrop} pkts dropped due to full buffer".format(
            time=self.time, nPktFullBufDrop=nPktFullBufDrop))

    def getPkts(self):
        """
        Return a list of packets that have undergone the propagation delay (self.delay)
        """
        pktList = self.channelBuffer.dequeue(self.time)
        self.logger.debug("[-] @{time} Channel: get {nPktGet}".format(
            time=self.time, nPktGet=len(pktList)))
        return pktList


class RandomDelayChannel(BaseChannel):
    """
    Each packet that enters the channel will be given a random delay. 
    """

    def __init__(self, serviceRate: int = 1, delay: int = 0, bufferSize: int = 0, rng: BaseRNG = RangeUniform(1, 10), pktDropProb: float = 0, loglevel: int = BaseChannel.LOGLEVEL):
        super().__init__(serviceRate=serviceRate, bufferSize=bufferSize,
                         pktDropProb=pktDropProb, loglevel=loglevel)
        self.rng = rng

        # self.channelBuffer = FIFOBuffer(bufferSize=self.bufferSize, delay=self.delay)
        self.channelBuffer = PriorityBuffer(
            bufferSize=self.bufferSize, rng=self.rng, loglevel=logging.DEBUG)

        self.initLogger(loglevel)

    def __str__(self):
        s = "channel:{channelClass}\n\tserviceRate:{serviceRate}\n\tdelayRNG:{rng}\n\tpktDropProb:{pktDropProb}\n{bufferState}".format(
            channelClass=type(
                self).__name__, serviceRate=self.serviceRate, rng=self.rng.__str__(), pktDropProb=self.pktDropProb, bufferState=self.channelBuffer.__str__()
        )
        return s

    def acceptPkts(self, pktList: list = []) -> None:
        nPktAccpt = 0
        nPktFullBufDrop = 0
        availProsPower = self.serviceRate  # number of packets still can be processed
        while availProsPower:
            for pkt in pktList:
                if self.ifKeepPkt():
                    if self.channelBuffer.enqueue(pkt, self.time):
                        nPktAccpt += 1
                        availProsPower -= 1
                    else:
                        nPktFullBufDrop += 1
            break

        nPktDrop = len(pktList) - nPktAccpt - nPktFullBufDrop
        self.logger.debug("[+] @{time} Channel: accept {nPktAccpt}, drop {nPktDrop}".format(
            time=self.time, nPktAccpt=nPktAccpt, nPktDrop=nPktDrop+nPktFullBufDrop))
        self.logger.debug("[-] @{time} Channel: {nPktFullBufDrop} pkts dropped due to full buffer".format(
            time=self.time, nPktFullBufDrop=nPktFullBufDrop))

    def getPkts(self):
        return super().getPkts()


if __name__ == "__main__":
    """
    channel = ConstDelayChannel(serviceRate=3, delay=10,
                          bufferSize=3, pktDropProb=0.5, loglevel=logging.DEBUG)
    """
    channel = RandomDelayChannel(serviceRate=3, bufferSize=3, rng=RangeUniform(
        1, 10), pktDropProb=0.5, loglevel=logging.DEBUG)

    print(channel)

    random.seed(1)
    # generate 20 packets from client 0 -> server 1
    pktList1, pktList2 = [], []
    for pid in range(10):
        pktList1.append(Packet(pid=pid, suid=0, duid=1, genTime=0, txTime=0))
        pktList2.append(Packet(pid=pid+10, suid=0,
                        duid=1, genTime=1, txTime=1))

    # @ time = 0
    channel.acceptPkts(pktList1)
    pktList_get = channel.getPkts()
    channel.timeElapse()
    for pkt in pktList_get:
        print("pop: ", pkt)
    print("@time={} feed {} pkts, pop {} pkts".format(channel.getChannelTime(),
          len(pktList1), len(pktList_get)))

    # @ time = 1
    channel.acceptPkts(pktList2)
    pktList_get = channel.getPkts()
    channel.timeElapse()
    print("@ time={} feed {} pkts, pop {} pkts".format(channel.getChannelTime(),
          len(pktList2), len(pktList_get)))

    for _ in range(2, 13):
        channel.acceptPkts()
        pktList_get = channel.getPkts()
        channel.timeElapse()
        if channel.getChannelTime() in {3, 8, 9}:
            assert len(
                pktList_get) == 1, "at time=3, 8, or 9, we expect one packet"

        print("@ time={} feed {} pkts, pop {} pkts".format(
            channel.getChannelTime(), 0, len(pktList_get)))

    print(channel)
