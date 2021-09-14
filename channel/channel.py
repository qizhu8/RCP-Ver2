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


# add parent directory to path
sys.path.append(os.path.normpath(os.path.join(
    os.path.abspath(__file__), '..', '..')))
    
from common import Packet
from buffer import FIFOBuffer




class BaseChannel(object):
    """
    BaseChannel:
        The base class of specific channel. 
    """

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

    def ifKeepPkt(self):
        return random.random() <= self.pktDropProb

    def __init__(self, serviceRate: int = 1, delay: int = 0, bufferSize: int = 0, pktDropProb: float = 0, verbose: bool = False):
        self.serviceRate = BaseChannel.checkPosInt(serviceRate, "serviceRate")
        self.delay = BaseChannel.checkNonNegInt(delay, "serviceRate")
        self.bufferSize = BaseChannel.parseQueueSize(bufferSize)
        self.pktDropProb = BaseChannel.checkProb(pktDropProb, "pktDropProb")
        self.verbose = verbose

        self.time = 0
        self.channelBuffer = FIFOBuffer(
            bufferSize=self.bufferSize, delay=self.delay, verbose=self.verbose)

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
    
    def __str__(self):
        s = "channel:{channelClass}\n\tserviceRate:{serviceRate}\n\tdelay:{delay}\n\tpktDropProb:{pktDropProb}\n-{bufferState}".format(
            channelClass=type(self).__name__, serviceRate=self.serviceRate, delay=self.delay, pktDropProb=self.pktDropProb, bufferState=self.channelBuffer.__str__()
        )
        return s

    """
    channel operations
    """
    def acceptPkts(self, pktList: list=[]) -> None:
        """
        Enqueue packets to the channel buffer with the current time. Each packet is randomly dropped with a packet loss probability, and will futher be dropped if the channel buffer is full.
        """

        nPktAccpt = 0
        nPktFullBufDrop = 0
        availProsPower = self.serviceRate # number of packets still can be processed
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
        if self.verbose:
            print("[+] @{time} Channel: accept {nPktAccpt}, drop {nPktDrop}".format(
                time=self.time, nPktAccpt=nPktAccpt, nPktDrop=nPktDrop+nPktFullBufDrop))
            print("[-] @{time} Channel: {nPktFullBufDrop} pkts dropped due to full buffer".format(
                time=self.time, nPktFullBufDrop=nPktFullBufDrop))

    def getPkts(self):
        """
        Return a list of packets that have undergone the propagation delay (self.delay)
        """
        return self.channelBuffer.dequeue(self.time)
    
    def timeElapse(self):
        self.time += 1


if __name__ == "__main__":
    channel = BaseChannel(serviceRate=3, delay=10,
                          bufferSize=3, pktDropProb=0.5, verbose=False)
    print(channel)

    random.seed(1)
    # generate 20 packets from client 0 -> server 1
    pktList1, pktList2 = [], []
    for pid in range(10):
        pktList1.append(Packet(pid=pid, suid=0, duid=1, genTime=0, txTime=0))
        pktList2.append(Packet(pid=pid, suid=0, duid=1, genTime=1, txTime=1))
    
    # @ time = 0
    channel.acceptPkts(pktList1)
    pktList_get = channel.getPkts()
    channel.timeElapse()
    for pkt in pktList_get:
        print("pop: ", pkt)
    print("@ time={} feed {} pkts, pop {} pkts".format(channel.getChannelTime(), len(pktList1), len(pktList_get)))

    # @ time = 1
    channel.acceptPkts(pktList2)
    pktList_get = channel.getPkts()
    channel.timeElapse()
    print("@ time={} feed {} pkts, pop {} pkts".format(channel.getChannelTime(), len(pktList2), len(pktList_get)))
    
    for _ in range(2, 13):
        channel.acceptPkts()
        pktList_get = channel.getPkts()
        channel.timeElapse()
        print("@ time={} feed {} pkts, pop {} pkts".format(channel.getChannelTime(), 0, len(pktList_get)))

    print(channel)
