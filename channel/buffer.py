# add parent directory to path



import os
import sys
from collections import deque
import heapq
import logging

from typing import List

sys.path.append(os.path.normpath(os.path.join(
    os.path.abspath(__file__), '..', '..')))
from common import BaseRNG, RangeUniform
from common import Packet

class BaseBuffer(object):
    """
    Base class of all specific implementation.
    """
    LOGLEVEL = logging.INFO

    def __init__(self, bufferSize: int = 0, loglevel: int = LOGLEVEL) -> None:
        """
        Assign variables.

        bufferSize: int, the maximum number of objects that can be stored in the buffer.
        delay: int,propagation delay
        verbose: bool, whether to print
        """
        self.bufferSize = 0 if bufferSize < 0 else bufferSize
        self.loglevel = loglevel
        self.nPktsInBuf = 0

    def initLogger(self, loglevel):
        self.logger = logging.getLogger(type(self).__name__)
        sh = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
        sh.setFormatter(formatter)
        self.logger.setLevel(loglevel)
        self.logger.addHandler(sh)

    def isFull(self) -> bool:
        """check whether the channel can still accept packets"""
        return self.bufferSize > 0 and self.nPktsInBuf >= self.bufferSize

    def isEmpty(self) -> bool:
        """check whether the channel is empty"""
        return self.nPktsInBuf == 0

    def size(self) -> int:
        """return the number of packets in buffer"""
        return self.nPktsInBuf

    def enqueue(self, packet: Packet, time: int = 0) -> bool:
        """
        Push the input packet into the queue.

        parameters:
            packet: Packet, the object to be pushed to the end of the queue
            time: int, current time
        output:
            enqueSucc: bool, whether successfully enqueue.
        """
        raise NotImplementedError

    def dequeue(self, time: int = -sys.maxsize) -> List[Packet]:
        """
        Pop out packets that should leave the queue at current time.

        parameters:
            time: int, current time

        outputs:
            pktList: list(Packet), list of packets that satisfy the condition
        """
        raise NotImplementedError


class FIFOBuffer(BaseBuffer):
    """
        FIFOBuffer maintains a FIFO queue. 

                            -------------
       <- deque (popleft)   |x| |x|       <- enque (append)
                            -------------

        If the buffer is full, it will no long accept packets

        When a packet enters the buffer, the enque time is also stored.
        The packet won't exit until its retention time in buffer >= rtt.
    """

    def __init__(self, bufferSize: int = 0, delay: int = 0, loglevel: int = BaseBuffer.LOGLEVEL) -> None:
        """
        bufferSize: int, the maximum number of objects that can be stored in the buffer.
        delay: int,propagation delay
        verbose: bool, whether to print
        """
        super(FIFOBuffer, self).__init__(
            bufferSize=bufferSize, loglevel=loglevel)

        self.delay = delay

        self.FIFOQueue = deque(maxlen=self.bufferSize)
        # used to store the enque time of each packet
        self.timeQueue = deque(maxlen=self.bufferSize)

        self.initLogger(self.loglevel)

    def enqueue(self, packet: Packet, time: int = 0) -> bool:
        """
        Push the packet into the tail of the queue.

        parameters:
            packet: Packet, the object to be pushed to the end of the queue
            time: int, current time
        output:
            enqueSucc: bool, whether successfully enqueue.
        """
        if self.isFull():
            logging.debug("[-] @{time} Queue full {curSize}/{cap} pkt {pkt}".format(
                curSize=self.nPktsInBuf, cap=self.bufferSize, time=time, pkt=packet))
            return False

        self.FIFOQueue.append(packet)
        self.timeQueue.append(time)

        self.nPktsInBuf += 1

        self.logger.debug("[+] @{time} Enqueue {curSize}/{cap} pkt {pkt}".format(
            curSize=self.nPktsInBuf, cap=self.bufferSize, time=time, pkt=packet))

        return True

    def dequeue(self, time: int = -sys.maxsize) -> List[Packet]:
        """
        Pop out packets locating at the top of the queue if their retension time >= delay
        One function call pops out all satisfied packets.

        parameters:
            time: int, current time

        outputs:
            pktList: list(Packet), list of packets that satisfy the condition
        """

        pktList = []
        # the threshold of time before which a packet should be poped out
        timeThres = time - self.delay
        while self.nPktsInBuf and self.timeQueue[0] <= timeThres:
            pktList.append(self.FIFOQueue.popleft())
            _ = self.timeQueue.popleft()
            self.nPktsInBuf -= 1

        if(len(pktList)):
            self.logger.debug("[+] @{time} Dequeu {nPkt} pkts {curSize}/{cap}".format(
                time=time, nPkt=len(pktList), curSize=self.nPktsInBuf, cap=self.bufferSize))
        else:
            self.logger.debug("[-] @{time} Empty queue {curSize}/{cap}".format(
                time=time, curSize=self.nPktsInBuf, cap=self.bufferSize))

        return pktList

    def __str__(self) -> None:
        s = "ChannelBuffer:\n\tbufferType:{bufferType}\n\tcapacity:{capacity}\n\tsize:{size}\n\tdelay:{delay}".format(
            capacity=self.bufferSize, bufferType=type(self).__name__, size=self.nPktsInBuf, delay=self.delay)
        return s


class PriorityBuffer(BaseBuffer):
    """
    Packets in queue are orderred based on its dequeue order.
    Each packet stores in a heap as a tuple (exit time, packet)
    """

    def __init__(self, bufferSize: int = 0, rng: BaseRNG = RangeUniform(2, 10), loglevel: int = BaseBuffer.LOGLEVEL) -> None:
        super(PriorityBuffer, self).__init__(
            bufferSize=bufferSize, loglevel=loglevel)

        self.rng = rng
        self.PriorityQueue = []

        self.initLogger(self.loglevel)

    def enqueue(self, packet: Packet, time: int) -> bool:
        """
        Push the packet into the queue. The order of the packet in queue is determined based on its dequeue time, enqueue time + a random delay.

        parameters:
            packet: Packet, the object to be pushed to the end of the queue
            time: int, current time
        output:
            enqueSucc: bool, whether successfully enqueue.
        """
        if self.isFull():
            self.logger.debug(
                "[-] @{time} Queue full {curSize}/{cap} pkt {pkt}".format(curSize=self.nPktsInBuf, cap=self.bufferSize, time=time, pkt=packet))
            return False

        dequeueTime = time + self.rng.nextNum()
        heapq.heappush(self.PriorityQueue, (dequeueTime, packet))

        self.nPktsInBuf += 1

        self.logger.debug("[+] @{time} Enqueue {curSize}/{cap} pkt {pkt} dequeue @>= {dequeueTime}".format(
            curSize=self.nPktsInBuf, cap=self.bufferSize, time=time, pkt=packet, dequeueTime=dequeueTime))

        return True

    def dequeue(self, time: int) -> List[Packet]:
        """
        Pop out packets in the buffer (heap) that their expected exit time <= current time.

        parameters:
            time: int, current time

        outputs:
            pktList: list(Packet), list of packets that satisfy the condition
        """

        pktList = []
        while self.nPktsInBuf and self.PriorityQueue[0][0] <= time:
            pktList.append(heapq.heappop(self.PriorityQueue)[1])
            self.nPktsInBuf -= 1

        if(len(pktList)):
            self.logger.debug("[+] @{time} Dequeu {nPkt} pkts {curSize}/{cap}".format(time=time,
                                                                                      nPkt=len(pktList), curSize=self.nPktsInBuf, cap=self.bufferSize))
        else:
            self.logger.debug("[-] @{time} Empty queue {curSize}/{cap}".format(time=time,
                                                                               curSize=self.nPktsInBuf, cap=self.bufferSize))

        return pktList


if __name__ == "__main__":
    """Test of FIFO Buffer"""
    print("=====Test of FIFO Buffer=====")
    buffer = FIFOBuffer(bufferSize=3, delay=2, loglevel=logging.DEBUG)

    buffer.enqueue(packet=Packet(pid=0, duid=88), time=0)
    buffer.enqueue(packet=Packet(pid=1, duid=88), time=1)
    buffer.enqueue(packet=Packet(pid=2, duid=88), time=1)
    assert buffer.isFull() == True, "buffer should be full now"
    # expect a full buffer -> drop packet
    buffer.enqueue(packet=Packet(pid=3, duid=88), time=2)

    pktList = buffer.dequeue(time=2)  # expect to see packet 0
    assert len(pktList) == 1, "expect to see only packet 0"
    print(pktList[0])

    # expect to see packet 1, 2, but not 3
    pktList = buffer.dequeue(time=3)
    assert len(pktList) == 2, "expect to see two packets"
    for pkt in pktList:
        print(pkt)
    assert buffer.isEmpty() == True, "buffer should be empty now"
    print("no more packet")
    print("=====End Test of FIFO Buffer=====")

    print("=====Test of Priority Buffer=====")
    import random
    random.seed(0)
    buffer = PriorityBuffer(bufferSize=5, rng=RangeUniform(
        3, 10), loglevel=logging.DEBUG)
    buffer.enqueue(packet=Packet(pid=0, duid=88), time=0)  # deque @ 9
    buffer.enqueue(packet=Packet(pid=1, duid=88), time=0)  # deque @ 6
    buffer.enqueue(packet=Packet(pid=2, duid=88), time=1)  # deque @ 10
    buffer.enqueue(packet=Packet(pid=3, duid=88), time=1)  # deque @ 7
    buffer.enqueue(packet=Packet(pid=4, duid=88), time=2)  # deque @ 5
    assert buffer.isFull() == True, "buffer should be full now"
    # expect to see full buffer message
    buffer.enqueue(packet=Packet(pid=5, duid=88), time=2)  # deque @

    pktList = buffer.dequeue(time=4)  # no pkt is expected
    assert pktList == [], "pktList should be an empty list"

    pktList = buffer.dequeue(time=5)  # should see packet pid=4
    assert len(pktList) == 1, "pktList should only contain one packet"
    print(pktList[0])

    # should see four packets with pid 1 3 0 2
    pktList = buffer.dequeue(time=10)
    for pkt in pktList:
        print(pkt)
    assert buffer.isEmpty() == True, "buffer should be empty now"
    print("=====End Test of Priority Buffer=====")
