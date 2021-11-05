"""
In this design, application.py only takes care of the Application layer of the 4-layer network stack (not the OSI, but the TCP/IP stack). 
"""
import os
import sys
import numpy as np
import pickle as pkl

from DLRCP.common import Packet, PacketType
from DLRCP.protocols.transportProtocol import BaseTransportLayerProtocol
from DLRCP.protocols.transportLayerHelper import TransportLayerHelper


class EchoClient(object):
    """
    EchoClient in charges of generating packets to sent to the server.
    Support traffic patterns:
    {"periodic", "random"}

    Common attributes:
        id: int, id of the the client. used to simulate the IP address
        serverId: int, whom to talk to
        transportObj: a transport layer instance (e.g. TCP, UDP instance)

    1. periodic: {"period": period, "pktPerPeriod":pktPerPeriod, "startTime":startTime, "lastTime":lastTime}
        $pktPerPeriod packets are periodically generated at the begining of each period. 
        param:
            period:         int
            pktPerPeriod:   int
            startTime:       int, optional, default 0, when to start generating the traffic
            lastTime:       int, optional, default -1, generate the traffic for how long, -1 if forever
    2. poisson: {"lambda":lambda, "startTime":startTime, "lastTime":lastTime}
        Generate traffic following poisson distribution
        param:
            lambda:         int, lambda of the poisson distribution
            startTime:       int, optional, default 0, when to start generating the traffic
            lastTime:       int, optional, default -1, generate the traffic for how long, -1 if forever
    """

    def parseTrafficSettings(self, trafficMode, trafficParam):
        def parseParamByMode(trafficMode, requiredKeys, optionalKeys):
            self.trafficMode = trafficMode

            # required keys
            for key in requiredKeys:
                assert key in trafficParam, key + " is required for " + self.trafficMode
                # setattr(self, key, trafficParam[key])
                self.trafficParam[key] = trafficParam[key]

            # optinal keys
            for key in optionalKeys:
                if key in trafficParam:
                    # setattr(self, key, trafficParam[key])
                    self.trafficParam[key] = trafficParam[key]
                else:
                    # setattr(self, key, optionalKeys[key])
                    self.trafficParam[key] = optionalKeys[key]

        def parsePeriodicParam(trafficParam):
            requiredKeys = {"period", "pktsPerPeriod"}
            optionalKeys = {"startTime": 0, "lastTime": -1}
            parseParamByMode(trafficMode="periodic",
                             requiredKeys=requiredKeys, optionalKeys=optionalKeys)

        def parsePoissonParam(trafficParam):
            requiredKeys = {"lambda"}
            optionalKeys = {"startTime": 0, "lastTime": -1}
            parseParamByMode(
                trafficMode="poisson", requiredKeys=requiredKeys, optionalKeys=optionalKeys)

        trafficParamHandleDict = {
            "periodic": parsePeriodicParam,
            "poisson": parsePoissonParam
        }
        assert isinstance(trafficMode, str), "trafficMode must be string"
        trafficMode = trafficMode.lower()
        assert trafficMode in trafficParamHandleDict, "support traffic modes are " + \
            list(trafficParamHandleDict.keys).__str__()

        trafficParamHandleDict[trafficMode](trafficParam)

    def __init__(self, clientId, serverId, protocolName, transportParam, trafficMode, trafficParam, verbose=False, create_file: bool = False):
        self.uid = clientId
        self.duid = serverId
        self.verbose = verbose

        self.startTime = -1
        self.lastTime = -1
        self.trafficMode = ""
        self.trafficParam = {}
        self.parseTrafficSettings(
            trafficMode=trafficMode, trafficParam=trafficParam)

        self.startTime_default = self.startTime
        self.lastTime_default = self.lastTime

        # the implemented protocol instance
        self.transportObj = TransportLayerHelper(
            suid=self.uid, duid=self.duid, protocolName=protocolName, params=transportParam, verbose=verbose, create_file=create_file)

        # init time
        self.time = -1

        # packet id
        self.pid = 0
        # binding traffic generator

        if self.trafficMode == "periodic":
            self.pktsPerTick = self.trafficParam["pktsPerPeriod"] / \
                self.trafficParam["period"]
        if self.trafficMode == "poisson":
            self.pktsPerTick = self.trafficParam["lambda"]

    def reset(self):
        self.transportObj.instance.reset()
        self.startTime = self.startTime_default
        self.lastTime = self.lastTime_default
        # init time
        self.time = -1

        # packet id
        self.pid = 0

    def ticking(self, ACKPktList=[]):
        self.time += 1
        # generate packets
        pktList = self.trafficGenerator()
        # feed packets to transport layer
        self.transportObj.receiveFromApplication(pktList)

        pktToSend = self.transportObj.sendPkts(ACKPktList)

        return pktToSend

    def trafficGenerator(self):
        def periodicTrafficGenerator():
            if (self.time - self.trafficParam["startTime"]) % self.trafficParam["period"] == 0:
                return self._genNewPkts(self.trafficParam["pktsPerPeriod"])
            else:
                return []

        def poissonTrafficGenerator():
            pktNum = np.random.poisson(lam=self.trafficParam["lambda"])
            return self._genNewPkts(pktNum)

        # check start and end time
        if self.time <= self.trafficParam["startTime"]:
            return []
        if self.trafficParam["lastTime"] != -1 and \
                self.time > self.trafficParam["lastTime"]+self.trafficParam["startTime"]:
            return []

        pktGenHandleDict = {
            "periodic": periodicTrafficGenerator,
            "poisson": poissonTrafficGenerator
        }

        return pktGenHandleDict[self.trafficMode]()

    def _genNewPkts(self, pktNum):
        """Generate $packetNumber new packets"""
        num = 0
        pktList = []
        while num < pktNum:
            pktList.append(
                Packet(
                    pid=self.pid,
                    suid=self.uid,
                    duid=self.duid,
                    genTime=self.time,
                    pktType=PacketType.MSG
                )
            )
            self.pid += 1
            num += 1
        return pktList

    def getDistinctPktSent(self):
        return self.transportObj.getDistinctPktSent()

    def getPktGen(self):
        return self.pid

    def getProtocolName(self):
        return self.transportObj.protocolName

    def getRTT(self):
        return self.transportObj.instance.getRTT()

    def getRTO(self):
        return self.transportObj.instance.getRTO()

    def getCalcUtilityHandler(self):
        return self.transportObj.instance.calcUtility

    def clientSidePerf(self, verbose=False):
        return self.transportObj.instance.clientSidePerf(verbose=verbose)


class EchoServer(object):
    """
    docstring
    """

    def parseACKMode(self, ACKMode):
        if ACKMode == None:
            return ACKMode
        assert isinstance(ACKMode, str), "ACKMode should be None or a string"
        ACKMode = ACKMode.upper()
        assert ACKMode in {
            "LC", "SACK"}, "ACKMode should be None or LC or SACK"
        return ACKMode

    def __init__(self, serverId, ACKMode=None, verbose=False):
        """
        If you would like to save the current dataset for future evaluation, set 
            store_dataset=True
        If you simply want to save the time by reusing the transmission data from previous slot,
        feed the dataset filename, e.g. 
            previous_dataFile="data.pkl"

        """
        self.uid = serverId
        self.verbose = verbose

        # packet ack counter, the latest sequential ack
        self.ACKMode = self.parseACKMode(ACKMode)
        self._inititalize()

    def _inititalize(self):
        self.ack = -1  # last ACKed packet id or maximum packet id

        # time
        self.time = -1

        # performance counter
        self.pktInfo = {}
        self.maxSeenPid = -1

        # a list recording the number of new acks in each tick
        self.pktsPerTick = []
        self.delayPerPkt = []
        self.serverSidePerfRecord = []

        # param to record utility per tick
        self.ack_prev = -1
        self.sumDelay = 0       # used to compute delay for all received packets
        self.sumDelay_prev = 0
        self.deliveredPkts = 0
        self.deliveredPkts_prev = 0
        self.clientSidePid = -1
        self.clientSidePid_prev = -1
        self.perfRecords = []
        self.newPids = set()

        #
        self.loadFromDatafile = False

    def reset(self):
        self._inititalize()

    def storePerf(self, filename, clientPid, distincPktsSent, clientSidePerf):
        # store the current states to dictionary, then to file
        data = {}
        data["perfRecords"] = self.perfRecords
        data["serverSidePerf"] = self.serverSidePerfRecord
        data["clientSidePerf"] = clientSidePerf
        data["clientPid"] = clientPid
        data["distincPktsSent"] = distincPktsSent
        data["pktsPerTick"] = self.pktsPerTick
        data["delayPerPkt"] = self.delayPerPkt
        with open(filename, 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    def calcPerfBasedOnDataFile(self, previous_dataFile, utilityCalcHandler):

        with open(previous_dataFile, 'rb') as f:
            data = pkl.load(f)
        self.perfRecords = data["perfRecords"]
        self.serverSidePerfRecord = data["serverSidePerf"]
        self.pktsPerTick = data["pktsPerTick"]
        self.delayPerPkt = data["delayPerPkt"]

        for idx in range(len(data["perfRecords"])):
            # deliveredPktsInc = self.perfRecords[idx][1]
            deliveryRate = self.perfRecords[idx][2]
            avgDelay = self.perfRecords[idx][3]

            utilPerPkt = utilityCalcHandler(
                delvyRate=deliveryRate, avgDelay=avgDelay,
            )

            self.perfRecords[idx][-1] = utilPerPkt

        self.maxSeenPid = data["distincPktsSent"]

        self.loadFromDatafile = True

        return data["clientSidePerf"], data["distincPktsSent"], data["clientPid"]

    def ticking(self, pktList):
        self.time += 1

        # initialize the number of pkts received to be 0
        self.pktsPerTick.append(0)

        ACKPktList = self._handlePkts(pktList)

        return ACKPktList

    def _handlePkts(self, pktList=[]):
        usefulPktList = []

        for pkt in pktList:
            if pkt.duid != self.uid:
                continue
            usefulPktList.append(pkt)

        if not self.ACKMode:
            return self._handlePktList_None(usefulPktList)
        if self.ACKMode == "SACK":
            return self._handlePktList_SACK(usefulPktList)
        if self.ACKMode == "LC":
            return self._handlePktList_LC(usefulPktList)

    def _handlePktList_None(self, usefulPktList):
        for pkt in usefulPktList:
            if pkt.pid not in self.pktInfo:
                self.pktsPerTick[-1] += 1
                self.delayPerPkt.append(self.time - pkt.genTime)
                self.newPids.add(pkt.pid)

            self.pktInfo[pkt.pid] = self.time - pkt.genTime
            self.maxSeenPid = max(self.maxSeenPid, pkt.pid)
        return []

    def _handlePktList_SACK(self, usefulPktList):
        ACKPacketList = []

        for pkt in usefulPktList:
            if pkt.pid not in self.pktInfo:
                self.pktsPerTick[-1] += 1
                self.delayPerPkt.append(self.time - pkt.genTime)
                self.newPids.add(pkt.pid)

            self.pktInfo[pkt.pid] = self.time - pkt.genTime
            self.maxSeenPid = max(self.maxSeenPid, pkt.pid)

            pkt.duid, pkt.suid = pkt.suid, pkt.duid
            pkt.pktType = PacketType.ACK
            ACKPacketList.append(pkt)
        return ACKPacketList

    def _handlePktList_LC(self, usefulPktList):
        # self.verbose = True
        def updateACK():
            """update self.ack to largest consecutive pkt id"""
            # update self.ACK
            for ack in range(self.ack+1, self.maxSeenPid+1):
                if ack not in self.pktInfo:  # we have cached this packet
                    break
                else:
                    self.ack = ack

        ACKPacketList = []

        # step 1: record all packets
        ACKNewPktList = []  # for printing only
        for pkt in usefulPktList:
            # update ACK. Note that self.ack is the largest consecutive packet id
            if pkt.pid > self.ack:
                # print("ACK @", self.time, " ", pkt.pid)
                if pkt.pid not in self.pktInfo:
                    self.pktsPerTick[-1] += 1
                    self.delayPerPkt.append(self.time - pkt.genTime)

                    self.pktInfo[pkt.pid] = self.time - pkt.genTime
                    self.maxSeenPid = max(self.maxSeenPid, pkt.pid)

                    updateACK()

                    if self.verbose:
                        ACKNewPktList.append(pkt.pid)

            # gen ACK packet
            pkt.duid, pkt.suid = pkt.suid, pkt.duid
            pkt.pktType = PacketType.ACK
            pkt.pid = self.ack
            ACKPacketList.append(pkt)

        if self.verbose and ACKNewPktList:
            print("Server {} LCACK {} by recv @ {}: ".format(self.uid,
                  self.ack, self.time), end="")
            for pid in ACKNewPktList:
                print(" {}".format(pid), end="")
            print()
            print("Server {} sends back {}: ".format(
                self.uid, [pkt.pid for pkt in ACKPacketList]), end="\n")

        return ACKPacketList

    def serverSidePerf(self, clientPid=-1):

        if self.loadFromDatafile:
            return self.serverSidePerfRecord

        if clientPid >= 0:
            # we know that the client tries to send clientPid packets
            self.maxSeenPid = clientPid

        # overall delivery prob
        sumDelay = 0
        if self.ACKMode == "LC":
            deliveredPkts = self.ack+1
            for pid in range(self.ack+1):
                sumDelay += self.pktInfo[pid]
        else:
            deliveredPkts = len(self.pktInfo)
            for pid in self.pktInfo:
                sumDelay += self.pktInfo[pid]

        # deal with divide by 0 problem
        if deliveredPkts != 0:
            avgDelay = sumDelay / deliveredPkts
        else:
            avgDelay = 0

        if self.maxSeenPid != 0:
            deliveryRate = deliveredPkts/(self.maxSeenPid)
        else:
            deliveryRate = 0

        self.serverSidePerfRecord = [deliveredPkts, deliveryRate, avgDelay]
        return deliveredPkts, deliveryRate, avgDelay

    def recordPerfInThisTick(self, clientPid=-1, utilityCalcHandler=None):
        # used to record the utility increment
        if clientPid >= 0:
            # we know that the client tries to send clientPid packets
            self.clientSidePid = clientPid
        else:
            self.clientSidePid = self.maxSeenPid

        # overall delivery prob

        if self.ACKMode == "LC":
            self.deliveredPkts = self.ack+1
            for pid in range(self.ack_prev+1, self.ack+1):
                self.sumDelay += self.pktInfo[pid]
        else:
            self.deliveredPkts = len(self.pktInfo)
            for pid in self.newPids:
                self.sumDelay += self.pktInfo[pid]
            self.newPids.clear()

        delayInc = self.sumDelay - self.sumDelay_prev
        deliveredPktsInc = self.deliveredPkts - self.deliveredPkts_prev

        clientDeliveredPktsInc = self.clientSidePid - self.clientSidePid_prev

        # deal with divide by 0 problem
        if deliveredPktsInc != 0:
            avgDelay = delayInc / deliveredPktsInc
        else:
            avgDelay = 0

        if clientDeliveredPktsInc != 0:
            deliveryRate = deliveredPktsInc/(clientDeliveredPktsInc)
        else:
            deliveryRate = 0

        self.sumDelay_prev = self.sumDelay
        self.clientSidePid_prev = self.clientSidePid
        self.deliveredPkts_prev = self.deliveredPkts

        record = [self.time, deliveredPktsInc, deliveryRate, avgDelay, 0]
        if utilityCalcHandler:
            record[-1] = utilityCalcHandler(
                delvyRate=deliveryRate, avgDelay=avgDelay,
            )
        # delvyRate: float, avgDelay
        # print("In this ", record)
        self.perfRecords.append(record)

        return

    def printPerf(self, clientPid=-1, clientProtocolName=""):
        # for LC algorithm, delivered packets is ACK+1.
        deliveredPkts, deliveryRate, avgDelay = self.serverSidePerf(clientPid)
        print("Server {} {} Performance:".format(self.uid, clientProtocolName))
        print("\tpkts received  %d out of %d" %
              (deliveredPkts, self.maxSeenPid+1))
        print("\tdelivery rate  {}% ".format(deliveryRate*100))
        print("\taverage delay  {}".format(avgDelay))
