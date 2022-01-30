import os
import csv
import json
import logging
import argparse
import subprocess
import numpy as np
import pickle as pkl
from tabulate import tabulate


from DLRCP.applications import EchoClient, EchoServer
from DLRCP.channel import ConstDelayChannel, RandomDelayChannel
from DLRCP.common import RangeUniform

def get_opts():
    parser = argparse.ArgumentParser(description='RCP-Ver 2.0')

    # load configuration file
    parser.add_argument('--configFile', type=str, default='',
                        help='configuration file (.json) to be loaded')

    # description
    parser.add_argument('--testDesc', type=str, default='test',
                        help='description of the test, also the folder name to store the result')
    # Paths
    parser.add_argument('--data-dir', type=str, default='./Results',
                        help='result data folder')
    parser.add_argument('--nonRCPDatadir', type=str, default='',
                        help='the folder to store the temp data for Non-RCP protocol test result. Default to be [data-dir]/[testDesc]/')

    # Channel Type
    parser.add_argument('--channelType', type=str, default='RandomDelayChannel', choices=['RandomDelayChannel', 'ConstDelayChannel'],
                        help='type of the channel RandomDelayChannel or ConstDelayChannel')
    parser.add_argument('--serviceRate', type=int, default=3,
                        help='channel service rate')
    parser.add_argument('--bufferSize', type=int, default=300,
                        help='buffer size of channel')
    parser.add_argument('--pktDropProb', type=float, default=0.1,
                        help='packet loss probability of the channel')
    parser.add_argument('--channelDelay', nargs='+', type=int,
                        default=[100, 150],
                        help='a list of delay used by the selected channel. If RandomDelayChannel, a list of the min&max of the random number. If ConstDelayChannel, one integer is needed')
    parser.add_argument('--fillChannel', dest='fillChannel', default=False,
                        action='store_true', help="whether to fill the channel with environment packets")
    parser.add_argument('--channelInstruct', dest='chInsList', default="",
                        help="scheduled instructions for the channel to execute. E.g. 200 serviceRate 3 400 channelDelay 120,140 ")

    # environment setting
    parser.add_argument('--bgClientNum', type=int, default=4,
                        help='# of UDP clients to simulate the background traffic')
    parser.add_argument('--bgClientPktRate', type=int, default=1,
                        help='# pkt to send by the background UDP client per tick')

    # utility setting
    parser.add_argument('--utilityMethod', type=str, default='TimeDiscount', choices=[
                        'TimeDiscount', 'SumPower'], help="utility method. 'TimeDiscount', 'SumPower'")
    parser.add_argument('--timeDivider', type=float, default=100,
                        help='ticks per time unit')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='exponent index of delivery rate (and delay)')
    # timeDiscountUtility -- utility = beta^(delay/timeDivision) * deliveryRate^alpha
    # sum of power utility -- utility = beta * (delay/timeDivision)^alpha + (1-beta) * deliveryRate^alpha
    parser.add_argument('--beta', type=float, default=0.5,
                        help='reward discount over time (used when calculating utility)')

    # Test protocol setting
    parser.add_argument('--pktRate', type=int, default=1,
                        help='number of pkts generated per tick for the client to be tested')

    parser.add_argument('--testPeriod', type=int, default=20000,
                        help='simulation period')

    # UDP
    parser.add_argument('--addUDP', default=False, action='store_true',
                        help="whether to add UDP to test")

    # ARQ finite window
    parser.add_argument('--addARQFinite', default=False, action='store_true',
                        help="whether to add ARQ with finite window to test")
    parser.add_argument('--ARQWin', type=int, default=200,
                        help='maximum window size of ARQ Finite')

    # ARQ infinite window
    parser.add_argument('--addARQInfinite', default=False, action='store_true',
                        help="whether to add ARQ with infinite window to test")

    # TCP NewReno
    parser.add_argument('--addNewReno', default=False, action='store_true',
                        help="whether to add TCP New Reno to test")

    # RCP-QLearning
    parser.add_argument('--addRCPQLearning', default=False, action='store_true',
                        help="whether to add RCP-QLearning to test")
    parser.add_argument('--RCPQLearningGamma', type=float, default=0.9,
                        help='reward decay coefficient')
    parser.add_argument('--RCPQLearningEpsilon', type=float, default=0.7,
                        help='DQN greedy policy epsilon')
    parser.add_argument('--RCPQLearningEpsilonDecay', type=float, default=0.9,
                        help='DQN greedy policy epsilon decay')

    # RCP-DQN
    parser.add_argument('--addRCPDQN', default=False, action='store_true',
                        help="whether to add RCP-QLearning to test")
    parser.add_argument('--RCPDQNGamma', type=float, default=0.9,
                        help='reward decay coefficient')
    parser.add_argument('--RCPDQNEpsilon', type=float, default=0.7,
                        help='DQN greedy policy epsilon')
    parser.add_argument('--RCPDQNEpsilonDecay', type=float, default=0.9,
                        help='DQN greedy policy epsilon decay')

    # RCP-RTQ
    parser.add_argument('--addRCPRTQ', default=False, action='store_true',
                        help="whether to add RCP-RTQ to test")

    # Additional options
    parser.add_argument('--clean-run', dest='clean_run',
                        default=False, action='store_true')

    ##
    opts = parser.parse_args()

    # load config file is specified
    opts = _loadOpts(opts)

    return opts


def saveOpts(opts):
    tgtPath = os.path.join(opts.data_dir, opts.testDesc)

    with open(os.path.join(tgtPath, "test_config.json"), 'w') as handle:
        json.dump(opts.__dict__, handle, indent=4)


def _loadOpts(opts):
    if opts.configFile:
        print("loading configurations from", opts.configFile)
        with open(opts.configFile, 'r') as f:
            opts.__dict__ = json.load(f)
    return opts


def prepareDataStorageFolder(opts):
    tgtPath = os.path.join(opts.data_dir, opts.testDesc)
    os.makedirs(tgtPath, exist_ok=True)

    if opts.nonRCPDatadir:
        tgtPath = os.path.join(opts.nonRCPDatadir)
        os.makedirs(tgtPath, exist_ok=True)


def cleanPrevDataFiles(opts):
    if opts.clean_run:
        print("cleaning all stored files")
        tgtPath = os.path.join(opts.data_dir, opts.testDesc)
        subprocess.run(["rm", os.path.join(tgtPath, "*.pkl")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.txt")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.csv")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.json")], shell=True)

        tgtPath = os.path.join(opts.nonRCPDatadir)
        subprocess.run(["rm", os.path.join(tgtPath, "*.pkl")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.txt")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.csv")], shell=True)
        subprocess.run(["rm", os.path.join(tgtPath, "*.json")], shell=True)


def genChannel(opts):
    assert opts.channelType in {
        "RandomDelayChannel", "ConstDelayChannel"}, "channelType not supported. Should be either RandomDelayChannel or ConstDelayChannel"

    if opts.channelType == "RandomDelayChannel":
        assert len(opts.channelDelay) >= 2, "If using RandomDelayChannel, the specified delay must be two integers, which are the min and max of the delay"
        channel = RandomDelayChannel(
            serviceRate=opts.serviceRate,
            bufferSize=opts.bufferSize,
            rng=RangeUniform(opts.channelDelay[0], opts.channelDelay[1]),
            pktDropProb=opts.pktDropProb,
            insList=opts.chInsList,
            loglevel=logging.INFO)
    if opts.channelType == "ConstDelayChannel":
        channel = ConstDelayChannel(
            serviceRate=opts.serviceRate,
            bufferSize=opts.bufferSize,
            delay=opts.channelDelay[0],
            pktDropProb=opts.pktDropProb,
            insList=opts.chInsList,
            loglevel=logging.INFO)

    return channel


def genBgClientsAndServers(opts):
    env_clients = []
    env_servers = []
    for clientId in range(1, opts.bgClientNum+1):

        client = EchoClient(clientId=clientId, serverId=1000+clientId,
                            protocolName="UDP",
                            transportParam={
                                # utility
                                "utilityMethod": opts.utilityMethod,
                                "alpha": opts.alpha,
                                "timeDivider": opts.timeDivider,
                                "beta": opts.beta,
                            },
                            trafficMode="periodic",
                            trafficParam={
                                "period": 1, "pktsPerPeriod": opts.bgClientPktRate},
                            verbose=False)
        server = EchoServer(serverId=1000+clientId,
                            ACKMode=None, verbose=False)

        env_clients.append(client)
        env_servers.append(server)
    return env_clients, env_servers


def genTestUDP(opts):
    if opts.addUDP:
        client_UDP = EchoClient(
            clientId=101, serverId=111,
            protocolName="UDP", transportParam={
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_UDP = EchoServer(serverId=111, ACKMode=None, verbose=False)
        return [client_UDP], [server_UDP]
    return [], []


def genTestARQFiniteWindow(opts):
    if opts.addARQFinite:
        client_ARQ_finit = EchoClient(
            clientId=201, serverId=211,
            protocolName="window arq", transportParam={
                "cwnd": opts.ARQWin, "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1,
                "ACKMode": "SACK",
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_ARQ_finit = EchoServer(
            serverId=211, ACKMode="SACK", verbose=False)
        return [client_ARQ_finit], [server_ARQ_finit]
    return [], []


def genTestARQinFiniteWindow(opts):
    if opts.addARQInfinite:
        client_ARQ_infinit = EchoClient(
            clientId=301, serverId=311,
            protocolName="window arq", transportParam={
                "cwnd": -1, "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1,
                "ACKMode": "SACK",
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_ARQ_infinit = EchoServer(
            serverId=311, ACKMode="SACK", verbose=False)
        return [client_ARQ_infinit], [server_ARQ_infinit]
    return [], []


def genTCPNewReno(opts):
    if opts.addNewReno:
        client_TCP_Reno = EchoClient(
            clientId=401, serverId=411,
            # IW=2 if SMSS>2190, IW=3 if SMSS>3, else IW=4
            protocolName="tcp_newreno", transportParam={
                "timeout": 30, "IW": 4,
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_TCP_Reno = EchoServer(serverId=411, ACKMode="LC", verbose=False)
        return [client_TCP_Reno], [server_TCP_Reno]
    return [], []


def genRCPQLearning(opts):
    if opts.addRCPQLearning:
        client_RL_Q = EchoClient(
            clientId=501, serverId=511,
            protocolName="RCP",
            transportParam={
                "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1,
                "utilityMethod": opts.utilityMethod,
                "RLEngine": "Q_Learning",
                "gamma": opts.RCPQLearningGamma,
                "epsilon": opts.RCPQLearningEpsilon,
                "epsilon_decay": opts.RCPQLearningEpsilonDecay,
                # whether only learn the data related to retransmission
                "learnRetransmissionOnly": False,
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False, create_file=False)
        server_RL_Q = EchoServer(serverId=511, ACKMode="SACK", verbose=False)
        return [client_RL_Q], [server_RL_Q]
    return [], []


def genRCPDQN(opts):
    if opts.addRCPDQN:
        client_RL_DQN = EchoClient(
            clientId=601, serverId=611,
            protocolName="RCP",
            transportParam={
                "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1,
                "RLEngine": "DQN",
                "gamma": opts.RCPDQNGamma,
                "epsilon": opts.RCPDQNEpsilon,
                "epsilon_decay": opts.RCPDQNEpsilonDecay,
                # whether only learn the data related to retransmission
                "learnRetransmissionOnly": False,
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_RL_DQN = EchoServer(serverId=611, ACKMode="SACK", verbose=False)
        return [client_RL_DQN], [server_RL_DQN]
    return [], []


def genRCPRTQ(opts):
    if opts.addRCPRTQ:
        client_RL_RTQ = EchoClient(
            clientId=701, serverId=711,
            protocolName="RCP",
            transportParam={
                "maxTxAttempts": 10, "timeout": 30, "maxPktTxDDL": -1,
                "utilityMethod": opts.utilityMethod,
                "RLEngine": "RTQ",
                "gamma": opts.RCPQLearningGamma,
                "epsilon": opts.RCPQLearningEpsilon,
                "epsilon_decay": opts.RCPQLearningEpsilonDecay,
                # whether only learn the data related to retransmission
                "learnRetransmissionOnly": False,
                # utility
                "utilityMethod": opts.utilityMethod,
                "alpha": opts.alpha,
                "timeDivider": opts.timeDivider,
                "beta": opts.beta,
            },
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_RL_RTQ = EchoServer(serverId=711, ACKMode="SACK", verbose=False)
        return [client_RL_RTQ], [server_RL_RTQ]
    return [], []


def fillChannel(opts, channel, bgClients):
    if opts.fillChannel:
        print("filling the channel with background packets")
        channel.initBuffer()

        while not channel.isFull():  # fill the channel with environment packets
            packetList_enCh = []
            for client in bgClients:
                packetList_enCh += client.ticking([])
            channel.acceptPkts(packetList_enCh)


def test_protocol(opts, channel, test_client, test_server, env_clients, env_servers, loadFromHistoryIfPossible=False):
    ignored_pkt, retrans_pkt, retransProb = 0, 0, 0

    serverPerfFilename = test_client.getProtocolName()+"_perf.pkl"

    if loadFromHistoryIfPossible and test_client.getProtocolName().lower() not in {"rcpdqn", "rcpq_learning", "rcprtq"}:
        if opts.nonRCPDatadir:  # stored in the specified temp folder
            serverPerfFilename = os.path.join(
                opts.nonRCPDatadir, serverPerfFilename)
        else:  # same as result folder
            serverPerfFilename = os.path.join(
                opts.data_dir, opts.testDesc, serverPerfFilename)

        # check whether can load the previous performance file directly
        if os.path.exists(serverPerfFilename):
            print("found file ", serverPerfFilename)
            clientSidePerf, distincPktsSent, clientPid = test_server.calcPerfBasedOnDataFile(
                serverPerfFilename,
                utilityCalcHandler=test_client.transportObj.instance.calcUtility,
            )

            # hacking the final state
            test_client.pid = clientPid
            test_client.transportObj.instance.distincPktsSent = distincPktsSent
            test_client.transportObj.instance.perfDict = clientSidePerf.copy()
            return
    else:
        serverPerfFilename = os.path.join(
            opts.data_dir, opts.testDesc, serverPerfFilename)

    channel.initBuffer()

    fillChannel(opts, channel, env_clients)

    channel.time = 0

    clientList = env_clients + [test_client]
    serverList = env_servers + [test_server]

    ACKPacketList = []
    packetList_enCh = []
    packetList_deCh = []

    # clear each client server
    for test_client, test_server in zip(clientList, serverList):
        test_client.reset()
        test_server.reset()

    packetList_enCh = []
    for time in range(1, opts.testPeriod+1):
        ACKPacketList = []
        # step 1: each server processes remaining pkts
        for serverId in range(len(serverList)):
            ACKPacketList += serverList[serverId].ticking(packetList_deCh)

        # step 2: clients generate packets
        packetList_enCh = []
        # for client in clientSet:
        for clientId in np.random.permutation(len(clientList)):
            packetList_enCh += clientList[clientId].ticking(ACKPacketList)

        # step 3: feed packets to channel
        # ACKPacketList += channel.putPackets(packetList_enCh) # allow channel feedback
        channel.acceptPkts(packetList_enCh)
        channel.timeElapse()  # channel.time ++

        # step 3: get packets from channel
        packetList_deCh = channel.getPkts()
        #print("get pkts", len(packetList_deCh))

        if time % 30 == 0:  # record performance for the past 30 ticks
            test_server.recordPerfInThisTick(
                test_client.getPktGen(),
                utilityCalcHandler=test_client.getCalcUtilityHandler()
            )

        if time % (opts.testPeriod//10) == 0:
            print("time ", time, " =================")
            print("RTT", test_client.getRTT())
            print("RTO", test_client.getRTO())
            # client.clientSidePerf(verbose=False)
            test_server.printPerf(
                test_client.getPktGen(),
                test_client.getProtocolName())

            if test_client.getProtocolName().lower() in {"rcpdqn", "rcpq_learning", 'rcprtq'}:
                # we store extra more stuff for rcp
                clientPerfDict = test_client.clientSidePerf()
                ignored_pkt = clientPerfDict["ignorePkts"] - ignored_pkt
                retrans_pkt = clientPerfDict["retransAttempts"] - retrans_pkt
                if retrans_pkt == 0:
                    retransProb = 0
                else:
                    retransProb = retrans_pkt / (retrans_pkt + ignored_pkt)
                # debug
                print("retransProb", retransProb)
                print("epsilon", clientPerfDict["epsilon"])
                print("loss", clientPerfDict["loss"])
                # client.transportObj.instance.perfDict["retranProb"] = retransProb

    test_server.storePerf(serverPerfFilename,
                          clientPid=test_client.pid,
                          distincPktsSent=test_client.getPktGen(),
                          clientSidePerf=test_client.clientSidePerf())


def printTestProtocolPerf(opts, test_clients, test_servers, storePerfBrief=True):
    header = ["ptcl", "pkts gen", "pkts sent", "pkts dlvy", "dlvy perc", "avg dly",
              "sys util", "l25p dlvy", "l25p dlvy perc", 'l25p dly', "l25p util", "loss"]
    table = []

    deliveredPktsPerSlot = dict()
    deliveredPktsPerSlot["protocols"] = []

    for client, server in zip(test_clients, test_servers):  # ignore the first two
        deliveredPktsPerSlot["protocols"].append(client.getProtocolName())
        deliveredPktsPerSlot[client.getProtocolName()] = dict()

        server.printPerf(client.getPktGen(), client.getProtocolName())
        clientPerfDict = client.transportObj.instance.clientSidePerf(
            verbose=True)

        # store data
        deliveredPktsPerSlot[client.getProtocolName()]["serverPerf"] = [
            server.pktsPerTick, server.delayPerPkt, server.perfRecords]
        deliveredPktsPerSlot[client.getProtocolName(
        )]["clientPerf"] = clientPerfDict

        # for display
        deliveredPkts, delvyRate, avgDelay = server.serverSidePerf(
            client.getPktGen())
        last25percTime = int(opts.testPeriod*0.25)
        last25percPkts = sum(server.pktsPerTick[-last25percTime:])
        last25percDelveyRate = last25percPkts / \
            (client.pktsPerTick*last25percTime)
        if(last25percPkts == 0):
            print(client.getProtocolName(), "have zero packts delivered ")
            last25percDelay = -1
        else:
            last25percDelay = sum(
                server.delayPerPkt[-last25percPkts:]) / last25percPkts
        last25percUtil = client.transportObj.instance.calcUtility(
            delvyRate=last25percDelveyRate, avgDelay=last25percDelay)

        table.append([client.getProtocolName(),
                      client.pid,
                      client.getPktGen(),
                      deliveredPkts,
                      delvyRate,
                      avgDelay,
                      client.transportObj.instance.calcUtility(
                          delvyRate=delvyRate, avgDelay=avgDelay),
                      last25percPkts,
                      last25percDelveyRate,
                      last25percDelay,
                      last25percUtil,
                      clientPerfDict["loss"]
                      ])

    deliveredPktsPerSlot["general"] = table
    deliveredPktsPerSlot["header"] = header
    print(tabulate(table, headers=header, floatfmt=".3f"))

    if storePerfBrief:
        tgtPath = os.path.join(opts.data_dir, opts.testDesc)
        """
        save briefing of performance
        """
        with open(os.path.join(tgtPath, 'perfBrief.txt'), 'w') as handle:
            handle.writelines(tabulate(table, headers=header, floatfmt=".3f"))
        with open(os.path.join(tgtPath, 'perfBrief.csv'), 'w') as handle:
            csvWriter = csv.writer(handle, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow(header)
            csvWriter.writerows(table)
    return table, header
