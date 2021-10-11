"""
Different from the scenario in SimulationEnvironment.py, all protocols to be evaluated are not going to compete with each other. At a time only the one protocol is going to be tested.
"""
import os
import sys
import logging
import numpy as np
import pickle as pkl
from tabulate import tabulate
import matplotlib.pyplot as plt


from DLRCP.applications import EchoClient, EchoServer
from DLRCP.channel import ConstDelayChannel, RandomDelayChannel
from DLRCP.common import RangeUniform

if len(sys.argv) > 3:
    alpha = int(sys.argv[1])
    beta1 = float(sys.argv[2])
    beta2 = float(sys.argv[3])
else:
    alpha = 2       # quadratic
    beta1 = 0.8     # emphasis on delivery
    beta2 = 0.2     # emphasis on delay
    timeDiscount = 0.9  # discount of time onto the reward
    # util = timeDiscount^(delay / timeDivider) * delivery^alpha
    timeDivider = 100

alpha = np.round(alpha, 2)

print("beta1={beta1}, beta2={beta2}".format(beta1=beta1, beta2=beta2))

# utilityCalcHandlerParams = {"beta1": beta1, "beta2": beta2, "alpha": alpha}
if len(sys.argv) > 4:
    pklFilename = sys.argv[4]
else:
    pklFilename = "perfData2_{alpha}_{beta1}_{beta2}.pkl".format(
        alpha=alpha, beta1=beta1, beta2=beta2)

print("results save to \\result\\"+pklFilename)


if len(sys.argv) > 5:
    simulationPeriod = int(sys.argv[5])  # unit ticks / time slots
else:
    simulationPeriod = int(20000)  # unit ticks / time slots

"""
add background traffic
"""
env_clients = []
env_servers = []
for clientId in range(1, 4+1):

    client = EchoClient(clientId=clientId, serverId=10+clientId,
                        protocolName="UDP", transportParam={},
                        trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                        verbose=False)
    server = EchoServer(serverId=10+clientId, ACKMode=None, verbose=False)

    env_clients.append(client)
    env_servers.append(server)


"""
Protocols to compare
"""
client_RL = EchoClient(clientId=101, serverId=111,
                       protocolName="RCP",
                       transportParam={"maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1,
                                       "alpha": alpha,
                                       # beta1: emphasis on delivery, beta2: emphasis on delay
                                       "beta1": beta1, "beta2": beta2,
                                       "timeDiscount": timeDiscount,
                                       "timeDivider": timeDivider,
                                       # "RLEngine": "Q_Learning",
                                       "RLEngine": "DQN",
                                       "gamma": 0.9,
                                       "learnRetransmissionOnly": True},  # whether only learn the data related to retransmission
                       trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                       verbose=False)
server_RL = EchoServer(serverId=111, ACKMode="SACK", verbose=False)

client_ARQ_finit = EchoClient(clientId=201, serverId=211,
                              protocolName="window arq", transportParam={"cwnd": 140, "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1, "ACKMode": "SACK",
                                                                         "alpha": alpha,
                                                                         # beta1: emphasis on delivery, beta2: emphasis on delay
                                                                         "beta1": beta1, "beta2": beta2,
                                                                         "timeDiscount": timeDiscount,
                                                                         "timeDivider": timeDivider,
                                                                         },
                              trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                              verbose=False)
server_ARQ_finit = EchoServer(serverId=211, ACKMode="SACK", verbose=False)

client_ARQ_infinit_cwnd = EchoClient(clientId=301, serverId=311,
                                     protocolName="window arq", transportParam={"cwnd": -1, "maxTxAttempts": -1, "timeout": 30, "maxPktTxDDL": -1, "ACKMode": "SACK",
                                                                                "alpha": alpha,
                                                                                # beta1: emphasis on delivery, beta2: emphasis on delay
                                                                                "beta1": beta1, "beta2": beta2,
                                                                                "timeDiscount": timeDiscount,
                                                                                "timeDivider": timeDivider,
                                                                                },
                                     trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                                     verbose=False)
server_ARQ_infinit_cwnd = EchoServer(
    serverId=311, ACKMode="SACK", verbose=False)

client_UDP = EchoClient(clientId=401, serverId=411,
                        protocolName="UDP", transportParam={
                            "alpha": alpha,
                            # beta1: emphasis on delivery, beta2: emphasis on delay
                            "beta1": beta1, "beta2": beta2,
                            "timeDiscount": timeDiscount,
                            "timeDivider": timeDivider,
                        },
                        trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                        verbose=False)
server_UDP = EchoServer(serverId=411, ACKMode=None, verbose=False)

client_TCP_Reno = EchoClient(clientId=501, serverId=511,
                             # IW=2 if SMSS>2190, IW=3 if SMSS>3, else IW=4
                             protocolName="tcp_newreno", transportParam={"timeout": 30, "IW": 4,
                                                                         "alpha": alpha,
                                                                         # beta1: emphasis on delivery, beta2: emphasis on delay
                                                                         "beta1": beta1, "beta2": beta2,
                                                                         "timeDiscount": timeDiscount,
                                                                         "timeDivider": timeDivider, },
                             trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": 1},
                             verbose=False)
server_TCP_Reno = EchoServer(serverId=511, ACKMode="LC", verbose=False)

# test_clients = [client_UDP]
# test_servers = [server_UDP]
test_clients = [client_ARQ_infinit_cwnd]
test_servers = [server_ARQ_infinit_cwnd]
# test_clients = [client_RL]
# test_servers = [server_RL]
# test_clients = [client_RL, client_UDP, client_ARQ, client_TCP_Reno]
# test_servers = [server_RL, server_UDP, server_ARQ, server_TCP_Reno]
# test_clients = [client_UDP, client_ARQ_finit, client_ARQ_infinit_cwnd, client_RL]
# test_servers = [server_UDP, server_ARQ_finit, server_ARQ_infinit_cwnd, server_RL]
# test_clients = [client_TCP_Reno]
# test_servers = [server_TCP_Reno]


def test_client(client, server):

    ignored_pkt, retrans_pkt, retransProb = 0, 0, 0

    serverPerfFilename = client.getProtocolName()+"_perf.pkl"

    # if client.getProtocolName().lower() not in {"mcp", "udp"}:
    #     #check whether can load the previous performance file directly

    #     if os.path.exists(serverPerfFilename):
    #         print("find file ", serverPerfFilename)
    #         clientSidePerf, distincPktsSent, clientPid = server.calcPerfBasedOnDataFile(
    #             serverPerfFilename,
    #             utilityCalcHandler=client.transportObj.instance.calcUtility,
    #             utilityCalcHandlerParams=utilityCalcHandlerParams
    #         )

    #         # hacking
    #         client.pid = clientPid
    #         client.transportObj.instance.distincPktsSent = distincPktsSent
    #         client.transportObj.instance.perfDict = clientSidePerf.copy()
    #         return

    # system time
    # A suggested bufferSize >= processRate * rtt
    """
    channel = SingleModeChannel(processRate=3, bufferSize=300, rtt=100, pktDropProb=0.1, verbose=False) # deprecated
    #"""
    """
    channel = ConstDelayChannel(serviceRate=3, delay=100,
                          bufferSize=300, pktDropProb=0.1, loglevel=logging.INFO)
    #"""
    # """
    channel = RandomDelayChannel(serviceRate=3, bufferSize=300, rng=RangeUniform(
        100, 150), pktDropProb=0.1, loglevel=logging.INFO)
    # """

    clientList = env_clients + [client]
    serverList = env_servers + [server]

    ACKPacketList = []
    packetList_enCh = []
    packetList_deCh = []

    # clear each client server
    for c, s in zip(clientList, serverList):
        c.transportObj.instance.time = -1
        c.pid = 0
        c.time = -1
        s.time = -1

    channel.initBuffer()

    while not channel.isFull():  # fill the channel with environment packets
        packetList_enCh = []
        for clientId in np.random.permutation(len(env_clients)):
            packetList_enCh += env_clients[clientId].ticking(ACKPacketList)
        channel.acceptPkts(packetList_enCh)

    channel.time = 0

    packetList_enCh = []
    for time in range(1, simulationPeriod+1):
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

        if time % 30 == 0:  # record performance for the past 30 slots
            server.recordPerfInThisTick(client.getPktGen(),
                                        utilityCalcHandler=client.transportObj.instance.calcUtility,
                                        )

        if time % (simulationPeriod//10) == 0:
            print("time ", time, " =================")
            print("RTT", client.getRTT())
            print("RTO", client.getRTO())
            client.clientSidePerf()
            server.printPerf(
                client.getPktGen(),
                client.getProtocolName())

            if client.getProtocolName().lower() in {"mcp"}:
                ignored_pkt = client.transportObj.instance.perfDict["ignorePkts"] - ignored_pkt
                retrans_pkt = client.transportObj.instance.perfDict["retransAttempts"] - retrans_pkt
                retransProb = retrans_pkt / (retrans_pkt + ignored_pkt)
                # debug
                client.transportObj.instance.perfDict["retranProb"] = retransProb

    server.storePerf(serverPerfFilename,
                     clientPid=client.pid,
                     distincPktsSent=client.getPktGen(),
                     clientSidePerf=client.transportObj.instance.clientSidePerf())


# test each pair of client and server
for client, server in zip(test_clients, test_servers):
    test_client(client, server)


"""
check contents, performance ....
"""
header = ["ptcl", "pkts gen", "pkts sent", "pkts dlvy", "dlvy perc", "avg dly",
          "sys util", "l25p dlvy", "l25p dlvy perc", 'l25p dly', "l25p util"]
table = []

deliveredPktsPerSlot = dict()
deliveredPktsPerSlot["protocols"] = []
for client, server in zip(test_clients, test_servers):  # ignore the first two
    deliveredPktsPerSlot["protocols"].append(client.getProtocolName())
    deliveredPktsPerSlot[client.getProtocolName()] = dict()

    server.printPerf(client.getPktGen(), client.getProtocolName())
    client.transportObj.instance.clientSidePerf()

    # store data
    deliveredPktsPerSlot[client.getProtocolName()]["serverPerf"] = [
        server.pktsPerTick, server.delayPerPkt, server.perfRecords]
    deliveredPktsPerSlot[client.getProtocolName(
    )]["clientPerf"] = client.transportObj.instance.clientSidePerf()

    # for display
    deliveredPkts, delvyRate, avgDelay = server.serverSidePerf(
        client.getPktGen())
    last25percTime = int(simulationPeriod*0.25)
    last25percPkts = sum(server.pktsPerTick[-last25percTime:])
    last25percDelveyRate = last25percPkts / (client.pktsPerTick*last25percTime)
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
        last25percUtil
    ])


deliveredPktsPerSlot["general"] = table
deliveredPktsPerSlot["header"] = header
print(tabulate(table, headers=header))

# store data
pklFilePath = os.path.join(
    "results", "alpha{alpha}".format(alpha=alpha), pklFilename)
with open(pklFilePath, 'wb') as handle:
    pkl.dump(deliveredPktsPerSlot, handle, protocol=pkl.HIGHEST_PROTOCOL)
print("save to ", pklFilename)

# plot MCP packet ignored time diagram
# plt.plot(client_RL.transportObj.instance.pktIgnoredCounter, label="MCP")
# plt.savefig("results/MCP_pktignore_{beta1}_{beta2}.png".format(beta1=beta1, beta2=beta2))

# np.set_printoptions(suppress=True)
# csvFileName="results/MCP_RL_perf_{beta1}_{beta2}.csv".format(beta1=beta1, beta2=beta2)
# np.savetxt(csvFileName, client_RL.transportObj.instance.RL_Brain.memory, delimiter=",", fmt='%f')
