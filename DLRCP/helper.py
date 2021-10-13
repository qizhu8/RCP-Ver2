import argparse
import logging

from DLRCP.applications import EchoClient, EchoServer
from DLRCP.channel import ConstDelayChannel, RandomDelayChannel
from DLRCP.common import RangeUniform


def get_opts():
    parser = argparse.ArgumentParser(description='RCP-Ver 2.0')

    # Paths
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='data folder')

    # Channel Type
    parser.add_argument('--channelType', type=str, default='RandomDelayChannel',
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
    parser.add_argument('--fillChannel', default=True, action='store_true',
                        help="whether to fill the channel with environment packets")

    # environment setting
    parser.add_argument('--bgClientNum', type=int, default=4,
                        help='# of UDP clients to simulate the background traffic')
    parser.add_argument('--bgClientPktRate', type=int, default=1,
                        help='# pkt to send by the background UDP client per tick')

    # utility setting
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='exponent index of delivery rate')
    parser.add_argument('--timeDiscount', type=float, default=0.9,
                        help='reward discount over time')
    parser.add_argument('--timeDivider', type=float, default=100,
                        help='1 time unit = timeDivider ticks')

    # Test protocol setting
    parser.add_argument('--pktRate', type=int, default=1,
                        help='number of pkts generated per tick for the client to be tested')

    parser.add_argument('--testPeriod', type=int, default=20000,
                        help='simulation period')

    # UDP
    parser.add_argument('--addUDP', default=True, action='store_true',
                        help="whether to add UDP to test")

    # ARQ finite window
    parser.add_argument('--addARQFinite', default=True, action='store_true',
                        help="whether to add ARQ with finite window to test")
    parser.add_argument('--ARQWin', type=int, default=200,
                        help='maximum window size of ARQ Finite')

    # ARQ infinite window
    parser.add_argument('--addARQInfinite', default=True, action='store_true',
                        help="whether to add ARQ with infinite window to test")

    # TCP NewReno
    parser.add_argument('--addNewReno', default=True, action='store_true',
                        help="whether to add TCP New Reno to test")

    # RCP-QLearning
    parser.add_argument('--addRCPQLearning', default=True, action='store_true',
                        help="whether to add RCP-QLearning to test")
    parser.add_argument('--RCPQLearningGamma', type=float, default=0.9,
                        help='reward decay coefficient')
    parser.add_argument('--RCPQLearningEpsilon', type=float, default=0.7,
                        help='DQN greedy policy epsilon')
    parser.add_argument('--RCPQLearningEpsilonDecay', type=float, default=0.9,
                        help='DQN greedy policy epsilon decay')

    # RCP-DQN
    parser.add_argument('--addRCPDQN', default=True, action='store_true',
                        help="whether to add RCP-QLearning to test")
    parser.add_argument('--RCPDQNGamma', type=float, default=0.9,
                        help='reward decay coefficient')
    parser.add_argument('--RCPDQNEpsilon', type=float, default=0.7,
                        help='DQN greedy policy epsilon')
    parser.add_argument('--RCPDQNEpsilonDecay', type=float, default=0.9,
                        help='DQN greedy policy epsilon decay')

    # Additional options
    parser.add_argument('--clean-run', dest='clean_run',
                        default=False, action='store_true')

    ##
    opts = parser.parse_args()
    return opts


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
            loglevel=logging.INFO)
    if opts.channelType == "ConstDelayChannel":
        channel = ConstDelayChannel(
            serviceRate=opts.serviceRate,
            bufferSize=opts.bufferSize,
            delay=opts.channelDelay[0],
            pktDropProb=opts.pktDropProb,
            loglevel=logging.INFO)

    return channel


def genBgClientsAndServers(opts):
    env_clients = []
    env_servers = []
    for clientId in range(1, opts.bgClientNum+1):

        client = EchoClient(clientId=clientId, serverId=1000+clientId,
                            protocolName="UDP",
                            transportParam={},
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
                "RLEngine": "Q_Learning",
                "gamma": opts.RCPQLearningGamma,
                "epsilon": opts.RCPQLearningEpsilon,
                "epsilon_decay": opts.RCPQLearningEpsilonDecay,
                "learnRetransmissionOnly": False},  # whether only learn the data related to retransmission
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
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
                "alpha": opts.alpha,
                "timeDiscount": opts.timeDiscount,
                "timeDivider": opts.timeDivider,
                "RLEngine": "DQN",
                "gamma": opts.RCPDQNGamma,
                "epsilon": opts.RCPDQNEpsilon,
                "epsilon_decay": opts.RCPDQNEpsilonDecay,
                "learnRetransmissionOnly": False},  # whether only learn the data related to retransmission
            trafficMode="periodic", trafficParam={"period": 1, "pktsPerPeriod": opts.pktRate},
            verbose=False)
        server_RL_DQN = EchoServer(serverId=611, ACKMode="SACK", verbose=False)
        return [client_RL_DQN], [server_RL_DQN]
    return [], []
