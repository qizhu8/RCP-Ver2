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
import DLRCP.helper as helper

opts = helper.get_opts()

"""
Generate Background Traffic
"""
bgClients, bgServers = helper.genBgClientsAndServers(opts)

"""
Generate channel
"""
channel = helper.genChannel(opts)

"""
Generate test clients
"""
client_UDP, server_UDP = helper.genTestUDP(opts)
client_ARQ_finit, server_ARQ_finit = helper.genTestARQFiniteWindow(opts)
client_ARQ_infinit, server_ARQ_infinit = helper.genTestARQinFiniteWindow(opts)
client_NewReno, server_NewReno = helper.genTCPNewReno(opts)
client_RL_Q, server_RL_Q = helper.genRCPQLearning(opts)
client_RL_DQN, server_RL_DQN = helper.genRCPDQN(opts)

test_clients = client_UDP + client_ARQ_finit + \
    client_ARQ_infinit + client_NewReno + client_RL_Q + client_RL_DQN
test_servers = server_UDP + server_ARQ_finit + \
    server_ARQ_infinit + server_NewReno + server_RL_Q + server_RL_DQN
