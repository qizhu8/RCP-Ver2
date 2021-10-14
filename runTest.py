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
Prepare data storage folder
"""
helper.prepareDataStorageFolder(opts)

"""
clean previous files
"""
helper.cleanPrevDataFiles(opts)

helper.saveOpts(opts)


"""
Generate Background Traffic
"""
env_clients, env_servers = helper.genBgClientsAndServers(opts)

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

"""
Run the simulation for each test client and test server pair
"""

for test_client, test_server in zip(test_clients, test_servers):
    helper.test_protocol(
        opts, 
        channel=channel, 
        test_client=test_client, 
        test_server=test_server, 
        env_clients=env_clients, 
        env_servers=env_servers, 
        loadFromHistoryIfPossible=not opts.clean_run)


"""
Print the summary of the final result
"""
perfTable, header = helper.printTestProtocolPerf(opts, test_clients, test_servers)

"""
Other data storage process
"""
if client_RL_Q:
    tgtFile = os.path.join(opts.data_dir, opts.testDesc, "QTable.txt")
    client_RL_Q[0].transportObj.instance.RL_Brain.saveModel(tgtFile)