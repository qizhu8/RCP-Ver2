import os
import sys
import logging
import numpy as npß
import pickle as pkl
import multiprocessing
from tabulate import tabulate
import matplotlib.pyplot as plt

from DLRCP.applications import EchoClient, EchoServer
from DLRCP.channel import ConstDelayChannel, RandomDelayChannel
from DLRCP.common import RangeUniform
import DLRCP.helper as helper

def main():
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
    client_Vegas, server_Vegas = helper.genTCPVegas(opts)
    client_CTCP, server_CTCP = helper.genTCPCompond(opts)
    client_RL_Q, server_RL_Q = helper.genRCPQLearning(opts)
    client_RL_DQN, server_RL_DQN = helper.genRCPDQN(opts)
    client_RL_RTQ, server_RL_RTQ = helper.genRCPRTQ(opts)

    test_clients = client_UDP + client_ARQ_finit + \
        client_ARQ_infinit + client_NewReno + client_Vegas + client_CTCP + client_RL_Q + client_RL_DQN + client_RL_RTQ
    test_servers = server_UDP + server_ARQ_finit + \
        server_ARQ_infinit + server_NewReno + server_Vegas + server_CTCP + server_RL_Q + server_RL_DQN + server_RL_RTQ

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
            loadFromHistoryIfPossible=opts.load_status_if_possible)

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

    if client_RL_RTQ:
        tgtFile = os.path.join(opts.data_dir, opts.testDesc, "RTQPerf.txt")
        client_RL_RTQ[0].transportObj.instance.RL_Brain.saveModel(tgtFile)

if __name__ == "__main__":
    main()