import numpy as np
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.normpath(os.path.join(
    os.path.abspath(__file__), '..', '..', '..')))

from RL_Brain import DQN_Brain
from Q_Brain import Q_Brain
from common import Packet
from common import BaseRNG, RangeUniform

from protocols.transportProtocol import BaseTransportLayerProtocol

class MCP(BaseTransportLayerProtocol):
    requiredKeys = {}
    optionalKeys = {
        "maxTxAttempts":-1, "timeout":-1, "maxPktTxDDL":-1,
        "alpha":2, # shape of utility function
        "beta1":0.9, "beta2":0.1, # beta1: emphasis on delivery, beta2: emphasis on delay
        "gamma":0.9,
        "learnRetransmissionOnly": False
    }

    def __init__(self, suid: int, duid: int, params: dict = ..., loglevel=BaseTransportLayerProtocol.LOGLEVEL) -> None:
        super().__init__(suid, duid, params=params, loglevel=loglevel)



    def _handleACK(self, ACKPktList: list(Packet)):
        """handle ACK Packets"""
        ACKPidList = []
        for pkt in ACKPktList:
            # only process designated packets
            if pkt.duid == self.suid and pkt.packetType == Packet.ACK:
                self.perfDict["receivedACK"] += 1

                if not self.window.isPktInWindow(pkt.pid):
                    continue

                ACKPidList.append(pkt.pid)

                rtt = self.time-self.buffer[pkt.pid].txTime
                self.RTTEst.Update(rtt, self.perfDict)

        raise Exception("The following has not been implemented yet...")
        self._handleACK_SACK(SACKPidList=ACKPidList) 

    


if __name__ == "__main__":
    suid = 0
    duid=1
    params = {"maxTxAttempts":-1, "timeout":30, "maxPktTxDDL":-1,
    "alpha":2,
    "beta1":0.8, "beta2":0.2, # beta1: emphasis on delivery, beta2: emphasis on delay
    "gamma":0.9,
    "learnRetransmissionOnly": True}, # whether only learn the data related to retransmission
    MCP(suid=suid, duid=duid, params=params)