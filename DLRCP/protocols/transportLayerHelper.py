import logging

from .transportProtocol import BaseTransportLayerProtocol
from .RCP import RCP
from .TCP import TCP_NewReno, Window_ARQ, TCP_Vegas, TCP_CTCP
from .UDP import UDP


class TransportLayerHelper(object):
    """
    TransportLayerHelper in charges of binding to a specific protocol implementation. 

    """

    def __init__(self, suid, duid, protocolName, params, verbose=False, create_file: bool=False):
        supportProtocols = {
            "udp": UDP,
            "window arq": Window_ARQ,
            "rcp": RCP,
            "tcp_newreno": TCP_NewReno,
            "tcp_vegas": TCP_Vegas,
            "tcp_ctcp": TCP_CTCP
        }
        self.suid = suid
        self.duid = duid

        if verbose:
            loglevel = logging.INFO
        else:
            loglevel = logging.ERROR

        self.protocolName = protocolName.lower()
        assert self.protocolName in supportProtocols, protocolName + \
            " is not supported. Choose from " + \
            list(supportProtocols.keys).__str__()

        self.instance = supportProtocols[self.protocolName](
            suid=suid, duid=duid, params=params, loglevel=loglevel, create_file=create_file)

        # some protocol will add a more detailed suffix, e.g. ARQ_finit_window
        self.protocolName = self.instance.protocolName

    def receiveFromApplication(self, pktList):
        self.instance.acceptNewPkts(pktList)

    def sendPkts(self, ACKPktList=[]):
        return self.instance.ticking(ACKPktList=ACKPktList)

    def getDistinctPktSent(self):
        return self.instance.perfDict["distinctPktsSent"]
