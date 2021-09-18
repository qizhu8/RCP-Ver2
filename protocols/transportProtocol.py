from collections import deque

class RTTEst(object):
    """
    RTT Estimator follows RFC 6298
    """
    def __init__(self):
        self.SRTT = 1    # mean of RTT
        self.RTTVAR = 1  # variance of RTT
        self.RTO = 0

    def RTOUpdate(self, rtt):
        """
        Roughtly the same as RFC 6298, using auto-regression. But the true rtt estimation, or RTO 
        contains two more variables, RTTVAR (rtt variance) and SRTT (smoothed rtt).
        R' is the rtt for a packet.
        RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
        SRTT <- (1 - alpha) * SRTT + alpha * R'

        The values recommended by RFC are alpha=1/8 and beta=1/4.


        RTO <- SRTT + max (G, K*RTTVAR) where K =4 is a constant, 
        G is a clock granularity in seconds, the number of ticks per second.
        We temporarily simulate our network as a 1 tick per second, so G=1 here

        http://sgros.blogspot.com/2012/02/calculating-tcp-rto.html
        """
        self.RTTVAR = self.RTTVAR * 0.75 + abs(self.RTTVAR-rtt) * 0.25
        self.SRTT = self.SRTT * 0.875 + rtt * (0.125)
        self.RTO = self.SRTT + max(1, 4*self.RTTVAR)
    
    def getRTT(self) -> float:
        return self.SRTT
    
    def getRTO(self) -> float:
        return self.RTO


class BaseTransportLayerProtocol(object):
    """
    Base class for all transport layer protocols
    """
    requiredKeys = {}
    optionalKeys = {"max TxAttempts": -1, "timeout": -1, "maxPktTxDDL": -1}

    def parseParamByMode(self, params, requiredKeys, optionalKeys):
        # required keys
        for key in requiredKeys:
            assert key in params, key + \
                " is required for " + type(self).__name__
            setattr(self, key, params[key])

        # optinal keys
        for key in optionalKeys:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, optionalKeys[key])

    def __init__(self, suid: int, duid: int, params: dict = {}, verbose: bool = False) -> None:
        self.suid = suid
        self.duid = duid
        self.verbose = verbose

        self.parseParamByMode(params=params, requiredKeys=self.__class__.requiredKeys,
                              optionalKeys=self.__class__.optionalKeys)
        
        self.RTTEst = RTTEst() # rtt, rto estimator

        # the buffer that new packets enters
        self.txBuffer = deque(maxlen=None) # infinite queue

        self.window = None # window that store un-ACKed packets

        # performance 
        self.distincPktsSent = 0 # used as a feedback information for the server to compute delivery rate
        self.perfDict = {}

        self.time = 0
    
    def receive(self, pktList):
        """
        Accept packets from application layer. 
        """
        self.txBuffer.extend(pktList)
    
    def ticking(self, ACKPktList):
        """
        1. process feedbacks based on ACKPktList
        2. prepare packets to (re)transmit
        """
        raise NotImplementedError

    def timeElapse(self):
        self.time += 1