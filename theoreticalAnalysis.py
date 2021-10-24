import numpy as np
from scipy.stats import norm
import pandas as pd
from tabulate import tabulate


def calcDelvyRate(gamma, rxMax):
    # calculated 
    return 1 - gamma**rxMax


def calcDelay(gamma, rtt, rxMax):
    rto_div_rtt = 3 # rto = 3 * rtt
    delay = rtt * (1 - (rto_div_rtt * ((rxMax-1)+1)* (gamma**rxMax) + (gamma - gamma ** rxMax)/(1-gamma)))
    return delay


def calcQij_1_approx(beta, a, b, rx, timeDivider):
    """Using Gaussian to approximate the cdf may too coarse"""
    a = rx*a/timeDivider
    b = rx*b/timeDivider
    mean = (b+a)/2
    var = (b-a)**2 / 12
    qij = 0
    segments = 1000
    dx = (b-a) / segments
    for x in np.linspace(a, b, segments):
        qij += (beta**x * norm.pdf(x, loc=mean, scale=np.sqrt(var)))*dx
    
    return qij

def calcUtility(delay, delvy, alpha, beta):
    r = (beta**(delay/timeDivider)) * (delvy**alpha)
    return r



print("="*30)
channelPktLossRate = 0.3
timeDivider = 100

alpha = 2
beta = 0.8
a, b = 100, 150
channelDelay = (a+b)/2
smax = 15


# find the optimal s
utility = []
for s in range(1, smax):
    delay_s = calcDelay(channelPktLossRate, channelDelay, s)
    deliveryRate_s = calcDelvyRate(channelPktLossRate, s)
    utility_s = calcUtility(delay_s, deliveryRate_s, alpha, beta)
    utility.append(utility_s)
s = np.argmax(utility)+1 # our s is 1-index
print(utility)
print("s* is", s)
AB = np.zeros((s, smax))

delay_s = calcDelay(channelPktLossRate, channelDelay, s)
deliveryRate_s = calcDelvyRate(channelPktLossRate, s)

utility_s = calcUtility(delay_s, deliveryRate_s, alpha, beta)

print("expected delay", delay_s)
print("expected delvy", deliveryRate_s)
print("expected util", utility_s)

qij_list = np.zeros(smax+1)
for drx in range(1, smax+1):
    qij_list[drx] = calcQij_1_approx(beta,a,b,drx, timeDivider)

qij_list /= sum(qij_list)

for col in range(1, smax+1):
    for row in range(0, s):
        if row+col < smax:
            AB[row, row+col] = qij_list[col]

# for drx in range(1, smax+1):
#     if drx == 0:
#         qij_d = 1
#     else:
#         qij_d = calcQij_1_approx(beta,a,b,drx, timeDivider)
#     for i in range(0, s):
#         if i+drx < smax:
#             AB[i, i+drx] = qij_d

A = AB[:, :s]
B = AB[:, s:]
print("AB")
print(AB)
print("A")
print(A)
print("B")
print(B)

h = np.asarray([[utility_s]*(smax - s)]).T
g = np.asarray([[calcUtility(channelDelay*rx, 1, alpha, beta) for rx in range(1, s+1)]]).T
v_threory = np.linalg.pinv(np.eye(s)-A).dot(B.dot(h) + (1-channelPktLossRate)*g)

print("h")
print(h)

print("g")
print(g)
print((1-channelPktLossRate)*g)

print("V*")
print(v_threory)

diff = v_threory - ((1-channelPktLossRate)*g + A.dot(v_threory) + B.dot(h))
print("diff")
print(diff)

v = np.concatenate([v_threory[:, 0], h[:, 0]])
print("concatenated v")
print(v)

print("utility")
print(utility)
print("s* is", s)
print(calcUtility(125.8, 0.5, alpha, beta))