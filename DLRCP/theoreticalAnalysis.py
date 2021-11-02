import numpy as np
from scipy.stats import norm
import pandas as pd
from tabulate import tabulate


def calc_delvy_rate_expect(gamma, rxMax):
    # calculated
    return 1 - gamma**rxMax


def calc_delay_expect(gamma, rtt, rxMax):
    rto_div_rtt = 3  # rto = 3 * rtt
    # delay = rtt * (1 - (rto_div_rtt * ((rxMax-1)+1)* (gamma**rxMax) + (gamma - gamma ** rxMax)/(1-gamma)))
    delay = rtt * (1 + rto_div_rtt*(gamma - gamma**rxMax) /
                   (1-gamma) - (rto_div_rtt*rxMax-2)*(gamma**rxMax))
    return delay


def calc_qij_approx_uniform(beta, a, b, rx, timeDivider):
    """Using Gaussian to approximate the cdf may too coarse, so far so good."""
    rto_div_rtt = 3  # rto = 3 * rtt

    a = (rto_div_rtt*rx)*a/timeDivider
    b = (rto_div_rtt*rx)*b/timeDivider
    mean = (b+a)/2
    var = (b-a)**2 / 12
    segments = 1000
    dx = (b-a) / segments
    x = np.linspace(a, b, segments)
    beta_exp_x = np.power(beta, x)
    if rx == 1:
        pdf = 1 / (b-a)  # uniform distribution
    else:  # gaussian approximation
        pdf = norm.pdf(x, loc=mean, scale=np.sqrt(var))
    qij = sum(beta_exp_x * pdf * dx)
    return qij


def calc_qij_approx_norm(beta, mean, var, rx, timeDivider):
    """Channel delay is a Gaussian"""
    rto_div_rtt = 3  # rto = 3 * rtt

    mean *= rx
    var *= rx

    segments = 1000
    sigma = np.sqrt(var)
    # sample from [-sigma_multi*sigma, +sigma_multi*sigma]+mean
    sigma_multi = 3
    dx = 2*sigma_multi*sigma / segments
    x = np.linspace(mean-sigma_multi*sigma, mean+sigma_multi*sigma, segments)
    beta_exp_x = np.power(beta, x)
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    qij = sum(beta_exp_x * pdf * dx)
    return qij


def calc_utility(delay, delvy, alpha, beta, timeDivider):
    r = (beta**(delay/timeDivider)) * (delvy**alpha)
    return r


def calc_V_theo_uniform(gamma, timeDivider, alpha, beta, channelDelay_a, channelDelay_b, smax):
    channelDelay = (channelDelay_a + channelDelay_b) / 2
    utility = []
    for s in range(1, smax):
        delay_s = calc_delay_expect(gamma, channelDelay, s)
        deliveryRate_s = calc_delvy_rate_expect(gamma, s)
        utility_s = calc_utility(
            delay_s, deliveryRate_s, alpha, beta, timeDivider)
        utility.append(utility_s)
    s = np.argmax(utility)+1  # our s is 1-index
    AB = np.zeros((s, smax))  # the [A, B] matrix in mathematical model
    qij_list = np.zeros(smax+1)
    for drx in range(1, smax+1):
        qij_list[drx] = calc_qij_approx_uniform(beta, a, b, drx, timeDivider)
    qij_list /= sum(qij_list)

    for col in range(1, smax+1):
        for row in range(0, s):
            if row+col < smax:
                AB[row, row+col] = qij_list[col]

    A = AB[:, :s]
    B = AB[:, s:]
    h = np.asarray([[utility_s]*(smax - s)]).T
    g = np.asarray([[calc_utility(channelDelay*rx, 1, alpha,
                   beta, timeDivider) for rx in range(1, s+1)]]).T
    v_threory = np.linalg.pinv(
        np.eye(s)-A).dot(B.dot(h) + (1-gamma)*g)
    v = np.concatenate([v_threory[:, 0], h[:, 0]])
    return v, s


def calc_V_theo_norm(gamma, timeDivider, alpha, beta, mean, var, smax):
    channelDelay = mean
    utility = []
    for s in range(1, smax):
        delay_s = calc_delay_expect(gamma, channelDelay, s)
        deliveryRate_s = calc_delvy_rate_expect(gamma, s)
        utility_s = calc_utility(
            delay_s, deliveryRate_s, alpha, beta, timeDivider)
        utility.append(utility_s)
    s = np.argmax(utility)+1  # our s is 1-index
    AB = np.zeros((s, smax))  # the [A, B] matrix in mathematical model
    qij_list = np.zeros(smax+1)
    for drx in range(1, smax+1):
        qij_list[drx] = calc_qij_approx_norm(beta, mean, var, drx, timeDivider)
    qij_list /= sum(qij_list)

    for col in range(1, smax+1):
        for row in range(0, s):
            if row+col < smax:
                AB[row, row+col] = qij_list[col]

    A = AB[:, :s]
    B = AB[:, s:]
    h = np.asarray([[utility_s]*(smax - s)]).T
    g = np.asarray([[calc_utility(channelDelay*rx, 1, alpha,
                   beta, timeDivider) for rx in range(1, s+1)]]).T
    v_threory = np.linalg.pinv(
        np.eye(s)-A).dot(B.dot(h) + (1-gamma)*g)
    v = np.concatenate([v_threory[:, 0], h[:, 0]])
    return v, s


if __name__ == "__main__":
    print("="*30)
    channelPktLossRate = 0.3
    timeDivider = 100

    alpha = 2
    beta = 0.6
    a, b = 100, 150
    channelDelay = (a+b)/2
    smax = 15

    # find the optimal s
    utility = []
    for s in range(1, smax):
        delay_s = calc_delay_expect(channelPktLossRate, channelDelay, s)
        deliveryRate_s = calc_delvy_rate_expect(channelPktLossRate, s)
        utility_s = calc_utility(
            delay_s, deliveryRate_s, alpha, beta, timeDivider)
        utility.append(utility_s)
    s = np.argmax(utility)+1  # our s is 1-index
    print(utility)
    print("s* is", s)
    AB = np.zeros((s, smax))

    delay_s = calc_delay_expect(channelPktLossRate, channelDelay, s)
    deliveryRate_s = calc_delvy_rate_expect(channelPktLossRate, s)

    utility_s = calc_utility(delay_s, deliveryRate_s, alpha, beta, timeDivider)

    print("expected delay", delay_s)
    print("expected delvy", deliveryRate_s)
    print("expected util", utility_s)

    qij_list = np.zeros(smax+1)
    for drx in range(1, smax+1):
        qij_list[drx] = calc_qij_approx_uniform(beta, a, b, drx, timeDivider)

    print("sum(qij_list)=", sum(qij_list))
    qij_list /= sum(qij_list)
    print(qij_list)

    for col in range(1, smax+1):
        for row in range(0, s):
            if row+col < smax:
                AB[row, row+col] = qij_list[col]

    A = AB[:, :s]
    B = AB[:, s:]

    h = np.asarray([[utility_s]*(smax - s)]).T
    g = np.asarray([[calc_utility(channelDelay*rx, 1, alpha,
                   beta, timeDivider) for rx in range(1, s+1)]]).T
    v_threory = np.linalg.pinv(
        np.eye(s)-A).dot(B.dot(h) + (1-channelPktLossRate)*g)

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

    """Summary of v* for all beta"""
    alpha = 2
    betaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    channelDelay_a, channelDelay_b = 100, 150
    smax = 15
    channelPktLossRate = 0.3
    timeDivider = 100
    vList = np.zeros((smax, len(betaList)))
    sList = np.zeros((1, len(betaList)))
    for beta_id, beta in enumerate(betaList):
        v, s = calc_V_theo_uniform(
            channelPktLossRate, timeDivider, alpha, beta, channelDelay_a, channelDelay_b, smax)
        print(v)
        vList[:, beta_id] = v
        sList[0, beta_id] = s
        # df[beta_id] =

    header = ",".join(["beta="+str(beta) for beta in betaList])
    np.savetxt("theore_v.txt", vList, fmt="%f",
               delimiter=",", header=header, comments="")
    np.savetxt("theore_s.txt", sList, fmt="%d",
               delimiter=",", header=header, comments="")
