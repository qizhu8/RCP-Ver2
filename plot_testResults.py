"""
Plot the utility, last25percent utility over timeDiscount for each protocol
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import DLRCP.theoreticalAnalysis as theoTool

attributeNameDict = {
        "protocol": "ptcl",
        "generated packets": "pkts gen",
        "sent packets": "pkts sent",
        "packets delivered": "pkts dlvy",
        "delivery percentage": "dlvy perc",
        "average delay": "avg dly",
        "system utility": "sys util",
        "packets delivered (last 25%)": "l25p dlvy",
        "delivery percentage (last 25%)": "l25p dlvy perc",
        "average delay (last 25%)": "l25p dly",
        "system utility (last 25%)": "l25p util",
        "loss": "loss"
    }

def plot_one_attribute(testPerfDicts, attributeName, configAttributeName, attributeNameDict, resultFolder):
    columnName = attributeNameDict[attributeName]
    orderedKey = list(testPerfDicts.keys())
    orderedKey = sorted(orderedKey, key=lambda x: testPerfDicts[x][0])
    protocolPerf = {}
    for key in orderedKey:
        pd = testPerfDicts[key][1]
        protocols = pd[attributeNameDict["protocol"]]
        columnOfInterest = pd[attributeNameDict[attributeName]]
        for protocolId, protocol in enumerate(protocols):
            if protocol not in protocolPerf:
                protocolPerf[protocol] = [columnOfInterest[protocolId]]
            else:
                protocolPerf[protocol].append(columnOfInterest[protocolId])
    
    # for plotting
    plt.clf()
    imgDir = os.path.join(resultFolder, "summary")
    os.makedirs(imgDir, exist_ok=True)
    imgPath = os.path.join(imgDir, attributeName + "_" + configAttributeName + ".png")

    timeDiscountList = [testPerfDicts[x][0] for x in orderedKey]
    for protocolId, protocol in enumerate(protocols):
        plt.plot(timeDiscountList, protocolPerf[protocol], '-o', label=protocol)
    
    plt.xlabel(configAttributeName)
    plt.ylabel(attributeName)
    plt.legend()
    plt.savefig(imgPath)
    # plt.show()
    print("Generating", imgPath)
    return protocols, protocolPerf


def process_one_attribute(resultFolder, subFolderPrefix, configAttributeName, attributeName):
    # map sementic name to column name in csv file

    # scan folder and load the "perfBrief.csv" in each subfolder
    subFolders = os.listdir(resultFolder)
    subFolders = filter(lambda s: s.startswith(subFolderPrefix), subFolders)
    subFolders = map(lambda s: os.path.join(resultFolder, s), subFolders)
    subFolders = filter(os.path.isdir, subFolders)  # filter only directories

    # get the "perfBrief.csv" from each subfolder
    testPerfDicts = {}
    for subfolder in subFolders:
        csvFileName = os.path.join(subfolder, 'perfBrief.csv')
        jsonFileName = os.path.join(subfolder, 'test_config.json')

        with open(jsonFileName, "r") as fp:
            configDict = json.load(fp)
        # get the attribute that changes over different experiments from the save json file
        firstDigit = int(configDict[configAttributeName])
        secondDigit = int(configDict[configAttributeName]*10) % 10
        key = str(firstDigit)+str(secondDigit)

        testPerfDicts[key] = [configDict[configAttributeName], pd.read_csv(csvFileName, delimiter=',')]
    
    plot_one_attribute(testPerfDicts, attributeName, configAttributeName, attributeNameDict, resultFolder)

def gather_Q_table(resultFolder, subFolderPrefix, configAttributeName):
    # scan folder and load the "QTable.txt" in each subfolder
    subFolders = os.listdir(resultFolder)
    subFolders = filter(lambda s: s.startswith(subFolderPrefix), subFolders)
    subFolders = map(lambda s: os.path.join(resultFolder, s), subFolders)
    subFolders = filter(os.path.isdir, subFolders)  # filter only directories

    QDict = {}
    RTQDict = {}
    for subfolder in subFolders:
        jsonFileName = os.path.join(subfolder, 'test_config.json')
        RTQEstFilename = os.path.join(subfolder, "RTQPerf.txt")
        QTableFilename = os.path.join(subfolder, "Qtable.txt")

        with open(jsonFileName, "r") as fp:
            configDict = json.load(fp)
        firstDigit = int(configDict[configAttributeName])
        secondDigit = int(configDict[configAttributeName]*10) % 10
        key = str(firstDigit)+str(secondDigit)

        timeDivider = float(configDict['timeDivider'])
        alpha = float(configDict['alpha'])
        beta = float(configDict['beta'])
        
        #RTQ
        maximumSimulationSmax = 15
        RTQPerf = np.loadtxt(RTQEstFilename, delimiter=",", usecols=[1])
        RTQ_pktLossRate, RTQ_delay, RTQ_delay_var, RTQ_smax =  RTQPerf
        RTQ_v, _ = theoTool.calc_V_theo_norm(RTQ_pktLossRate, timeDivider, alpha, beta, RTQ_delay, RTQ_delay_var, maximumSimulationSmax)
        RTQDict[key] = [RTQ_smax, RTQ_v]


        # Q-learning
        Q = np.loadtxt(QTableFilename, delimiter=",")
        v = np.max(Q, axis=1)
        for s in range(Q.shape[0]):
            if Q[s, 0] > Q[s, 1]:
                break
        QDict[key] = [configDict[configAttributeName], Q, v, s]
    
    orderedKey = list(QDict.keys())
    orderedKey = sorted(orderedKey, key=lambda x: QDict[x][0])
    # save RTQ
    RTQ_vList = np.zeros((maximumSimulationSmax, len(orderedKey)))
    RTQ_sList = np.zeros((1, len(orderedKey)))
    for key_id, key in enumerate(orderedKey):
        s, v = RTQDict[key]
        RTQ_vList[:len(v), key_id] = v
        RTQ_sList[0, key_id] = s
    
    # save rst
    saveDir = os.path.join(resultFolder, "summary")
    header = ",".join([configAttributeName+"="+str(QDict[key][0]) for key in orderedKey])
    np.savetxt(os.path.join(saveDir, "RTQ_v.txt"), RTQ_vList, fmt="%f", delimiter=",", header=header, comments="")
    np.savetxt(os.path.join(saveDir, "RTQ_s.txt"), RTQ_sList, fmt="%d", delimiter=",", header=header, comments="")

    # save Q-Learning
    smax = max([len(QDict[key][2]) for key in orderedKey])
    vList = np.zeros((smax, len(orderedKey)))
    sList = np.zeros((1, len(orderedKey)))
    for key_id, key in enumerate(orderedKey):
        _, Q, v, s = QDict[key]
        vList[:len(v), key_id] = v
        sList[0, key_id] = s
    
    # save rst
    saveDir = os.path.join(resultFolder, "summary")
    header = ",".join([configAttributeName+"="+str(QDict[key][0]) for key in orderedKey])
    np.savetxt(os.path.join(saveDir, "Q_learning_v.txt"), vList, fmt="%f", delimiter=",", header=header, comments="")
    np.savetxt(os.path.join(saveDir, "Q_learning_s.txt"), sList, fmt="%d", delimiter=",", header=header, comments="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RCP-Ver 2.0 - plot for time discount')
    # parser.add_argument('--resultFolder', type=str, default= 'Results/case_study_TimeDiscount_alpha_2_0', required=False,
    #                     help='parent folder that stores multiple test results')
    # parser.add_argument('--subFolderPrefix', type=str, default= 'TimeDiscount', required=False,
    #                     help='prefix of the subfolder that stores each test result')
    # parser.add_argument('--configAttributeName', type=str, default= 'beta',
    #                     help='the attribute name that changes among the experimetns')

    parser.add_argument('--resultFolder', type=str, default= '', required=True,
                        help='parent folder that stores multiple test results')
    parser.add_argument('--subFolderPrefix', type=str, default= '', required=True,
                        help='prefix of the subfolder that stores each test result')
    parser.add_argument('--configAttributeName', type=str, default= 'beta',
                        help='the attribute name that changes among the experimetns')

    opts = parser.parse_args()

    attributeNameList=attributeNameDict.keys()
    for attributeName in attributeNameList:
        process_one_attribute(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName, attributeName)
    
    gather_Q_table(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName)
