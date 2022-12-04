"""
Plot the utility, last25percent utility over timeDiscount for each protocol
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import matplotlib.pyplot as plt

import DLRCP.theoreticalAnalysis as theoTool

imgExtensions = [".png", ".pdf"]

class DummyClass:
    def __init__(self):
        self.attr = ""

# we have changed the protocol name multiple times... tired of that
protocolName = {
    'UDP': 'UDP',
    'ARQ_inf_wind': 'UDP with ARQ',
    'ARQ_finit_wind': 'ARQ finit window',
    'TCP_NewReno': 'TCP-NewReno',
    'TCP_Vegas': 'TCP-Vegas',
    'TCP_CTCP': "Compound TCP",
    'RCPQ_Learning': 'QRCP', # the Q-Learning based policy
    'RCPRTQ': 'CERCP',       # the control limit policy
    'RCPDQN': 'RCP-DQN',     # the DQN implementation
}
protocolColor = {
    'UDP': 'blue',
    'ARQ_inf_wind': 'orange',
    'ARQ_finit_wind': 'yellow',
    'TCP_NewReno': 'black',
    'TCP_Vegas': 'darkgray',
    'TCP_CTCP': 'wheat',
    'RCPQ_Learning': 'green',
    'RCPRTQ': 'red',
    'RCPDQN': 'pink'
}

labelDisplay = {
    "beta": r"$\beta$",
    "pktDropProb": r"$\gamma$"
}

# protocols that will plot both overall performance and the last 25% time performance
protocolPlotLast25 = { 
    'RCPQ_Learning': 'RTQ',
}

# attributes that can be read from the performance briefing (perfBrief.csv)
attributeNameDict_csv = {
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
    "loss": "loss",
}

# attributes that can be read from pkl files
attributeNameDict_pkl = {
    "retransProb": "Retransmission Probability"
}

# attributes that will be plot over simulation time per experiment
attributeNameDict_time = {
    "retransProb": "Retransmission Probability"
}


def plot_one_attribute_csv(testPerfDicts, attributeName, configAttributeName, attributeNameDict, resultFolder):
    # plot one attribute that is read from the performance briefing
    columnName = attributeNameDict[attributeName]
    l25columnName = None
    if attributeName + " (last 25%)" in attributeNameDict:
        l25columnName = attributeNameDict[attributeName + " (last 25%)"]
    orderedKey = list(testPerfDicts.keys())
    orderedKey = sorted(orderedKey, key=lambda x: testPerfDicts[x][0])
    protocolPerf = {}
    for key in orderedKey:
        pd = testPerfDicts[key][1]
        protocols = pd[attributeNameDict["protocol"]]
        columnOfInterest = pd[columnName]
        l25columnOfInterest = pd[l25columnName] if l25columnName is not None else None
        for protocolId, protocol in enumerate(protocols):
            if protocol not in protocolPerf:
                protocolPerf[protocol] = [columnOfInterest[protocolId]]
            else:
                protocolPerf[protocol].append(columnOfInterest[protocolId])

            if protocol in protocolPlotLast25 and l25columnOfInterest is not None:
                if protocol+"-final" not in protocolPerf:
                    protocolPerf[protocol+"-final"] = [l25columnOfInterest[protocolId]]
                else:
                    protocolPerf[protocol+"-final"].append(l25columnOfInterest[protocolId])
    
    # for plotting
    plt.clf()
    imgDir = os.path.join(resultFolder, "summary")
    os.makedirs(imgDir, exist_ok=True)

    timeDiscountList = [testPerfDicts[x][0] for x in orderedKey]
    for protocolId, protocol in enumerate(protocols):
        plt.plot(timeDiscountList, protocolPerf[protocol], '-o', color=protocolColor[protocol], label=protocolName[protocol],)
        if protocol+"-final" in protocolPerf:
            plt.plot(timeDiscountList, protocolPerf[protocol+"-final"], '-o', color=protocolColor[protocol], label=protocolName[protocol]+"-final", alpha=0.7)
    
    """We also want to plot the last 25% time performance of RTQ"""
    
    plt.xlabel(labelDisplay[configAttributeName])
    # plt.xlabel("$\eta$")
    plt.ylabel(attributeName)
    plt.legend()
    for imgExtension in imgExtensions:
        imgPath = os.path.join(imgDir, attributeName + "_" + configAttributeName + imgExtension)
        plt.savefig(imgPath)
    # plt.show()
    print("Generating", imgPath)
    return protocols, protocolPerf


def process_one_attribute_csv(resultFolder, subFolderPrefix, configAttributeName, attributeName):
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
        # firstDigit = int(configDict[configAttributeName])
        # secondDigit = int(configDict[configAttributeName]*10) % 10
        # key = str(firstDigit)+str(secondDigit)
        key = configDict[configAttributeName]

        testPerfDicts[key] = [key, pd.read_csv(csvFileName, delimiter=',')]
    
    plot_one_attribute_csv(testPerfDicts, attributeName, configAttributeName, attributeNameDict_csv, resultFolder)


def interpretPklData(data):
    perfRecord = np.asarray(data["perfRecords"])
    svrPerfDf = pd.DataFrame({
        "time": perfRecord[:, 0],
        "delivered pkt since before": perfRecord[:, 1],
        "pkt delivery rate": perfRecord[:, 2],
        "avg delay": perfRecord[:, 3],
        "utility": perfRecord[:, 4]
        })
    
    svrPktID, svrPktDelvyRate, svrAvgDelay = data["serverSidePerf"]

    cltPerfDf = pd.DataFrame({
        "retransSoFar": data["clientSidePerf"]["retransSoFar"], 
        "retransProb": data["clientSidePerf"]["retransProbSoFar"], 
        "ignorePktsSoFar": data["clientSidePerf"]["ignorePktsSoFar"], 
        })
    
    return {
        "serverPerfDf": svrPerfDf,
        "clientPerfDf": cltPerfDf,
        "perfRecord": perfRecord
    }

def process_one_attribute_pkl(resultFolder, subFolderPrefix, configAttributeName, attributeName):
    # scan folder and load the "perfBrief.csv" in each subfolder
    subFolders = os.listdir(resultFolder)
    subFolders = filter(lambda s: s.startswith(subFolderPrefix), subFolders)
    subFolders = map(lambda s: os.path.join(resultFolder, s), subFolders)
    subFolders = filter(os.path.isdir, subFolders)  # filter only directories

    # get the "perfBrief.csv" from each subfolder
    testPerfDicts = {}
    for subfolder in subFolders:
        jsonFileName = os.path.join(subfolder, 'test_config.json')
        RCPQ_Learning_perf = os.path.join(subfolder, 'RCPQ_Learning_perf.pkl')
        RCPRTQ_perf = os.path.join(subfolder, 'RCPRTQ_perf.pkl')
        
        with open(jsonFileName, "r") as fp:
            configDict = json.load(fp)

        key = configDict[configAttributeName]
        testPerfDicts[key] = [key, None, None]

        if os.path.isfile(RCPQ_Learning_perf):
            with open(RCPQ_Learning_perf, 'rb') as f:
                testPerfDicts[key][1] = interpretPklData(pkl.load(f))
        
        if os.path.isfile(RCPRTQ_perf):
            with open(RCPRTQ_perf, 'rb') as f:
                testPerfDicts[key][2] = interpretPklData(pkl.load(f))
    plot_one_attribute_pkl(resultFolder, testPerfDicts, configAttributeName, attributeName)

def plot_one_attribute_pkl(resultFolder, testPerfDicts, configAttributeName, attributeName):
    perfVsConfigPermData = {}

    configAttributeValues = []
    RCPQ_Learning_data = []
    RCPRTQ_data = []
    for key in testPerfDicts: # "key" can be think as beta
        if testPerfDicts[key][0] is not None:
            configAttributeValues.append(testPerfDicts[key][0])

        if testPerfDicts[key][1] is not None:
            data = testPerfDicts[key][1]["clientPerfDf"][attributeName].iloc[-1]
            RCPQ_Learning_data.append(data)

        if testPerfDicts[key][2] is not None:
            data = testPerfDicts[key][2]["clientPerfDf"][attributeName].iloc[-1]
            RCPRTQ_data.append(data)
    
    if len(configAttributeValues):
        perfVsConfigPermData[configAttributeName] = configAttributeValues

    if len(RCPQ_Learning_data):
        perfVsConfigPermData["RCPQ_Learning"] = RCPQ_Learning_data
    
    if len(RCPRTQ_data):
        perfVsConfigPermData["RCPRTQ"] = RCPRTQ_data

    perfVsConfigPermDf = pd.DataFrame(data=perfVsConfigPermData)
    perfVsConfigPermDf.sort_values(configAttributeName, inplace=True)

    perfVsConfigPermDf.set_index(configAttributeName, inplace=True)

    # for plotting
    plt.clf()
    imgDir = os.path.join(resultFolder, "summary")
    os.makedirs(imgDir, exist_ok=True)

    timeDiscountList = perfVsConfigPermDf.index
    for protocolId, protocol in enumerate(perfVsConfigPermDf.columns):
        plt.plot(timeDiscountList, perfVsConfigPermDf[protocol], '-o', color=protocolColor[protocol], label=protocolName[protocol],)
        
    
    """We also want to plot the last 25% time performance of RTQ"""
    
    # plt.xlabel(configAttributeName)
    plt.xlabel("$\eta$")
    plt.ylabel(attributeName)
    plt.legend()
    for imgExtension in imgExtensions:
        imgPath = os.path.join(imgDir, attributeName + "_" + configAttributeName + imgExtension)
        plt.savefig(imgPath)
    # plt.show()
    print("Generating", imgPath)


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
        # firstDigit = int(configDict[configAttributeName])
        # secondDigit = int(configDict[configAttributeName]*10) % 10
        # key = str(firstDigit)+str(secondDigit)
        key = configDict[configAttributeName]

        timeDivider = float(configDict['timeDivider'])
        alpha = float(configDict['alpha'])
        beta = float(configDict['beta'])
        
        #RTQ
        if os.path.isfile(RTQEstFilename):
            maximumSimulationSmax = 15
            RTQPerf = np.loadtxt(RTQEstFilename, delimiter=",", usecols=[1])
            RTQ_pktLossRate, RTQ_delay, RTQ_delay_var, RTQ_rto, RTQ_smax =  RTQPerf
            RTQ_v, _ = theoTool.calc_V_theo_norm(RTQ_pktLossRate, timeDivider, alpha, beta, RTQ_delay, RTQ_delay_var, maximumSimulationSmax)
            # RTQ_v, _ = theoTool.calc_V_theo_uniform(RTQ_pktLossRate, timeDivider, alpha, beta, RTQ_delay-2*RTQ_delay_var, RTQ_delay+2*RTQ_delay_var, maximumSimulationSmax)
            RTQDict[key] = [RTQ_smax, RTQ_v]
        else:
            print("no ", RTQEstFilename)


        # Q-learning
        if os.path.isfile(QTableFilename):
            Q = np.loadtxt(QTableFilename, delimiter=",")
            v = np.max(Q, axis=1)
            for s in range(Q.shape[0]):
                if Q[s, 0] > Q[s, 1]:
                    break
            QDict[key] = [configDict[configAttributeName], Q, v, s]
        else:
            print("no ", QTableFilename)
    
    # orderedKey is a sorted based on beta values
    orderedKey = list(QDict.keys())
    orderedKey = sorted(orderedKey, key=lambda x: QDict[x][0])

    ## save RTQ
    if RTQDict:
        RTQ_vList = np.zeros((maximumSimulationSmax, len(orderedKey)))
        RTQ_sList = np.zeros((1, len(orderedKey)))
        for key_id, key in enumerate(orderedKey):
            s, v = RTQDict[key]
            RTQ_vList[:len(v), key_id] = v
            RTQ_sList[0, key_id] = s
        
        saveDir = os.path.join(resultFolder, "summary")
        header = ",".join([configAttributeName+"="+str(QDict[key][0]) for key in orderedKey])
        np.savetxt(os.path.join(saveDir, "RTQ_v.txt"), RTQ_vList, fmt="%f", delimiter=",", header=header, comments="")
        np.savetxt(os.path.join(saveDir, "RTQ_s.txt"), RTQ_sList, fmt="%d", delimiter=",", header=header, comments="")

    # save Q-Learning
    if QDict:
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


def process_one_attribute_time(resultFolder, subFolderPrefix, configAttributeName, attributeName):
    # scan folder and load the "perfBrief.csv" in each subfolder
    subFolders = os.listdir(resultFolder)
    subFolders = filter(lambda s: s.startswith(subFolderPrefix), subFolders)
    subFolders = map(lambda s: os.path.join(resultFolder, s), subFolders)
    subFolders = filter(os.path.isdir, subFolders)  # filter only directories

    # get the "perfBrief.csv" from each subfolder
    testPerfDicts = {}
    for subfolder in subFolders:
        jsonFileName = os.path.join(subfolder, 'test_config.json')
        RCPQ_Learning_perf = os.path.join(subfolder, 'RCPQ_Learning_perf.pkl')
        RCPRTQ_perf = os.path.join(subfolder, 'RCPRTQ_perf.pkl')
        
        with open(jsonFileName, "r") as fp:
            configDict = json.load(fp)

        key = configDict[configAttributeName]
        testPerfDicts[key] = [key, None, None, subfolder]

        if os.path.isfile(RCPQ_Learning_perf):
            with open(RCPQ_Learning_perf, 'rb') as f:
                testPerfDicts[key][1] = interpretPklData(pkl.load(f))
        
        if os.path.isfile(RCPRTQ_perf):
            with open(RCPRTQ_perf, 'rb') as f:
                testPerfDicts[key][2] = interpretPklData(pkl.load(f))
    plot_one_experiment_perf_time(resultFolder, testPerfDicts, configAttributeName, attributeName)


def plot_one_experiment_perf_time(resultFolder, testPerfDicts, configAttributeName, attributeName):
    perfVsConfigPermData = {}

    configAttributeValues = []
    
    for key in testPerfDicts: # "key" can be think as beta
        RCPQ_Learning_data = []
        RCPRTQ_data = []

        if testPerfDicts[key][0] is not None:
            configAttributeValues.append(testPerfDicts[key][0])

        if testPerfDicts[key][1] is not None:
            RCPQ_Learning_data = testPerfDicts[key][1]["clientPerfDf"][attributeName]
            

        if testPerfDicts[key][2] is not None:
            RCPRTQ_data = testPerfDicts[key][2]["clientPerfDf"][attributeName]
            
        if testPerfDicts[key][3] is not None:

            # RCP + Q Learning
            plt.clf()
            plt.plot(list(range(len(RCPQ_Learning_data))), RCPQ_Learning_data, '-', color=protocolColor["RCPQ_Learning"], label=protocolName["RCPQ_Learning"],)
            # plt.xlabel(configAttributeName)
            plt.xlabel("time")
            plt.ylabel(attributeNameDict_pkl[attributeName])
            plt.legend()
            for imgExtension in imgExtensions:
                imgPath = os.path.join(testPerfDicts[key][3], attributeName + "_" + protocolName["RCPQ_Learning"] +"_overtime" + imgExtension)
                plt.savefig(imgPath)
            # plt.show()
            print("Generating", imgPath)

            # RCP Heuristic
            plt.clf()
            plt.plot(list(range(len(RCPRTQ_data))), RCPRTQ_data, '-', color=protocolColor["RCPRTQ"], label=protocolName["RCPRTQ"],)
            # plt.xlabel(configAttributeName)
            plt.xlabel("time")
            plt.ylabel(attributeNameDict_pkl[attributeName])
            plt.legend()
            for imgExtension in imgExtensions:
                imgPath = os.path.join(testPerfDicts[key][3], attributeName + "_" + protocolName["RCPRTQ"] +"_overtime" + imgExtension)
                plt.savefig(imgPath)
            # plt.show()
            print("Generating", imgPath)


            plt.clf()
            plt.plot(list(range(len(RCPQ_Learning_data))), RCPQ_Learning_data, '-', color=protocolColor["RCPQ_Learning"], label=protocolName["RCPQ_Learning"],)
            plt.plot(list(range(len(RCPRTQ_data))), RCPRTQ_data, '-', color=protocolColor["RCPRTQ"], label=protocolName["RCPRTQ"],)
            # plt.xlabel(configAttributeName)
            plt.xlabel("time")
            plt.ylabel(attributeNameDict_pkl[attributeName])
            plt.legend()
            for imgExtension in imgExtensions:
                imgPath = os.path.join(testPerfDicts[key][3], attributeName + "_overtime" + imgExtension)
                plt.savefig(imgPath)
            # plt.show()
            print("Generating", imgPath)

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
    # opts = DummyClass()
    # opts.resultFolder = "Results/dynamic_channel_error_TimeDiscount_alpha_2_0"
    # opts.subFolderPrefix = "TimeDiscount"
    # opts.configAttributeName = "pktDropProb"

    for attributeName in attributeNameDict_csv:
        process_one_attribute_csv(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName, attributeName)
    
    # plot attributes that are in the pkl file
    for attributeName in attributeNameDict_pkl:
        process_one_attribute_pkl(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName, attributeName)

    for attributeName in attributeNameDict_time:
        process_one_attribute_time(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName, attributeName)


    gather_Q_table(opts.resultFolder, opts.subFolderPrefix, opts.configAttributeName)
