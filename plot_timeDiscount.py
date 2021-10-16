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
        "system utility (last 25%)": "l25p util"
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
    
    # for protocolId, protocol in enumerate(protocols):
    #     print(protocol, protocolPerf[protocol])
    # print(orderedKey)

    # for plotting
    plt.clf()
    imgDir = os.path.join(resultFolder, "summary")
    os.makedirs(imgDir, exist_ok=True)
    imgPath = os.path.join(imgDir, attributeName + "_" + configAttributeName + ".png")

    timeDiscountList = [testPerfDicts[x][0] for x in orderedKey]
    for protocolId, protocol in enumerate(protocols):
        plt.plot(timeDiscountList, protocolPerf[protocol], label=protocol)
    
    plt.xlabel(configAttributeName)
    plt.ylabel(attributeName)
    plt.legend()
    plt.savefig(imgPath)
    # plt.show()
    return protocols, protocolPerf


def main(resultFolder, subFolderPrefix, configAttributeName, attributeName):
    

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
        # get the timeDiscount
        firstDigit = int(configDict[configAttributeName])
        secondDigit = int(configDict[configAttributeName]*10) % 10
        key = str(firstDigit)+str(secondDigit)

        testPerfDicts[key] = [configDict[configAttributeName], pd.read_csv(csvFileName, delimiter=',')]
    
    plot_one_attribute(testPerfDicts, attributeName, configAttributeName, attributeNameDict, resultFolder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RCP-Ver 2.0 - plot for time discount')

    parser.add_argument('--resultFolder', type=str, default= '', required=True,
                        help='configuration file (.json) to be loaded')

    opts = parser.parse_args()
    resultFolders = []
    resultFolders.append(opts.resultFolder)
    
    # resultFolders = [
    #     "./Results/ChangeTimeDiscount_alpha_0_5/",
    #     "./Results/ChangeTimeDiscount_alpha_1_0/",
    #     "./Results/ChangeTimeDiscount_alpha_2_0/",
    #     "./Results/ChangeTimeDiscount_alpha_3_0/",
    #     "./Results/ChangeTimeDiscount_alpha_4_0/",
    #     ]
    subFolderPrefix = "timeDiscount"
    configAttributeName = "timeDiscount"
    attributeNameList=attributeNameDict.keys()
    for resultFolder in resultFolders:
        for attributeName in attributeNameList:
            main(resultFolder, subFolderPrefix, configAttributeName, attributeName)