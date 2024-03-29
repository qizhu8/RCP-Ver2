"""
In this experiment, we test the changes of utility over different time discount.

utility = timeDiscount ^ (delay) * deliveryRate^alpha

we set alpha = 2 (default value) in this experiment.

timeDiscount->1 : small reaction to delay
timeDiscount->0 : large reaction to delay
"""
import os
import sys
import time
import subprocess
import multiprocessing

if len(sys.argv) >= 2:
    subprocess.run(["rm -rf", "Results"], shell=True)

PYTHON3 = sys.executable  # get the python interpreter

experimentDesc = "dynamic_channel_error"
utilityMethodList = ["TimeDiscount"]
alphaList = [2]
alphaDigitPrecision = 2
betaList = [0.8]
# betaList = [0.8]
betaDigitPrecision = 3 # number of digits to represent beta
errorList = [0.0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3]
errorDigitPrecision = 3
testPeriod = 40000


def serializeDigit(num, digitPrec):
    """
    turn number to a list of digits
    E.g, 0.82 -> [0, 8, 2]
    """
    s = []
    for _ in range(digitPrec):
        s.append(str(int(num) % 10))
        num *= 10
    return s

def formulateChInstructions(chInstructList):
    """
    insList = ['10', 'serviceRate', '3', '20', 'channelDelay', '100,200', '30', 'channelDelay', '100','nonsense']
    """
    insList = []
    for ins in chInstructList:
        t, attrib, value = ins
        insList.append(str(int(t)))
        insList.append(str(attrib))
        insList.append(str(value))
    return " ".join(insList)


def run_test_beta(args):
    beta, alpha, errorRate, utilityMethod, resultFolderName, tempFileFolderName = args
    print("conducting experiment for ", utilityMethod, " error rate =", errorRate)
    testDesc = "channel_error"+"_{serializedDigit}".format(
        serializedDigit="_".join(serializeDigit(errorRate, errorDigitPrecision))
    )
    argList = [PYTHON3, "runTest.py",
                    "--testPeriod", str(int(testPeriod)),
                    "--bgClientNum", "0",
                    "--serviceRate", "4",
                    "--pktDropProb", str(errorRate),
                    "--channelDelay","100", "150",
                    # "--fillChannel",
                    "--utilityMethod", utilityMethod,
                    "--beta", str(beta),
                    "--testDesc", testDesc,
                    "--data-dir", resultFolderName,
                    "--nonRCPDatadir", tempFileFolderName,
                    "--alpha", str(alpha),
                    #add test protocols
                    "--addUDP",
                    "--addARQInfinite",
                    # "--addARQFinite",
                    "--addRCPQLearning",
                    # "--addRCPDQN",
                    "--addRCPRTQ",
                    "--no-load-status", # we shouldn't load previously stored UDP and ARQ status due to the changes of channel
                    ]
    # whether to use multi-processing to run the test of different protocols
    # Appropriate for the first test

    subprocess.run(argList)

def main():
    startTime = time.time()
    for utilityMethod in utilityMethodList:
        for alpha in alphaList:
            alpha_desc = "_".join(serializeDigit(alpha, alphaDigitPrecision))

            resultFolderName = os.path.join(
                "Results", experimentDesc + "_" + "channel_error" + "_alpha_"+alpha_desc)

            tempFileFolderName = os.path.join(resultFolderName, "tempResult")

            argList = []
            for beta in betaList:
                for errorRate in errorList:
                    args = [beta, alpha, errorRate, utilityMethod, resultFolderName, tempFileFolderName]
                    argList.append(args)
            
            # must run one test to generate the temp result for UDP/ARQ
            run_test_beta(argList[0])

            # use multiprocessing to generate the remaining test results
            n_worker = multiprocessing.cpu_count()
            needed_worker = min(n_worker-1, len(argList[1:]))
            if needed_worker:
                pool = multiprocessing.Pool(processes=needed_worker)
                pool.map(run_test_beta, argList[1:])
                pool.close()
                pool.join()


            os.makedirs(os.path.join(resultFolderName, "summary"), exist_ok=True)
            # save the command to run the plot generation command
            cmd = " ".join([PYTHON3, "plot_testResults.py", "--resultFolder", resultFolderName,
                        "--subFolderPrefix", "channel_error", "--configAttributeName", 'beta'])
            with open(os.path.join(resultFolderName, "summary", "plot_cmd.sh"), 'w') as f:
                f.write(cmd)

            subprocess.run([PYTHON3, "plot_testResults.py", "--resultFolder", resultFolderName,
                        "--subFolderPrefix", "channel_error", "--configAttributeName", 'pktDropProb'])
            
                
    endTime = time.time()
    print("running all simulations in ", endTime-startTime, " seconds")
if __name__ == "__main__":
    main()