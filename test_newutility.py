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


# utilityMethodList = ["TimeDiscount", "SumPower"]
# utilityMethodList = ["TimeDiscount"]
utilityMethodList = ["SumPower"]
alphaList = [2]
alphaDigitPrecision = 2
# alphaList = [0.5, 1, 2, 3, 4]
betaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
betaDigitPrecision = 3 # number of digits to represent beta
# betaList = [0.8]
# betaList = [0.9, 1]

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


def run_test_beta(args):
    beta, alpha, utilityMethod, resultFolderName, tempFileFolderName = args
    print("conducting experiment for ", utilityMethod, " beta=", beta)
    testDesc = utilityMethod+"_{serializedDigit}".format(
        serializedDigit="_".join(serializeDigit(beta, betaDigitPrecision))
    )
    argList = [PYTHON3, "runTest.py",
                    "--testPeriod", "40000",
                    "--bgClientNum", "3",
                    "--serviceRate", "3",
                    "--pktDropProb", "0.3",
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
                    # "--addRCPRTQ", # RTQ cannot work with this utility function
                    "--utilityMethod", "SumPower"
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
                "Results", "case_newutility_" + utilityMethod+"_alpha_"+alpha_desc)

            tempFileFolderName = os.path.join(resultFolderName, "tempResult")

            argList = []
            for expId, beta in enumerate(betaList):
                args = [beta, alpha, utilityMethod, resultFolderName, tempFileFolderName]
                argList.append(args)
            
            # must run one test to generate the temp result for UDP/ARQ
            run_test_beta(argList[0])

            # use multiprocessing to generate the remaining test results
            n_worker = multiprocessing.cpu_count()
            needed_worker = min(n_worker-1, len(argList[1:]))
            pool = multiprocessing.Pool(processes=needed_worker)
            pool.map(run_test_beta, argList[1:])
            pool.close()
            pool.join()

            subprocess.run([PYTHON3, "plot_testResults.py", "--resultFolder", resultFolderName,
                        "--subFolderPrefix", utilityMethod, "--configAttributeName", 'beta'])
    endTime = time.time()
    print("running all simulations in ", endTime-startTime, " seconds")
if __name__ == "__main__":
    main()