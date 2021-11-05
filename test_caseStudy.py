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
utilityMethodList = ["TimeDiscount"]
# utilityMethodList = ["SumPower"]
alphaList = [2]
# alphaList = [0.5, 1, 2, 3, 4]
betaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# betaList = [0.8]
# betaList = [0.9]


def run_test_beta(args):
    beta, alpha, utilityMethod, resultFolderName, tempFileFolderName = args
    print("conducting experiment for ", utilityMethod, " beta=", beta)
    testDesc = utilityMethod+"_{firstDigit}_{secondDigit}".format(
        firstDigit=int(beta) % 10,
        secondDigit=int(beta * 10) % 10
    )
    argList = [PYTHON3, "runTest.py",
                    "--bgClientNum", "0",
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
                    #add test protocol
                    "--addUDP",
                    "--addARQInfinite",
                    # "--addARQFinite",
                    "--addRCPQLearning",
                    # "--addRCPDQN",
                    "--addRCPRTQ",
                    ]

    subprocess.run(argList)

def main():
    startTime = time.time()
    for utilityMethod in utilityMethodList:
        for alpha in alphaList:
            alphaFirstDigit = int(alpha) % 10
            alphaSecondDigit = int(alpha*10) % 10
            alpha_desc = "{firstDigit}_{secondDigit}".format(
                firstDigit=alphaFirstDigit,
                secondDigit=alphaSecondDigit
            )
            resultFolderName = os.path.join(
                "Results", "case_study_" + utilityMethod+"_alpha_"+alpha_desc)

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
            if needed_worker > 0: # we still have work to do
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