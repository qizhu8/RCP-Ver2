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
beta = 0.8
pktLossList = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def run_test_beta(args):
    beta, alpha, pktLoss, utilityMethod, resultFolderName = args
    print("conducting experiment for ", utilityMethod, " pktloss=", pktLoss)
    testDesc = "pktloss"+"_{firstDigit}_{secondDigit}".format(
        firstDigit=int(pktLoss) % 10,
        secondDigit=int(pktLoss * 10) % 10
    )
    subprocess.run([PYTHON3, "runTest.py",
                    "--bgClientNum", "0",
                    "--serviceRate", "3",
                    "--pktDropProb", str(pktLoss),
                    "--channelDelay","100", "150",
                    # "--fillChannel",
                    "--utilityMethod", utilityMethod,
                    "--beta", str(beta),
                    "--testDesc", testDesc,
                    "--data-dir", resultFolderName,
                    "--nonRCPDatadir", "",
                    "--alpha", str(alpha),
                    #add test protocol
                    "--addUDP",
                    "--addARQInfinite",
                    # "--addARQFinite",
                    "--addRCPQLearning",
                    # "--addRCPDQN",
                    "--addRCPRTQ",
                    ])

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
                "Results", "case_study_pktLoss_" + utilityMethod+"_alpha_"+alpha_desc)

            

            argList = []
            for expId, pktLoss in enumerate(pktLossList):
                args = [beta, alpha, pktLoss, utilityMethod, resultFolderName]
                argList.append(args)
            
            # must run one test to generate the temp result for UDP/ARQ

            # use multiprocessing to generate the remaining test results
            n_worker = multiprocessing.cpu_count()
            needed_worker = min(n_worker-1, len(argList))
            pool = multiprocessing.Pool(processes=needed_worker)
            pool.map(run_test_beta, argList)
            pool.close()
            pool.join()

            subprocess.run([PYTHON3, "plot_testResults.py", "--resultFolder", resultFolderName,
                        "--subFolderPrefix", "pktloss", "--configAttributeName", 'pktDropProb'])
    endTime = time.time()
    print("running all simulations in ", endTime-startTime, " seconds")

if __name__ == "__main__":
    main()