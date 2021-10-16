"""
In this experiment, we test the changes of utility over different time discount.

utility = timeDiscount ^ (delay) * deliveryRate^alpha

we set alpha = 2 (default value) in this experiment.

timeDiscount->1 : small reaction to delay
timeDiscount->0 : large reaction to delay
"""
import os, sys 
import subprocess

PYTHON3 = sys.executable # get the python interpreter 


# utilityMethodList = ["SumPower", "TimeDiscount"]
# utilityMethodList = ["TimeDiscount"]
utilityMethodList = ["SumPower"]
alphaList = [2]
# alphaList = [0.5, 1, 2, 3, 4]
betaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# betaList = [0.9, 1]

for utilityMethod in utilityMethodList:
    for alpha in alphaList:
        alphaFirstDigit = int(alpha) % 10
        alphaSecondDigit = int(alpha*10) % 10
        alpha_desc = "{firstDigit}_{secondDigit}".format(
                firstDigit = alphaFirstDigit,
                secondDigit = alphaSecondDigit
            )
        resultFolderName = os.path.join("Results", utilityMethod+"_alpha_"+alpha_desc)
        tempFileFolderName = os.path.join(resultFolderName, "tempResult")

        for expId, beta in enumerate(betaList):
            print("conducting experiment for ", utilityMethod, " beta=", beta)
            testDesc = utilityMethod+"_{firstDigit}_{secondDigit}".format(
                firstDigit = int(beta) % 10,
                secondDigit = int(beta* 10) % 10
            )
            subprocess.run([PYTHON3, "runTest.py", 
            "--utilityMethod", utilityMethod, 
            "--beta", str(beta), 
            "--testDesc", testDesc, 
            "--data-dir", resultFolderName, 
            "--nonRCPDatadir", tempFileFolderName, 
            "--alpha", str(alpha),
            ])

        subprocess.run([PYTHON3, "plot_testResults.py", "--resultFolder", resultFolderName, "--subFolderPrefix", utilityMethod, "--configAttributeName", 'beta'])
