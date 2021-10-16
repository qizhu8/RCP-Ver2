"""
In this experiment, we test the changes of utility over different time discount.

utility = timeDiscount ^ (delay) * deliveryRate^alpha

we set alpha = 2 (default value) in this experiment.

timeDiscount->1 : small reaction to delay
timeDiscount->0 : large reaction to delay
"""
import os
import subprocess

alphaList = [0.5, 1]
# alphaList = [0.5, 1, 2, 3, 4]
# alphaList = [2]
timeDiscountList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for alpha in alphaList:
    alphaFirstDigit = int(alpha) % 10
    alphaSecondDigit = int(alpha*10) % 10
    alpha_desc = "{firstDigit}_{secondDigit}".format(
            firstDigit = alphaFirstDigit,
            secondDigit = alphaSecondDigit
        )
    resultFolderName = "Results/ChangeTimeDiscount_alpha_"+alpha_desc
    tempFileFolderName = os.path.join(resultFolderName, "tempResult")

    for expId, timeDiscount in enumerate(timeDiscountList):
        print("conducting experiment for timeDiscount", timeDiscount)
        testDesc = "timeDiscount_{firstDigit}_{secondDigit}".format(
            firstDigit = int(timeDiscount) % 10,
            secondDigit = int(timeDiscount* 10) % 10
        )
        subprocess.run(["python3", "runTest.py", "--timeDiscount", str(timeDiscount), "--testDesc", testDesc, "--data-dir", resultFolderName, 
        "--nonRCPDatadir", tempFileFolderName, 
        "--alpha", str(alpha)])

    subprocess.run(["python3", "plot_timeDiscount.py", "--resultFolder", resultFolderName])
