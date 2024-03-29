# RCP-Ver2

This project is for the Reinforcement Learning based retransmission control protocol (RCP). 

# Preparation

* Python3 >= 3.7
* Python Libraries to run tests:
    * numpy >= 1.19.0               for numerical calculation and data storage
    * Pytorch >= 1.7.0 (optional)   for the DQN implementation (this method is not included in the paper)
    * tabulate >= 0.8.0             for displaying final results
* Python Libraries to plot figure
    * matplotlib

You can easily install all libraries by

```
# manually install
python3 -m pip install numpy, torch, tabulate

# follow our provided requirements list 
python3 -m pip install -y requirements.txt
```

# Run the first test

We provide you with a runable python script with default parameters.

```
python3 runTest.py
```

You are expected to see results in the terminal while running and in ```Results/test/```.

# Run with customized test

## Reproduce experiment results

```
# run the experiment 1 code. The plots can be seen in ```Results/case_study_TimeDiscount_alpha_2_0/summary/```
python3 test_caseStudy.py

# run the experiment 2 code. The plots can be seen in ```Results/case_study_competition_TimeDiscount_alpha_2_0/summary/```
python3 test_competition.py
```

## By feeding different attributes

```
# check the possible attributes
python3 runTest.py --help

# delete potential previous results in the data folder
python3 runTest.py --clean-run

# change the channel service rate to 4 pkts/tick
python3 runTest.py --serviceRate 4
```

## By feeding a configuration json file

```
python3 runTest.py --configFile your_config.json
```
