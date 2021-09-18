"""
As the filename describes, this script implement an Reinforment Learning decision maker.
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DQNNet(nn.Module):
    """Our decision making network"""

    def __init__(self, nStates, nActions):
        super(DQNNet, self).__init__()

        # two layers
        self.fc1 = nn.Linear(nStates, 20)
        self.fc2 = nn.Linear(20, 30)
        self.out = nn.Linear(30, nActions)

    def forward(self, state):
        # two layers
        x = torch.sigmoid(self.fc1(state))
        x = torch.sigmoid(self.fc2(x))

        return self.out(x)

    def saveModel(self):
        modelName = "model_" + time.strftime("%Y%m%d_%H%M%S")
        torch.save(self.state_dict(), modelName)

    def loadModel(self, modelName):
        self.load_state_dict(torch.load(modelName))
        self.eval()

class Experience(object):
        """
        The data structure to store one piece of experience
        """
        def __init__(self, prevState, action, reward, curState) -> None:
            self.prevState = prevState
            self.action = action
            self.reward = reward
            self.curState = curState

class ExperienceMemory(object):
    """
    We use a ring buffer to store each experience pieces.
    """

    def __init__(self, nStates:int, capacity:int) -> None:
        """
        inputs:
            nState: the dimension of the DQN state
            capacity: the maximum number of experiences to store
        """
        self.nStates = nStates
        self.capacity = capacity
        
        self.memory = [None for _ in range(self.capacity)]
        self.nxtMemPtr = 0 # pos in memory to store new experience
        self.nExp = 0 # number of experiences in memory
    
    def storeOneExperience(self, prevState, action, reward, curState)->None:
        self.memory[self.nxtMemPtr] = Experience(prevState=prevState, action=action, reward=reward, curState=curState)
        
        self.nxtMemPtr = (self.nxtMemPtr+1) % self.capacity
        if self.nExp < self.capacity:
            self.nExp = self.nExp+1 
    
    def getExperiences(self, quantity:int):
        """
        Randomly return min(quantity, availableExperience) experiences.
        """
        sampleIdxs = np.random.choice(self.nExp, min(self.nExp, quantity))

        prevStates, actions, rewards, curStates = [], [], [], []
        for idx in sampleIdxs:
            prevStates.append(self.memory[idx].prevState)
            actions.append(self.memory[idx].action)
            rewards.append(self.memory[idx].reward)
            curStates.append(self.memory[idx].curState)
        
        return prevStates, actions, rewards, curStates


class DQN(object):
    LOGLEVEL = logging.INFO

    def __init__(self,
                 nStates: int,                  # dimension of the system state
                 nActions: int,                 # dimension of the action space
                 batchSize: int = 64,           #
                 memoryCapacity: int = 1e4,     # maximum number of experiences to store
                 learningRate: float = 1e-6,    # initial learning rate
                 lrDecay: float = 0.95,         # the multiplier to decay the learning rate
                 lrDecayInterval: int = 100,    # the period to decay the learning rate
                 updateFrequency: int = 100,    # period to copying evalNet weights to tgtNet
                 epsilon: float = 0.95,         # greedy policy parameter
                 convergeLossThresh: int = 1,   # turn off greedy policy when loss is below
                 eta: float = 0.9,              # reward discount
                 deviceStr: str = "cuda",       # preferred computing device cpu or cuda
                 epsilon_decay: float = 0.99,   # the decay of greedy policy parameter, epsilon
                 loglevel: int = LOGLEVEL,
                 ) -> None:

        self.initLogger(loglevel)

        # automatically transfer to cuda if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.nActions = nActions
        self.nStates = nStates

        # create eval net and target net
        # evaluation network (learning from back-propagation)
        self.evalNet = DQNNet(nStates=self.nStates,
                              nActions=self.nActions).to(self.device)
        # target network (learning from copying evalNet weights)
        self.tgtNet = DQNNet(nStates=self.nStates,
                             nActions=self.nActions).to(self.device)

        # selection of optimizer
        self.optimizer = torch.optim.Adam(
            self.evalNet.parameters(), lr=learningRate)
        # self.optimizer = torch.optim.RMSprop(self.evalNet.parameters(), lr=learningRate)
        # self.optimizer = torch.optim.SGD(self.evalNet.parameters(), lr=learningRate)

        # once invoked (.step()), the learning rate will decrease to its gamma times
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=lrDecay)

        # loss function
        self.loss = sys.maxsize  # initial loss
        self.lossFunc = nn.MSELoss()

        # greedy policy epsilon and its decay
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # RL counters
        # if self.divergeCounter > 3, turn on Greedy because of bad perf
        self.divergeCounter = sys.maxsize
        self.convergeCounter = 0

        self.batchSize = batchSize
        self.learningCounter = 0  # learning steps before updating tgtNet
        # how often to update the network parameter
        self.updateFrequencyFinal = updateFrequency
        self.updateFrequencyCur = self.updateFrequencyFinal/2

        # memory of experiences
        # self.memoryCounter = 0  # number of experience pieces
        # self.memoryCapacity = int(memoryCapacity)
        # storing [curState, action, reward, nextState], so nStates*2 + 2
        # self.memory = np.zeros((self.memoryCapacity, nStates*2+2))
        self.memory = ExperienceMemory(nStates=self.nStates, capacity=self.memoryCapacity)

        # reward discount
        self.eta = eta

        # other input parameters
        self.convergeLossThresh = convergeLossThresh
        self.globalEvalOn = False  # True: ignore greedy random policy
        self.isConverge = False  # whether the network meets the convergence condition
        self.updateFrequency = updateFrequency

    def initLogger(self, loglevel):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:{classname}:%(message)s'.format(classname=type(self).__name__))
            sh.setFormatter(formatter)
            # self.logger.addHandler(sh)

    def loadModel(self, modelFile):
        self.evalNet.loadModel(modelFile)
        self.tgtNet.loadModel(modelFile)

    def saveModel(self):
        self.evalNet.saveModel()

    def chooseAction(self, state, evalOn=False):
        state = torch.unsqueeze(torch.FloatTensor(
            state), 0).to(self.device)  # to vector

        # epsilon greedy
        if evalOn or self.globalEvalOn or np.random.uniform() < self.epsilon:
            # actionRewards if of shape 1 x nAction
            actionRewards = self.evalNet.forward(state)

            # action = torch.argmax(actionRewards, 1)
            # the [1] pointed to argmax
            # action = torch.max(actionRewards, 1)[1]
            # action = action.cpu().data.numpy()[0]

            action = actionRewards.argmax(
                dim=0, keepdim=False).cpu().data.numpy()
        else:
            action = np.random.randint(0, self.nActions)
        return action

    def storeExperience(self, s, a, r, s_):
        experience = np.hstack((s, [a, r], s_))
        storageAddr = self.memoryCounter % self.memoryCapacity
        self.memory[storageAddr, :] = experience
        self.memoryCounter += 1

    def learn(self):
        # check whether to update tgtNet
        if self.memory.nExp <= 0:
            return

        # if self.learningCounter > self.updateFrequencyCur:
        if self.learningCounter > self.updateFrequencyFinal:
            self.tgtNet.load_state_dict(self.evalNet.state_dict())
            self.learningCounter = 0
            self.epsilon = 1-(1 - self.epsilon)*self.epsilon_decay
        self.learningCounter += 1

        # randomly sample $batch experiences
        availableExperiences = min(self.memoryCapacity, self.memoryCounter)

        # sampleIdxs = np.random.choice(availableExperiences, min(
        #     availableExperiences, self.batchSize))
        # experiences = self.memory[sampleIdxs, :]
        states, actions, rewards, nextStates = self.memory.getExperiences(quantity=self.batchSize)

        # use the history state, history action, reward, and the ground truth new state
        # to train a regression network that predicts the reward correct.
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        nextStates = torch.FloatTensor(nextStates).to(self.device)

        # q value based on evalNet
        # select the maximum reward based on actions
        curQ = self.evalNet(states).gather(dim=1, index=actions)

        # q value based on the system
        # no gradient needed (detach) <=> no back propagation needed
        nextQ = self.tgtNet(nextStates).detach()
        # apply Bellman's equation
        tgtQ = rewards + self.eta * nextQ.max(dim=1)[0].view(-1, 1) 
        loss = self.lossFunc(curQ, tgtQ)

        self.loss = loss.cpu().detach().numpy()
        if self.learningCounter == 0:
            # self.lr_scheduler.step() # decay learning rate
            self.logger.info("loss=", self.loss)

        if loss < self.convergeLossThresh:
            self.isConverge = True

        if loss > 20*self.convergeLossThresh:
            self.isConverge = False

        # if loss > 100*self.convergeLossThresh:
        #     self.epsilon = self.epsilon_init

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
