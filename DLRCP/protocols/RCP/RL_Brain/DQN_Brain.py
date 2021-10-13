import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Memory import ExperienceMemory
from .DecisionBrain import DecisionBrain


class DQNNet(nn.Module):
    """Our decision making network that maps states to its Q value"""

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


class DQN_Brain(DecisionBrain):

    def __init__(self,
                 stateDim: int,                 # dimension of the system state
                 nActions: int,                 # dimension of the action space
                 batchSize: int = 64,           #
                 memoryCapacity: int = 1e4,     # maximum number of experiences to store
                 learningRate: float = 1e-6,    # initial learning rate
                 learningPeriod: int = 1,       # how often to learn
                 lrDecay: float = 0.95,         # the multiplier to decay the learning rate
                 # lrDecayInterval: int = 100,   # the period to decay the learning rate
                 updateFrequency: int = 100,    # period to copying evalNet weights to tgtNet
                 epsilon: float = 0.95,         # greedy policy parameter
                 epsilon_decay: float = 0.99,   # the decay of greedy policy parameter, epsilon
                 convergeLossThresh: int = 1,   # turn off greedy policy when loss is below
                 eta: float = 0.9,              # reward discount
                 loglevel: int = DecisionBrain.LOGLEVEL,
                 verbose=None                   # deprecated
                 ) -> None:

        super().__init__(convergeLossThresh=convergeLossThresh, epsilon=epsilon, epsilon_decay=epsilon_decay, loglevel=loglevel)

        # automatically transfer to cuda if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.nActions = nActions
        self.nStates = stateDim

        # create eval net and target net
        # evaluation network (learning from back-propagation)
        self.evalNet = DQNNet(nStates=self.nStates,
                              nActions=self.nActions).to(self.device)
        # target network (learning from copying evalNet weights)
        self.tgtNet = DQNNet(nStates=self.nStates,
                             nActions=self.nActions).to(self.device)

        # selection of optimizer
        self.optimizer = optim.Adam(
            self.evalNet.parameters(), lr=learningRate, weight_decay=0)

        # once invoked (.step()), the learning rate will decrease to its gamma times
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=lrDecay)

        # loss function
        self.loss = sys.maxsize         # initial loss
        self.lossFunc = nn.MSELoss()    # since we are doing "regression", we use MSE
        # self.lossFunc = nn.L1Loss()

        # RL counters
        # if self.divergeCounter > 3, turn on Greedy because of bad perf
        self.divergeCounter = sys.maxsize
        self.convergeCounter = 0

        self.batchSize = batchSize
        self.learningCounter = 0  # learning steps before updating tgtNet
        self.learningPeriod = learningPeriod
        # how often to update the network parameter
        self.updateFrequencyFinal = updateFrequency
        self.updateFrequencyCur = self.updateFrequencyFinal/2

        # memory of experiences
        self.memoryCapacity = int(memoryCapacity)
        self.memory = ExperienceMemory(
            nStates=self.nStates, capacity=self.memoryCapacity)

        # reward discount
        self.eta = eta

        # other input parameters
        self.updateFrequency = updateFrequency

    def loadModel(self, modelFile):
        self.evalNet.loadModel(modelFile)
        self.tgtNet.loadModel(modelFile)

    def saveModel(self):
        self.evalNet.saveModel()

    def chooseMaxQAction(self, state):
        """choose the action that counts for the maximum Q value"""
        state = torch.unsqueeze(torch.FloatTensor(
            state), 0).to(self.device)  # to vector
        actionRewards = self.evalNet.forward(state)
        action = actionRewards.argmax(
            dim=1, keepdim=False).data.numpy()[0]
        return action

    def digestExperience(self, prevState, action, reward, curState):
        self.memory.storeOneExperience(prevState, action, reward, curState)
        self.learningCounter = (self.learningCounter + 1) % self.learningPeriod
        if self.learningCounter % self.learningPeriod == self.learningPeriod-1:
            self.learn()

    def learn(self):
        # check whether to update tgtNet
        if self.memory.nExp <= 0:
            return

        # if self.learningCounter > self.updateFrequencyCur:
        if self.learningCounter > self.updateFrequencyFinal:
            # transfer the weight of the evalNet to tgtNet
            self.tgtNet.load_state_dict(self.evalNet.state_dict())
            self.decayEpsilon()

            self.learningCounter = 0
        self.learningCounter += 1

        # randomly sample $batch experiences
        states, actions, rewards, nextStates = self.memory.getRandomExperiences(
            quantity=self.batchSize)

        # use the history state, history action, reward, and the ground truth new state
        # to train a regression network that predicts the reward correct.
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        nextStates = torch.FloatTensor(nextStates).to(self.device)

        # q value based on evalNet
        # select the maximum reward based on actions
        curQ = self.evalNet(states).gather(dim=1, index=actions).squeeze()

        # q value based on the system
        # no gradient needed (detach) <=> no back propagation needed
        nextQ = self.tgtNet(nextStates).detach()
        # apply Bellman's equation
        # tgtQ = rewards + self.eta * nextQ.max(dim=1)[0].view(-1, 1)
        tgtQ = rewards + self.eta * nextQ.max(dim=1)[0]

        loss = self.lossFunc(curQ, tgtQ)

        self.loss = loss.cpu().detach().numpy()
        if self.learningCounter == 0:
            # self.lr_scheduler.step() # decay learning rate
            self.logger.info("loss=", self.loss)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        super().learn()
