import numpy as np

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
    
    def getRandomExperiences(self, quantity:int, includeLastAdded:bool=True):
        """
        Randomly return min(quantity, availableExperience) experiences.
        includeLastAdded: True: always include the last added experience
        """
        prevStates, actions, rewards, curStates = [], [], [], []
        
        if includeLastAdded:
            quantity -= 1

        if quantity >= 0:
            sampleIdxs = np.random.choice(self.nExp, min(self.nExp, quantity))
    
        if includeLastAdded:
            lastIndex = (self.nxtMemPtr -1 + self.capacity) % self.capacity
            sampleIdxs = np.append(sampleIdxs, lastIndex)

        for idx in sampleIdxs:
            prevStates.append(self.memory[idx].prevState)
            actions.append([self.memory[idx].action])  # torch.gather prefered [[act1], [act2]]
            rewards.append(self.memory[idx].reward)
            curStates.append(self.memory[idx].curState)
        
        return prevStates, actions, rewards, curStates