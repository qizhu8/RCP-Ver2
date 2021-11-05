import heapq
import sys
from typing import List, Union

class Instruction(object):
    """This class stores the basic information of instructions"""
    def __init__(self, time:int, attributeName: str, value) -> None:
        self.time = time
        self.attributeName = attributeName
        self.value = value

    def __lt__(self, other) -> bool:
        return self.time < other.time
    
    def __str__(self) -> str:
        msg = "{time}: {attr}={val}".format(
            time=self.time,
            attr=self.attributeName,
            val=self.value
        )
        return msg

class Scheduler(object):
    """
    This class defines a scheduler that instructs the time to make the change to the system.
    """
    # mapping of configeration attribute name and the parsing function
    attribTypeDict = {
        "serviceRate": int,
        "bufferSize": int,
        "pktDropProb": float,
        "channelDelay": int,  # or list of delay
        "fillChannel": bool,
        "alpha":float,
        "beta":float,
    }

    def __init__(self, insList:List=[]) -> None:
        super().__init__()

        self.instructions = [] # a heap of instructions.
        self.addInstructionList(insList)

    def getNextInstructTime(self):
        """
        Return the schedule time of the next instruction in list.
        """
        if self.instructions:
            return self.instructions[0].time
        else:
            return sys.maxsize
    
    def addInstructionList(self, insList)->None:
        insNum = len(insList) // 3

        for ins_id in range(insNum):
            scheduleTime, attributeName, value = insList[ins_id*3:ins_id*3+3]
            if attributeName in Scheduler.attribTypeDict:
                if attributeName in {"channelDelay"}:
                    valueList = value.split(',')
                    if len(valueList) > 1: # channel with multiple attributes
                        value = [int(val) for val in valueList]
                    else:
                        value = int(valueList[0])
                else:
                    value = Scheduler.attribTypeDict[attributeName](value)
                self.addInstruction(int(scheduleTime), attributeName, value)

    def addInstruction(self, scheduleTime, attributeName, value):
        """Add the instruction to scheduler"""
        heapq.heappush(self.instructions, Instruction(scheduleTime, attributeName, value))
    
    def getInstruction(self, curtime:int) -> List[Instruction]:
        """Pop instructions if curtime meets the scheduled time"""
        instructionList = []
        while self.getNextInstructTime() <= curtime:
            instructionList.append(heapq.heappop(self.instructions))

        return instructionList
    
    def __str__(self) -> str:
        msgList = [ins.__str__() for ins in self.instructions]

        return "\n".join(msgList)

if __name__ == "__main__":
    ins1 = [10, "fridge", "open the door"]
    ins2 = [20, "elephant", "be squeezed into the fridge" ]
    ins3 = [30, "fridge", "close the door"]

    scheduler1 = Scheduler()
    scheduler1.addInstruction(ins3[0], ins3[1], ins3[2])
    scheduler1.addInstruction(ins1[0], ins1[1], ins1[2])
    scheduler1.addInstruction(ins2[0], ins2[1], ins2[2])

    print("top instruction @", scheduler1.getNextInstructTime())


    for time in range(40):
        for ins in scheduler1.getInstruction(time):
            print("@", time, ins)

    print("scheduler2")

    insList = ['10', 'serviceRate', '3', '20', 'channelDelay', '100,200', '30', 'channelDelay', '100','nonsense']
    scheduler2 = Scheduler(insList)
    print(scheduler2)
    for time in range(40):
        for ins in scheduler2.getInstruction(time):
            print("@", time, ins)
