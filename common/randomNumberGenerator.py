import random

class BaseRNG(object):

    def __init__(self):
        pass

    def nextNum(self):
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError

class RangeUniform(BaseRNG):

    def __init__(self, a: int, b: int) -> None:
        super(RangeUniform, self).__init__()

        if a < b:
            self.low, self.high = a, b
        else:
            self.low, self.high = b, a

    def nextNum(self) -> int:
        return random.randint(self.low, self.high)
    
    def __str__(self) -> str:
        s = "Random Integer Generator in [{a}, {b}]".format(a=self.low, b=self.high)
        return s


if __name__ == "__main__":
    rangeUniform = RangeUniform(3, 10)
    print(rangeUniform)
    nlist = [rangeUniform.nextNum() for _ in range(20)]
    print(nlist)

