import random


class BaseRNG(object):

    def __init__(self):
        pass

    def nextNum(self):
        raise NotImplementedError


class RangeUniform(BaseRNG):

    def __init__(self, a: int, b: int) -> None:
        super(RangeUniform, self).__init__()

        if a < b:
            self.low, self.high = a, b
        else:
            self.low, self.high = b, a

    def nextNum(self) -> int:
        return random.randrange(self.low, self.high, 1)


if __name__ == "__main__":
    rangeUniform = RangeUniform(3, 10)
    nlist = [rangeUniform.nextNum() for _ in range(20)]
    print(nlist)

