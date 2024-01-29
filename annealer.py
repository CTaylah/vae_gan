class Annealer():
    def __init__(self, start: int, max: int, rate: float=0.01):
        self.start = start
        self.rate = rate
        self.max = max
        self.current = start

    def __call__(self, epoch: int):
        self.current = min(self.start + self.rate * epoch, self.max)
        return self.current