
class Annealer():
    def __init__(self, starting_value: float, max: float, rate: float, start_epoch: int = 0):
        self.start_value = starting_value
        self.rate = rate
        self.max = max
        self.current = starting_value
        self.start_epoch = start_epoch
        self.stop = False

    def __call__(self, epoch: int):
        if self.stop:
            return 0;
        if epoch < self.start_epoch:
            return self.start_value
        self.current = min(self.start_value + self.rate * (epoch - self.start_epoch), self.max)
        return self.current
