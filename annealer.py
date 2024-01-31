
class Annealer():
    def __init__(self, starting_value: int, max: int, rate: float, start_epoch: int = 0):
        self.start_value = starting_value
        self.rate = rate
        self.max = max
        self.current = starting_value
        self.start_epoch = start_epoch

    def __call__(self, epoch: int):
        if epoch < self.start_epoch:
            return self.start_value
        self.current = min(self.start_value + self.rate * (epoch - self.start_epoch), self.max)
        return self.current