class EarlyStoppingCriterion:
    def __init__(self, epochs=3, min_diff=0.01):
        super().__init__()
        self.epochs = epochs
        self.min_diff = min_diff
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, history):
        val_loss = history['val'][-1]
        if val_loss < self.best_loss - self.min_diff:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.epochs
