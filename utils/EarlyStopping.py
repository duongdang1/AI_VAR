"""
    Monitor the validation loss during the training of VARCNN
    Stop the training process early if the validation loss stops
    improving:
    => preventing overfitting and potentially saving computational resources
"""
class EarlyStopping: 
    def __init__(self,patience = 10, delta = 0, verbose = False):
        self.patience = patience # number of epochs with no improvement
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    """
        This method is invoked each time the validation loss is evaluated 
        during training.
    """
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        return self.early_stop