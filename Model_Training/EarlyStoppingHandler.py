import copy
import os
from datetime import datetime

import torch


class EarlyStoppingHandler:

    def __init__(self, patience):
        self.best_val_loss = 1000
        self.counter = 0
        self.best_model = None
        self.patience = patience

    def handle_early_stopping(self, model, val_loss, epoch, num_epochs, model_path):
        if val_loss < self.best_val_loss:  # Early stopping
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            print(f"Early stopping in {self.patience - self.counter} epochs")
            if self.counter >= self.patience:
                print("Early stopping triggered. No improvement in validation loss.")
                # save the model
                torch.save(self.best_model.state_dict(),
                           os.path.join(model_path,
                                        f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_epoche_{epoch}_of_{num_epochs}_best_model.pth"))
                return True
        return False

    def get_best_model(self):
        return self.best_model
