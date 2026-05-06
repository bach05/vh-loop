# src/training/train_backend.py

from abc import ABC, abstractmethod


class TrainingBackend(ABC):

    @abstractmethod
    def setup_trainer(self, train_dataset, eval_dataset=None, collator=None, debug=False):
        pass

    @abstractmethod
    def train(self):
        pass