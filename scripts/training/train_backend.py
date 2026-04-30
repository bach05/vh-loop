# src/training/train_backend.py

from abc import ABC, abstractmethod


class TrainingBackend(ABC):
    @abstractmethod
    def train(self, train_dataset, eval_dataset=None, collator=None):
        pass