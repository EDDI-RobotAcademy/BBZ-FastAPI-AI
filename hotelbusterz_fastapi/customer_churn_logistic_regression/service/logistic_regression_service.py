from abc import ABC, abstractmethod


class LogisticRegressionService(ABC):
    @abstractmethod
    def trainData(self):
        pass

    @abstractmethod
    def predictChurnPercent(self, X_new):
        pass