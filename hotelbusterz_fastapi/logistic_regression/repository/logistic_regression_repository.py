from abc import ABC, abstractmethod



class LogisticRegressionRepository(ABC):

    @abstractmethod
    def loadData(self):
        pass

    @abstractmethod
    def splitTrainTestData(self, X_scaled, y):
        pass

    @abstractmethod
    def transformFromScaler(self, dataFrame):
        pass

    @abstractmethod
    def createModel(self):
        pass

    @abstractmethod
    def fitModel(self, model, X_train, y_train):
        pass

    @abstractmethod
    def churnMetric(self, model, X_test):
        pass

    @abstractmethod
    def loadSurveyModel(self,):
        pass

    @abstractmethod
    def loadSurveyModelScaler(self,):
        pass
