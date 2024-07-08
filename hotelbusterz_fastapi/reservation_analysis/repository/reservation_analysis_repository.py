from abc import ABC, abstractmethod


class ReservationAnalysisRepository(ABC):
    @abstractmethod
    def prepareReservationInfo(self, dataFrame):
        pass

    @abstractmethod
    def splitTrainTestData(self, X_scaled, y):
        pass

    @abstractmethod
    def createModel(self):
        pass

    @abstractmethod
    def fitModel(self, model, X_train, y_train, epochs, validation_split, batch_size, verbose):
        pass

    @abstractmethod
    def saveModel(self, model):
        pass

    @abstractmethod
    def transformFromScaler(self, scaler, X_pred):
        pass
    @abstractmethod
    def loadKmeansModel(self):
        pass

    @abstractmethod
    def predictFromModel(self, reservationModel, X_pred_scaler):
        pass
