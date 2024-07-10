import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reservation_analysis.repository.reservation_analysis_repository import ReservationAnalysisRepository


class ReservationAnalysisRepositoryImpl(ReservationAnalysisRepository):
    NUM_OF_POINTS = 100000
    NUM_OF_FEATURES = 4

    def prepareReservationInfo(self, dataFrame):
        print(dataFrame[['len_of_reservation', 'num_of_adult', 'num_of_child', 'is_exist_car']].shape)
        # X = dataFrame[['len_of_reservation', 'num_of_adult', 'num_of_child', 'is_exist_car']].values.reshape(100000, 4)
        X = dataFrame[['len_of_reservation']].values.reshape(100000, 1)
        y = dataFrame['product_id'].values
        # y = tf.one_hot(dataFrame['product_id'], depth=5).numpy()
        print(X)
        print("data setting complete... starting scaler")
        return X, y

    async def splitTrainTestData(self, X_scaled, y):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=37)
        return X_train, X_test, y_train, y_test

    def createModel(self):
        # n_init은 중심위치 시도 횟수(default = 10)
        # model = KMeans(n_clusters=5, n_init=10)

        model = LogisticRegression(max_iter=10000)
        return model

    async def fitModel(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def saveModel(self, model, fileName):
        joblib.dump(model, fileName)
        print('model saving complete')

    async def transformFromScaler(self, scaler, X_pred):
        return scaler.transform(X_pred)

    async def loadKmeansModel(self, modelPath):
        return joblib.load(modelPath)

    async def predictFromModel(self, reservationModel, X_pred_scaler):
        return reservationModel.predict(X_pred_scaler).flatten()[0]
