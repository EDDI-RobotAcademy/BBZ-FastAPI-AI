import os

import joblib
import pandas as pd
import tensorflow as tf
import numpy as np

from hotelbusterz_fastapi.reservation_analysis.repository.reservation_analysis_repository_impl import ReservationAnalysisRepositoryImpl
from hotelbusterz_fastapi.reservation_analysis.service.reservation_analysis_service import ReservationAnalysisService


class ReservationAnalysisServiceImpl(ReservationAnalysisService):
    NUMBER_OF_MODELS = 10

    def __init__(self):
        self.__reservationAnalysisRepository = ReservationAnalysisRepositoryImpl()

    async def readModel(self):
        print(f"service -> readModel()")

        currentDir = os.getcwd()
        filepath = os.path.join(currentDir, "..", "assets", "reservation_info.xlsx")

        try:
            dataFrame = pd.read_excel(filepath)
            return dataFrame
        except FileNotFoundError:
            print("cannot find Database")

    async def trainModel(self):
        dataFrame = await self.readModel()
        X_scaled, y, scaler = await self.__reservationAnalysisRepository.prepareReservationInfo(dataFrame)
        joblib.dump(scaler, "reservationModelScaler.pkl")

        X_train, X_test, y_train, y_test = await self.__reservationAnalysisRepository.splitTrainTestData(X_scaled, y)

        modelList = []
        for index in range(self.NUMBER_OF_MODELS):
            print(f"Start train Model{index + 1}/{self.NUMBER_OF_MODELS}")

            model = await self.__reservationAnalysisRepository.createModel()
            await self.__reservationAnalysisRepository.fitModel(model, X_train, y_train, epochs=100,
                                                                validation_split=0.2, batch_size=64, verbose=1)
            model.save(f"reservationModel_{index + 1}.h5")
            modelList.append(model)

        return f"Trained {self.NUMBER_OF_MODELS} models successfully ended!"

    async def predictReservationFromModel(self, len_of_reservation, num_of_adult, num_of_child, is_exist_car):
        print(f"service -> predictReservationFromModel()")
        print()
        print(f"len_of_reservation:{len_of_reservation}")
        print(f"num_of_adult:{num_of_adult}")
        print(f"num_of_child:{num_of_child}")
        print(f"is_exist_car:{is_exist_car}")
        print()

        scaler = joblib.load('reservationModelScaler.pkl')

        reservationPredictionList = []

        for index in range(1, self.NUMBER_OF_MODELS + 1):
            reservationModel = tf.keras.models.load_model(f"reservationModel_{index + 1}.h5")
            X_pred = np.array([[len_of_reservation, num_of_adult, num_of_child, is_exist_car]])
            X_pred_scaler = await self.__reservationAnalysisRepository.transformFromScaler(scaler, X_pred)

            reservationPredict = await self.__reservationAnalysisRepository.predictFromModel(reservationModel,
                                                                                             X_pred_scaler)
            reservationPredictionList.append(reservationPredict)
        averagePrediction = np.mean(reservationPredictionList)
        print(f"predicted hotel's product_id: {averagePrediction}")

        return averagePrediction
