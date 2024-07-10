import os

import joblib
import pandas as pd
import tensorflow as tf
import numpy as np

from reservation_analysis.repository.reservation_analysis_repository_impl import ReservationAnalysisRepositoryImpl
from reservation_analysis.service.reservation_analysis_service import ReservationAnalysisService


class ReservationAnalysisServiceImpl(ReservationAnalysisService):
    NUMBER_OF_MODELS = 10

    def __init__(self):
        self.__reservationAnalysisRepository = ReservationAnalysisRepositoryImpl()

    async def readModel(self):
        print(f"service -> readModel()")

        currentDir = os.getcwd()
        # pip install openpyxl
        filepath = os.path.join(currentDir, "..", "assets", "survey_info.xlsx")

        try:
            dataFrame = pd.read_excel(filepath)
            return dataFrame
        except FileNotFoundError:
            print("cannot find Database")

    async def trainModel(self):
        dataFrame = await self.readModel()
        print('dataFrame loaded')
        print(dataFrame)

        X, y = self.__reservationAnalysisRepository.prepareReservationInfo(dataFrame)
        print('scaling complete')
        print(f"X:{X}")
        print(f"y:{y}")

        # print('dumping scaler complete')

        # model = await self.__reservationAnalysisRepository.createModel()
        # model.fit(X_scaled, y)
        # print('model fitting complete')
        # await self.__reservationAnalysisRepository.saveModel(model)
        #
        # return "training sequences complete!"

        modelList = []
        for index in range(self.NUMBER_OF_MODELS):
            print(f"Start train Model{index + 1}/{self.NUMBER_OF_MODELS}")
            model = self.__reservationAnalysisRepository.createModel()
            fittedModel = await self.__reservationAnalysisRepository.fitModel(model, X, y)
            self.__reservationAnalysisRepository.saveModel(fittedModel, f"reservationModel_{index + 1}.h5")
            modelList.append(model)
            print()

        return f"Trained {self.NUMBER_OF_MODELS} models successfully ended!"

    async def predictReservationFromModel(self, len_of_reservation, num_of_adult, num_of_child, is_exist_car):
        print(f"service -> predictReservationFromModel()")
        print()
        print(f"len_of_reservation:{len_of_reservation}")
        print(f"num_of_adult:{num_of_adult}")
        print(f"num_of_child:{num_of_child}")
        print(f"is_exist_car:{is_exist_car}")
        print()

        # model = await self.__reservationAnalysisRepository.loadKmeansModel()
        # reservationPredict = model.predict([[len_of_reservation, num_of_adult, num_of_child, is_exist_car]])
        # print(reservationPredict)
        # # return reservationPredict


        reservationPredictionList = []

        for index in range(1, self.NUMBER_OF_MODELS):
            reservationModel = await self.__reservationAnalysisRepository.loadKmeansModel(
                f"reservationModel_{index + 1}.h5")
            # X_pred = np.array([[len_of_reservation, num_of_adult, num_of_child, is_exist_car]])
            X_pred = np.array([[len_of_reservation]])
            # X_pred_scaler = await self.__reservationAnalysisRepository.transformFromScaler(scaler, X_pred)

            reservationPredict = await self.__reservationAnalysisRepository.predictFromModel(reservationModel,
                                                                                             X_pred)
            reservationPredictionList.append(reservationPredict)
        averagePrediction = np.mean(reservationPredictionList)
        print(f"predicted hotel's product_id: {averagePrediction}")

        return round(averagePrediction)
