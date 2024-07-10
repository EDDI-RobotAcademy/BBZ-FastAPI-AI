import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd

from customer_churn_logistic_regression.repository.logistic_regression_repository import LogisticRegressionRepository


class LogisticRegressionRepositoryImpl(LogisticRegressionRepository):
    def loadData(self):
        currentDirectory = os.getcwd()
        print(f"currentDirectory: {currentDirectory}")

        filePath = os.path.join(
            currentDirectory, "..", "assets", "log_data_pivot.csv")

        try:
            dataFrame = pd.read_csv(filePath)
            X = dataFrame
            # print(X)
            y = (dataFrame['ORDER'] >=1).astype(int)

            print('데이터 로드 완료')
            return X, y
        except FileNotFoundError:
            print(f"파일이 존재하지 않습니다: {filePath}")

    def splitTrainTestData(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        print('데이터 분할 완료')
        return X_train, X_test, y_train, y_test

    def transformFromScaler(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print('데이터 스케일링 완료')
        return X_train_scaled, X_test_scaled, scaler

    def createModel(self):
        clf = LogisticRegression()
        # 하이퍼파라미터 여기서 튜닝하시면 됩니다 -> 뭔가문제생겨서 튜닝제거
        print('모델 생성 완료')
        return clf

    def fitModel(self, model, X_train_scaled, y_train):
        surveyModel = model.fit(X_train_scaled, y_train)
        print('모델 학습 완료')
        return surveyModel

    def churnMetric(self, surveyModel, X_test_scaled, y_test):
        y_pred = surveyModel.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))
        # 모델의 성능을 보는 부분입니다

    def loadSurveyModel(self,):
        print('학습된 모델 로드 완료')
        return joblib.load('surveyModel.joblib')

    def loadSurveyModelScaler(self,):
        print('학습된 X스케일러 로드 완료')
        return joblib.load('scaler.joblib')

