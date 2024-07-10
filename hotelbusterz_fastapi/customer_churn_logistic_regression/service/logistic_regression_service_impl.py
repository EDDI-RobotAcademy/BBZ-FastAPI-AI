import joblib
import numpy as np

from customer_churn_logistic_regression.repository.logistic_regression_repository_impl import \
    LogisticRegressionRepositoryImpl
from customer_churn_logistic_regression.service.logistic_regression_service import LogisticRegressionService


class LogisticRegressionServiceImpl(LogisticRegressionService):
    def __init__(self):
        self.logisticRegressionRepositoryImpl = LogisticRegressionRepositoryImpl()

    def trainData(self):
        X, y =self.logisticRegressionRepositoryImpl.loadData()

        X_train, X_test, y_train, y_test = (
            self.logisticRegressionRepositoryImpl.splitTrainTestData(X, y))

        X_train_scaled, X_test_scaled, scaler = (
            self.logisticRegressionRepositoryImpl.transformFromScaler(X_train, X_test))

        model = self.logisticRegressionRepositoryImpl.createModel()

        surveyModel = self.logisticRegressionRepositoryImpl.fitModel(
                model, X_train_scaled, y_train)

        # self.logisticRegressionRepositoryImpl.churnMetric()
        # 분석모델 성능 출력

        joblib.dump(surveyModel, 'surveyModel.joblib')
        joblib.dump(scaler, 'scaler.joblib')

    def predictChurnPercent(self, X_new):
        loadedSurveyModel = self.logisticRegressionRepositoryImpl.loadSurveyModel()
        loadedSurveyModelScaler = self.logisticRegressionRepositoryImpl.loadSurveyModelScaler()

        print(X_new)
        # 포스트맨으로 보낼 때 이슈 처리 부분, 웹 입력은 다를 수도 있음
        X_newNum = [X_new.num_of_adult, X_new.num_of_child, X_new.have_breakfast, X_new.is_exist_car, X_new.len_of_reservation]
        print(X_newNum)
        X_newNum =  np.array(X_newNum)
        X_new_scaled = loadedSurveyModelScaler.transform(X_newNum.reshape(1, -1))
        churn_percent = loadedSurveyModel.predict_proba(X_new_scaled)
        print(f"해당 데이터로 분석한 이탈 확률: {churn_percent}")
        return churn_percent
