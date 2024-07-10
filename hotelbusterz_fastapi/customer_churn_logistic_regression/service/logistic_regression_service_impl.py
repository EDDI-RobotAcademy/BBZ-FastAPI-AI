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

        # 분석모델 성능 출력
        print(self.logisticRegressionRepositoryImpl.churnMetric(model, X_test_scaled, y_test))
        joblib.dump(surveyModel, 'surveyModel.joblib')
        joblib.dump(scaler, 'scaler.joblib')

    def predictChurnPercent(self, X_new):
        loadedSurveyModel = self.logisticRegressionRepositoryImpl.loadSurveyModel()
        loadedSurveyModelScaler = self.logisticRegressionRepositoryImpl.loadSurveyModelScaler()

        print('new X: ',X_new)
        # 포스트맨으로 보낼 때 이슈 처리 부분, 웹 입력은 다를 수도 있음
        # 똑같이 키-값으로 전달되네.... 값만 전달 못시키나
        X_newNum = [X_new.feature1,
                    X_new.feature2,
                    X_new.feature3,
                    X_new.feature4,
                    X_new.feature5,
                    X_new.feature6,
                    X_new.feature7,
                    X_new.feature8]
        print('처리된: ',X_newNum)
        X_newNum =  np.array(X_newNum)
        X_new_scaled = loadedSurveyModelScaler.transform(X_newNum.reshape(1, -1))
        churn_percent = loadedSurveyModel.predict_proba(X_new_scaled)
        # print(f"해당 데이터로 분석한 이탈 확률: {churn_percent}")
        churn_percent = round(churn_percent[-1][0] * 100, 2)

        return churn_percent
