import base64
from io import BytesIO

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

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

        print(self.logisticRegressionRepositoryImpl.churnMetric(model, X_test_scaled, y_test))

        joblib.dump(surveyModel, 'surveyModel.joblib')
        joblib.dump(scaler, 'scaler.joblib')

        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.savefig("roc_curve.png")
        plt.close()


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
        print(f"해당 데이터로 분석한 이탈 확률: {churn_percent}")
        churn_percent = round(churn_percent[-1][0] * 100, 2)

        return churn_percent
