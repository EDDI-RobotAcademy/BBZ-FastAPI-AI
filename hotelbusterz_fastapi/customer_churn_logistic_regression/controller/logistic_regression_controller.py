from fastapi import APIRouter, Depends
from pydantic import BaseModel

from customer_churn_logistic_regression.service.logistic_regression_service_impl import LogisticRegressionServiceImpl

logisticRegressionRouter = APIRouter()


async def injectLogisticRegressionService() -> LogisticRegressionServiceImpl:
    return LogisticRegressionServiceImpl()


@logisticRegressionRouter.get("/logistic-regression")
async def logisticRegression(logisticRegressionService: LogisticRegressionServiceImpl =
                             Depends(injectLogisticRegressionService)):
    print("survey customer_churn_logistic_regression()")
    base64_image = logisticRegressionService.trainData()
    print("이탈 예측 모델 학습 완료")

    return base64_image

class PredictRequest(BaseModel):
    feature1: int
    feature2: int
    feature3: int
    feature4: int
    feature5: int
    feature6: int
    feature7: int
    feature8: int


@logisticRegressionRouter.post("/churn-predict")
async def churnPredict(X_new: PredictRequest, logisticRegressionService: LogisticRegressionServiceImpl =
Depends(injectLogisticRegressionService)):
    churn_percent = logisticRegressionService.predictChurnPercent(X_new)
    return churn_percent
