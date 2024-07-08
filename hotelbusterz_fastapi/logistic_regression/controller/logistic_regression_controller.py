from fastapi import APIRouter, Depends
from pydantic import BaseModel

from logistic_regression.service.logistic_regression_service_impl import LogisticRegressionServiceImpl


logisticRegressionRouter = APIRouter()

async def injectLogisticRegressionService() -> LogisticRegressionServiceImpl:
    return LogisticRegressionServiceImpl()

@logisticRegressionRouter.get("/logistic-regression")
async def logisticRegression(logisticRegressionService: LogisticRegressionServiceImpl =
                            Depends(injectLogisticRegressionService)):

    print("survey logistic_regression()")
    logisticRegressionService.trainData()
    print("이탈 예측 모델 학습 완료")


class PredictRequest(BaseModel):
    num_of_adult: int
    num_of_child: int
    have_breakfast: int
    is_exist_car: int
    len_of_reservation: int
@logisticRegressionRouter.post("/churn-predict")

async def churnPredict(X_new: PredictRequest, logisticRegressionService: LogisticRegressionServiceImpl =
                       Depends(injectLogisticRegressionService)):

    churn_percent = logisticRegressionService.predictChurnPercent(X_new)

    return churn_percent