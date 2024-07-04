from fastapi import APIRouter, Depends

from logistic_regression.service.logistic_regression_service_impl import LogisticRegressionServiceImpl


logisticRegressionRouter = APIRouter()

async def injectLogisticRegressionService() -> LogisticRegressionServiceImpl:
    return LogisticRegressionServiceImpl()

@logisticRegressionRouter.get("/logistic-regression")
async def logistic_regression_test():
    print("logistic_regression_test()")