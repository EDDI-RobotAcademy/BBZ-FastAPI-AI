import aiomysql
from fastapi import FastAPI

# from async_db.database import getMySqlPool
# from logistic_regression.controller.logistic_regression_controller import logisticRegressionRouter


import warnings

from hotelbusterz_fastapi.async_db.database import getMySqlPool
from hotelbusterz_fastapi.reservation_analysis.controller.reservation_analysis_controller import \
    reservationAnalysisRouter

warnings.filterwarnings("ignore", category=aiomysql.Warning)

async def lifespan(app: FastAPI):
    # Startup
    app.state.dbPool = await getMySqlPool()

    yield

    # Shutdown
    app.state.dbPool.close()
    await app.state.dbPool.wait_closed()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# app.include_router(logisticRegressionRouter)
app.include_router(reservationAnalysisRouter)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=33333)