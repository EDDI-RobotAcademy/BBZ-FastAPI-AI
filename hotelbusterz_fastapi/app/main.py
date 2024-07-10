import os

import aiomysql
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from async_db.database import getMySqlPool
from customer_churn_logistic_regression.controller.logistic_regression_controller import logisticRegressionRouter

import warnings

from reservation_analysis.controller.reservation_analysis_controller import reservationAnalysisRouter

warnings.filterwarnings("ignore", category=aiomysql.Warning)


async def lifespan(app: FastAPI):
    # Startup
    app.state.dbPool = await getMySqlPool()

    yield

    # Shutdown
    app.state.dbPool.close()
    await app.state.dbPool.wait_closed()


app = FastAPI(lifespan=lifespan)
load_dotenv()

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.connections = set()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(reservationAnalysisRouter)
app.include_router(logisticRegressionRouter)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.41", port=33333)