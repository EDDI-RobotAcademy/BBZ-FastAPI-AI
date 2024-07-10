from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import JSONResponse

from reservation_analysis.controller.request_form.reservation_request_form import ReservationRequestForm
from reservation_analysis.service.reservation_analysis_service_impl import ReservationAnalysisServiceImpl

reservationAnalysisRouter = APIRouter()


async def injectReservationAnalysisService() -> ReservationAnalysisServiceImpl:
    return ReservationAnalysisServiceImpl()


@reservationAnalysisRouter.get("/reservation-train")
async def reservationTrain(
        reservationAnalysisService: ReservationAnalysisServiceImpl = Depends(injectReservationAnalysisService)):
    try:
        print(f"controller -> reservationTrain()")
        result = await reservationAnalysisService.trainModel()
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@reservationAnalysisRouter.post("/reservation-predict")
async def reservationPredict(request: ReservationRequestForm,
                             reservationAnalysisService: ReservationAnalysisServiceImpl = Depends(
                                 injectReservationAnalysisService)):
    print(f"controller -> reservationPredict()")
    try:
        result = await reservationAnalysisService.predictReservationFromModel(request.len_of_reservation,
                                                                              request.num_of_adult,
                                                                              request.num_of_child,
                                                                              request.is_exist_car)
        print("result:", result)
        # result = result.tolist()
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
