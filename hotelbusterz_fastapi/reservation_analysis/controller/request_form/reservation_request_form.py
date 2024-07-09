from pydantic import BaseModel


class ReservationRequestForm(BaseModel):
    len_of_reservation: int
    num_of_adult: int
    num_of_child: int
    is_exist_car: int
