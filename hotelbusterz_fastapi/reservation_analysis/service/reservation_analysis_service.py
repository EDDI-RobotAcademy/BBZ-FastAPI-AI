from abc import ABC, abstractmethod


class ReservationAnalysisService(ABC):
    @abstractmethod
    def trainModel(self):
        pass
    @abstractmethod
    def predictReservationFromModel(self, len_of_reservation, num_of_adult, num_of_child, is_exist_car):
        pass
