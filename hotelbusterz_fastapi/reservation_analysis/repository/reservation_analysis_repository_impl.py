import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hotelbusterz_fastapi.reservation_analysis.repository.reservation_analysis_repository import \
    ReservationAnalysisRepository


class ReservationAnalysisRepositoryImpl(ReservationAnalysisRepository):
    async def prepareReservationInfo(self, dataFrame):
        print(dataFrame[['len_of_reservation', 'num_of_adult', 'num_of_child', 'is_exist_car']].shape)
        X = dataFrame[['len_of_reservation', 'num_of_adult', 'num_of_child', 'is_exist_car']].values.reshape(100000, 4)
        y = dataFrame['product_id'].values
        print(X)
        print("data setting complete... starting scaler")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler

    async def splitTrainTestData(self, X_scaled, y):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=37)
        return X_train, X_test, y_train, y_test

    async def createModel(self):
        # n_init은 중심위치 시도 횟수(default = 10)
        kmeans = KMeans(n_clusters=5, n_init=10)
        return kmeans

    async def fitModel(self, model, X_train, y_train, epochs, validation_split, batch_size, verbose):
        model.fit(X_train, y_train, epochs, epochs=epochs, validation_split=validation_split, batch_size=batch_size,
                  verbose=verbose)

    async def saveModel(self, model):
        joblib.dump(model, 'kmeans_reservation.joblib')
        print('model saving complete')

    async def transformFromScaler(self, scaler, X_pred):
        return scaler.transform(X_pred)

    async def loadKmeansModel(self):
        return joblib.load("kmeans_reservation.joblib")

    async def predictFromModel(self, reservationModel, X_pred_scaler):
        return reservationModel.predict(X_pred_scaler).flatten()[0]
