class STDPredictor:
    def __init__(self, n_ratings=0):
        self.__n_ratings = n_ratings
        self.__prediction = None
        self.__uid = None

    def fit(self, uid):
        self.__uid = uid

        g_movies = uid.data.groupby('movieID')
        g_movies_rcount = g_movies.size()

        self.__prediction = [(mid, self.__calc_std_for_movie(mid)) for mid in list(g_movies.groups) if g_movies_rcount[mid] > self.__n_ratings]

    def __calc_std_for_movie(self, mid):
        return self.__uid.data[self.__uid.data['movieID'] == mid]['rating'].std()

    def predict(self, _):
        return self.__prediction

