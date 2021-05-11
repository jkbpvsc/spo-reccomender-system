class AveragePredictor:
    def __init__(self, b):
        self.__b = b
        self.__uid = None
        self.__predictions = None

    def fit(self, uid):
        self.__uid = uid
        self.__calc_averages()

    def __calc_averages(self):
        self.grouped_by_movies = self.__uid.data.groupby('movieID')
        self.n_review_per_movie = self.grouped_by_movies.size()
        self.g_avg = self.__uid.data['rating'].mean()
        self.g_movie_rating_sum = self.grouped_by_movies['rating'].sum()

        self.__predictions = [(mid, self.__calc_average_for_movie(mid)) for mid in list(self.grouped_by_movies.groups)]

    def __calc_average_for_movie(self, mid):
        return (self.g_movie_rating_sum[mid] + self.__b * self.g_avg) / (self.n_review_per_movie[mid] + self.__b)

    def predict(self, _):
        return self.__predictions




