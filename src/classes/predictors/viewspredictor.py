class ViewsPredictor:
    def __init__(self):
        self.__uid = None
        self.__prediction = None

    def fit(self, uid):
        self.__uid = uid

        movies_grouped = uid.data.groupby('movieID')
        review_count_by_movie = movies_grouped.size()

        self.__prediction = [(mid, review_count_by_movie[mid]) for mid in list(movies_grouped.groups)]

    def predict(self, _):
        return self.__prediction

