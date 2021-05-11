import pandas as pd


class SlopeOnePredictor:
    def __init__(self):
        self.__uid = None
        self.__prediction = None
        self.__diff_eval_cache = {}

    def fit(self, uid):
        self.__uid = uid

    def predict(self, user_id):
        movies = self.__uid.data['movieID'].unique()

        return [(mid, self.__get_movie_prediction(user_id, mid)) for mid in movies]

    def __get_movie_prediction(self, user_id, mid):

        co_rated_movies = list(self.__uid.data[self.__uid.data['userID'] == user_id]['movieID'].unique())
        if mid in co_rated_movies:
            co_rated_movies.remove(mid)

        a, b = 0, 0
        for movie_id in co_rated_movies:
            deviation, size = self.__dif_evaluations(mid, movie_id)
            user_rating = self.__get_user_rating_for_movie(user_id, movie_id)
            a += (deviation + user_rating) * size
            b += size

        if b == 0:
            return 0

        return a / b

    def get_movie_prediction(self, user_id, mid):
        return self.__get_user_rating_for_movie(user_id, mid)

    def __get_user_rating_for_movie(self, user_id, mid):
        return self.__uid.data[
            (self.__uid.data['userID'] == user_id) &
            (self.__uid.data['movieID'] == mid)
            ]['rating'].item()

    def __dif_evaluations(self, mid_1, mid_2):
        if (mid_1, mid_2) not in self.__diff_eval_cache:
            self.__lazy_load_diff_evaluation(mid_1, mid_2)

        return self.__diff_eval_cache[(mid_1, mid_2)]

    def __lazy_load_diff_evaluation(self, mid_1, mid_2):
        if (mid_2, mid_1) in self.__diff_eval_cache:
            self.__lazy_load_inverted_diff(mid_1, mid_2)
        else:
            m1_ratings = self.__get_movie_ratings(mid_1)[['userID', 'rating']]
            m2_ratings = self.__get_movie_ratings(mid_2)[['userID', 'rating']]

            ratings_intersection = pd.merge(m1_ratings, m2_ratings, how='inner', on='userID')
            n_ratings = len(ratings_intersection)

            diff = sum([(x['rating_x'] - x['rating_y']) / n_ratings for _, x in ratings_intersection.iterrows()])

            self.__diff_eval_cache[(mid_1, mid_2)] = diff, n_ratings

    def __lazy_load_inverted_diff(self, mid_1, mid_2):
        cached_value = self.__diff_eval_cache[(mid_2, mid_1)]
        inv_value = -1 * cached_value[0]
        size = cached_value[1]
        self.__diff_eval_cache[(mid_1, mid_2)] = (inv_value, size)

    def __get_movie_ratings(self, mid):
        return self.__uid.data[self.__uid.data['movieID'] == mid]
