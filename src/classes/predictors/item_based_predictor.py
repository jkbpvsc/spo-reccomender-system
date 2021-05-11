from math import sqrt
from operator import itemgetter

import pandas as pd


class ItemBasedPredictor:
    def __init__(self, min_values=0, threshold=0):
        self.__min_values = min_values
        self.__threshold = threshold
        self.__uid = None
        self.__prediction = None
        self.__g_avg_ratings = None
        self.__similarity_map = None

    def fit(self, uid):
        self.__uid = uid
        self.__load_g_avg_rating()
        self.__generate_adjusted_ratings_table()

    def __generate_adjusted_ratings_table(self):
        df = self.__uid.data
        df['a_rating'] = df.apply(self.__compute_adjusted_rating, axis=1)

    def __compute_adjusted_rating(self, row):
        average_user_rating = self.__get_avg_user_rating(row['userID'])
        return row['rating'] - average_user_rating

    def similarity(self, mid_1, mid_2):
        movie_1_reviews = self.__get_m_ratings(mid_1)
        movie_2_reviews = self.__get_m_ratings(mid_2)

        merged_reviews = pd.merge(movie_1_reviews, movie_2_reviews, how='inner', on=['userID'])

        if len(merged_reviews) <= self.__min_values:
            return 0

        scalar_mul = 0
        abs_m1_vector = 0
        abs_m2_vector = 0

        for _, row in merged_reviews.iterrows():

            adj_m1_rating = row['a_rating_x']
            adj_m2_rating = row['a_rating_y']

            scalar_mul += adj_m1_rating * adj_m2_rating

            abs_m1_vector += pow(adj_m1_rating, 2)
            abs_m2_vector += pow(adj_m2_rating, 2)

        similarity_score = (
                scalar_mul /
                (
                        sqrt(abs_m1_vector) *
                        sqrt(abs_m2_vector)
                )
        )

        if similarity_score <= self.__threshold:
            return 0

        return similarity_score

    def __get_avg_user_rating(self, user_id):
        if not self.__g_avg_ratings:
            self.__load_g_avg_rating()

        return self.__g_avg_ratings[user_id]

    def __get_m_ratings(self, mid):
        return self.__uid.data[self.__uid.data['movieID'] == mid][['userID', 'a_rating']]

    def __load_g_avg_rating(self):
        user_ids = self.__uid.data['userID'].unique()
        self.__g_avg_ratings = {}
        g_uid_rating = self.__uid.data[['userID', 'rating']].groupby('userID')
        for uid in user_ids:
            self.__g_avg_ratings[uid] = g_uid_rating.get_group(uid).mean()['rating']

    def predict(self, user_id):
        user_rated_items = self.__uid.data[self.__uid.data['userID'] == user_id][['movieID', 'rating']]
        movieIDs = self.__uid.data['movieID'].unique()

        out = []
        for mid in movieIDs:
            out.append((
                mid,
                self.__get_prediction_for_product(mid, user_rated_items)
            ))

        return out

    def __get_prediction_for_product(self, movie_id, user_rated_items):
        sa, sb = 0, 0
        for _, row in user_rated_items.iterrows():
            similarity_score = self.similarity(movie_id, row['movieID'])
            sa += similarity_score * row['rating']
            sb += similarity_score

        return sa / sb

    def similar_items(self, mid, n):
        movies = self.__uid.data['movieID'].unique()

        res = [(movie, self.similarity(mid, movie)) for movie in movies if movie != mid ]
        res.sort(key=itemgetter(1), reverse=True)

        return res[:n]
