from operator import itemgetter
import math

class Recommender:
    def __init__(self, predictor):
        self.__predictor = predictor
        self.__uid = None
        self.__test_data = None

    def fit(self, user_item_data):
        self.__predictor.fit(user_item_data)
        self.__uid = user_item_data

    def recommend(self, user_id, n=10, rec_seen=True):
        prediction = self.__predictor.predict(user_id)

        if not rec_seen:
            prediction = list(filter(lambda x: not self.__has_seen_movie(user_id, x[0]), prediction))

        prediction.sort(key=itemgetter(1), reverse=True)

        return prediction[:n]

    def __has_seen_movie(self, user_id, movie_id):
        df = self.__uid.data[
            (self.__uid.data['userID'] == user_id) &
            (self.__uid.data['movieID'] == movie_id)
        ]
        df = df.dropna()
        return not df.empty

    def evaluate(self, uim, n):
        test_data = uim.data

        mae = self.__calculate_mae(test_data, n)
        rmse = self.__calculate_rmse(test_data, n)
        precision = self.__calculate_precision(test_data, n)
        recall = self.__calculate_recall(test_data, n)
        f1 = self.__calculate_f1(precision, recall)

        return (
            mae,
            rmse,
            precision,
            recall,
            f1
        )

    def __calculate_mae(self, test_data, n):
        users = self.__uid.data['userID'].unique()
        sum_a, sum_b = 0, 0

        for user_id in users:
            recommendations_array = self.recommend(user_id, n, False)
            recommended_movie_ids = [i[0] for i in recommendations_array]

            user_rated_test_set_items = test_data[
                (test_data['userID'] == user_id) &
                (test_data['movieID'].isin(recommended_movie_ids))
                ]

            user_recommendations = dict(recommendations_array)

            for _, test_set_item in user_rated_test_set_items.iterrows():
                test_set_rating = test_set_item['rating'].item()

                movie_id = test_set_item['movieID'].item()
                recommended_rating = user_recommendations[movie_id]

                deviation = abs(recommended_rating - test_set_rating)

                sum_a += deviation

            test_set_size = len(user_rated_test_set_items)
            sum_b += test_set_size

        return sum_a / sum_b

    def __calculate_rmse(self, test_data, n):
        users = self.__uid.data['userID'].unique()
        sum_a, sum_b = 0, 0

        for user_id in users:
            recommendations_array = self.recommend(user_id, n, False)
            recommended_movie_ids = [i[0] for i in recommendations_array]

            user_rated_test_set_items = test_data[
                (test_data['userID'] == user_id) &
                (test_data['movieID'].isin(recommended_movie_ids))
                ]

            user_recommendations = dict(recommendations_array)

            for _, test_set_item in user_rated_test_set_items.iterrows():
                test_set_rating = test_set_item['rating'].item()

                movie_id = test_set_item['movieID'].item()
                recommended_rating = user_recommendations[movie_id]

                deviation = (recommended_rating - test_set_rating) ** 2

                sum_a += deviation

            test_set_size = len(user_rated_test_set_items)
            sum_b += test_set_size

        return math.sqrt(sum_a / sum_b)

    def __calculate_precision(self, test_data, n):
        users = self.__uid.data['userID'].unique()

        sum_hits, sum_test_set = 0, 0

        for user_id in users:
            user_recommendations_array = self.recommend(user_id, n, True)
            user_recommendations = dict(user_recommendations_array)

            user_ratings = self.__get_user_ratings(test_data, user_id)

            movie_ids = user_ratings['movieID'].unique()

            sum_hits += sum(x in user_recommendations for x in movie_ids)
            sum_test_set += len(user_recommendations_array)

        return sum_hits / sum_test_set

    def __calculate_recall(self, test_data, n):
        users = self.__uid.data['userID'].unique()

        sum_hits, sum_test_set = 0, 0

        for user_id in users:
            user_recommendations = dict(self.recommend(user_id, n, True))

            user_ratings = self.__get_user_ratings(test_data, user_id)

            movie_ids = user_ratings['movieID'].unique()

            sum_hits += sum(x in user_recommendations for x in movie_ids)
            sum_test_set += len(test_data)

        return sum_hits / sum_test_set

    def __calculate_f1(self, P, R):
        return (2 * P * R) / (P + R)

    def __get_user_ratings(self, test_data, user_id):
        user_ratings = test_data[test_data['userID'] == user_id]
        average_user_rating = user_ratings['rating'].mean()

        return user_ratings[user_ratings['rating'] >= average_user_rating]
