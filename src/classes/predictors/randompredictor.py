from src.classes.utils import *
import random


class RandomPredictor:
    def __init__(self, min, max):
        self.__min = min
        self.__max = max
        self.__user_item_data = None

    def fit(self, user_item_data):
        self.__user_item_data = user_item_data

    def predict(self, _):
        movie_ids = self.__user_item_data.data_frame.groupby('movieID', as_index=False)

        return [(id, random.randint(self.__min, self.__max)) for id in list(movie_ids.groups)]