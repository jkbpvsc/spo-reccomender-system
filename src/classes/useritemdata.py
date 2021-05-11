from src.classes.utils import read_dat_file
import pandas as pd

class UserItemData:
    def __init__(self, path, from_date=None, to_date=None, min_ratings=None):
        self.data = read_dat_file(path)

        if from_date:
            self.filter_from_date(from_date)

        if to_date:
            self.filter_to_date(to_date)

        if min_ratings:
            self.filter_min_ratings(min_ratings)

    def nratings(self):
        return len(self.data)

    def filter_from_date(self, from_date):
        year = from_date.year
        month = from_date.month
        day = from_date.day

        self.data = self.data[
            (self.data['date_year'] > year) |
            ((self.data['date_year'] == year) & (self.data['date_month'] > month)) |
            ((self.data['date_year'] == year) & (self.data['date_month'] == month) & (self.data['date_day'] >= day))
        ]

    def filter_to_date(self, to_date):
        year = to_date.year
        month = to_date.month
        day = to_date.day

        self.data = self.data[
            (self.data['date_year'] < year) |
            ((self.data['date_year'] == year) & (self.data['date_month'] < month)) |
            ((self.data['date_year'] == year) & (self.data['date_month'] == month) & (self.data['date_day'] < day))
        ]

    def filter_min_ratings(self, min_ratings):
        g_rating_c = self.data.groupby('movieID').size().rename('count').to_frame()
        g_rating_filtered = g_rating_c[g_rating_c['count'] > min_ratings]

        self.data = pd.merge(self.data, g_rating_filtered, how='inner', on='movieID')
