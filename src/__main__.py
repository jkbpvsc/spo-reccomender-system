import datetime as dt
from operator import itemgetter

from src.classes.predictors.averagepredictor import AveragePredictor
from src.classes.predictors.item_based_predictor import ItemBasedPredictor
from src.classes.predictors.randompredictor import RandomPredictor
from src.classes.predictors.slope_one_predictor import SlopeOnePredictor
from src.classes.predictors.std_predictor import STDPredictor
from src.classes.recommender import Recommender
from src.classes.useritemdata import *
from src.classes.moviedata import MovieData
from src.classes.predictors.viewspredictor import ViewsPredictor


def make_date(row):
    return dt.datetime(
        year=int(row['date_year']),
        month=int(row['date_month']),
        day=int(row['date_day']),
        hour=int(row['date_hour']),
        minute=int(row['date_minute']),
        second=int(row['date_second'])
    )


def filter_to_date(df, from_date):
    return df[df['date'] > from_date]


def task_1():
    uid = UserItemData('data/user_ratedmovies.dat')
    print(uid.nratings())

    uim = UserItemData('data/user_ratedmovies.dat', from_date=dt.datetime(2007, 1, 12), to_date=dt.datetime(2008, 2, 16), min_ratings=100)
    print(uim.nratings())

def task_2():
    md = MovieData('data/movies.dat')
    print(md.get_title(1))

def task_3():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rp.fit(uim)
    pred = rp.predict(78)
    print(type(pred))
    items = [1, 3, 20, 50, 100]
    for item in items:
        print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))

def task_4():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_5():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = AveragePredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_6():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = ViewsPredictor()
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_7():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = STDPredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_8():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    # uim = UserItemData('data/user_ratedmovies.dat')
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    # print(uim.data)
    print("Similarity between the movies 'Men in black'(1580) and 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
    print("Similarity between the movies 'Men in black'(1580) and 'Schindler's List'(527): ", rp.similarity(1580, 527))
    print("Similarity between the movies 'Men in black'(1580) and 'Independence day'(780): ", rp.similarity(1580, 780))

def task_9():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    # uim = UserItemData('data/user_ratedmovies.dat')
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print("Predictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))


def task_10():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    movies = uim.data['movieID'].unique()
    movies.sort()

    comparisons = []

    for m1 in movies:
        for m2 in movies:
            if not m1 < m2:
                continue

            sim = rp.similarity(m1, m2)
            comparisons.append((m1, m2, sim))
            print(m1, m2, sim)

    comparisons.sort(key=itemgetter(2), reverse=True)
    for result in comparisons[:21]:
        print(
            "Movie1: {}, Movie2: {}, similarity: {}".format(
                md.get_title(result[0]),
                md.get_title(result[1]),
                result[2]
            )
        )

def task_11():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rp.similar_items(7361, 30)
    print('Movies similar to "The Lord of the Rings: The Fellowship of the Ring": ')
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}, id: {}".format(md.get_title(idmovie), val, idmovie))

def task_12():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(133337, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_13():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    print("Predictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def task_14():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000, to_date=datetime.datetime(2008, 1, 1))
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    uim_test = UserItemData('data/user_ratedmovies.dat', min_ratings=200, from_date=datetime.datetime(2008, 1, 2))

    mse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
    print(mse, mae, precision, recall, f)


if __name__ == '__main__':
    task_13()