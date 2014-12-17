from parseData import get_test_reviews, get_split_review_mats, getMeta
import numpy as np
import sys
from itertools import izip
import math
from most_likely_movies import get_best_params


def top_recommendations_poisson():
    params = get_best_params("poisson")
    info = getMeta()
    beta = params["beta"]
    theta = params["theta"]

    reviews = get_test_reviews()

    precision = 0.0
    num_users = 0
    for user in xrange(info["users"]):
        movie_ratings = []
        for movie in xrange(info["movies"]):
            rating = np.dot(theta[user, :], beta[movie, :])
            movie_ratings.append((movie, rating))

        movie_ratings = sorted(movie_ratings, key=lambda x: x[1])
        top_movies_for_user = set(movie for movie, rating in movie_ratings[-20:])

        user_precision = 0.0
        movies = reviews[user, :].nonzero()[0]

        for movie in movies:
            if movie in top_movies_for_user:
                user_precision += 1
        if len(movies) > 0:
            num_users += 1
            print user_precision, len(movies)
            precision += (user_precision / len(movies))
    return precision / num_users

def test_poisson():
    params = get_best_params("poisson")
    beta = params["beta"]
    theta = params["theta"]

    reviews = get_test_reviews()
    rmse = 0.0
    rmses = []
    count = 0
    for user, movie in izip(*reviews.nonzero()):
        true_rating = reviews[user, movie]
        mean_rating = np.dot(theta[user, :], beta[movie, :])
        mean_rating = max(1, min(5, mean_rating + 1))
        rmse += (true_rating - mean_rating) ** 2
        rmses.append((true_rating - mean_rating) ** 2)
        count += 1

    return math.sqrt(rmse / count)


def test_iid_users():
    train, test = get_split_review_mats()
    avg_ratings = train.sum(axis=0) / (train != 0).sum(axis=0).astype(np.float)

    rmse = 0.0
    count = 0
    for user, movie in izip(*test.nonzero()):
        true_rating = test[user, movie]
        predicted = avg_ratings[movie]

        if np.isnan(predicted):
            # The movie wasn't rated by any users in the training data set
            continue

        rmse += (predicted - true_rating) ** 2
        count += 1

    return math.sqrt(rmse / count)

def test_lda():
    params = get_best_params("lda")
    info = getMeta()
    phi = params["phi"]
    kappa = params["kappa"]

    reviews = get_test_reviews()
    rmse = 0.0
    count = 0

    rating_values = np.asarray([0,1.0,2.0,3.0,4.0,5.0])
    for user, movie in izip(*reviews.nonzero()):
        topic = np.argmax(phi[movie,:])
        estimated_rating = np.dot(kappa[:,user,topic]/np.sum(kappa[:,user,topic]), rating_values)
        true_rating = reviews[user, movie]
        rmse += (true_rating - estimated_rating) ** 2
        count += 1
    return math.sqrt(rmse / count)

def main():
    pass


if __name__ == '__main__':
    print top_recommendations_poisson()
    # print test_iid_users()
    # print test_poisson()
