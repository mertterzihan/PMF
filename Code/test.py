from parseData import get_test_reviews, get_split_review_mats
import numpy as np
import sys
from itertools import izip
import math
from most_likely_movies import get_best_params


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
        mean_rating = max(5, mean_rating + 1)
        rmse += (true_rating - mean_rating) ** 2
        rmses.append((true_rating - mean_rating) ** 2)
        count += 1
    print max(rmses)
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
            rmse += 25
        else:
            rmse += (predicted - true_rating) ** 2
        count += 1

    return math.sqrt(rmse / count)


def main():
    pass


if __name__ == '__main__':
    print test_iid_users()
