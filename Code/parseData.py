import numpy as np

NUM_REVIEWS = 100000

def parseData():
    with open("../ml-100k/u.data") as f:
        for line in f:
            info = line.split()
            # Indexes are 1 based
            yield int(info[0]) - 1, int(info[1]) - 1, int(info[2])


def create_user_movie_matrix():
    """ DEPRECATED """
    info = getMeta()
    user_movies = np.zeros((info["users"], info["movies"]),
                           dtype=np.uint8)

    for user_id, movie_id, rating in parseData():
        user_movies[user_id, movie_id] = rating

    return user_movies


def getMeta():
    info = {}
    with open("../ml-100k/u.info") as f:
        info["users"] = int(f.readline().split()[0])
        info["movies"] = int(f.readline().split()[0])
        info["ratings"] = int(f.readline().split()[0])

    return info


def get_split_review_mats():
    info = getMeta()
    train = np.zeros((info["users"], info["movies"]),
                     dtype=np.uint8)
    test = np.zeros((info["users"], info["movies"]),
                    dtype=np.uint8)

    data_iter = parseData()
    count = 0

    for user, movie, rating in data_iter:
        test[user, movie] = rating
        count += 1
        if count > NUM_REVIEWS * 0.2:
            break

    for user, movie, rating in data_iter:
        train[user, movie] = rating

    return train, test


def get_train_reviews():
    info = getMeta()
    train = np.zeros((info["users"], info["movies"]),
                     dtype=np.uint8)

    data_iter = parseData()
    count = 0

    for user, movie, rating in data_iter:
        count += 1
        if count > NUM_REVIEWS * 0.2:
            break

    for user, movie, rating in data_iter:
        train[user, movie] = rating

    return train


def get_test_reviews():
    info = getMeta()
    test = np.zeros((info["users"], info["movies"]),
                    dtype=np.uint8)

    data_iter = parseData()
    count = 0

    for user, movie, rating in data_iter:
        test[user, movie] = rating
        count += 1
        if count > NUM_REVIEWS * 0.2:
            break

    return test

if __name__ == '__main__':
    create_user_movie_matrix()
