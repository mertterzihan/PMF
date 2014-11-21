from scipy import sparse
import numpy as np


def parseData():
    with open("../ml-100k/u.data") as f:
        for line in f:
            info = line.split()
            # Indexes are 1 based
            yield int(info[0]) - 1, int(info[1]) - 1, int(info[2])


def create_user_movie_matrix():
    info = getMeta()
    user_movies = sparse.lil_matrix((info["users"], info["movies"]),
                                    dtype=np.uint8)

    for user_id, movie_id, rating in parseData():
        user_movies[user_id, movie_id] = rating

    print "Finished Creating User-Movie Matrix"

    return user_movies.asformat("csr")


def getMeta():
    info = {}
    with open("../ml-100k/u.info") as f:
        info["users"] = int(f.readline().split()[0])
        info["movies"] = int(f.readline().split()[0])
        info["ratings"] = int(f.readline().split()[0])

    return info


if __name__ == '__main__':
    create_user_movie_matrix()
