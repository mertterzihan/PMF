import numpy as np
from random import shuffle
from itertools import product
from parseData import create_user_movie_matrix, getMeta


def gamma(shape, rate, size=None):
    return np.random.gamma(shape, 1.0 / rate, size)


class BayesianPoissonFactorization(object):

    def __init__(self, a, ap, b, c, cp, d, ntopics, ratings):
        """ ratings is a sparse user-movie ratings matrix """

        self.a = a
        self.ap = ap
        self.b = b
        self.c = c
        self.cp = cp
        self.d = d
        self.ratings = ratings
        self.ntopics = ntopics
        self.nusers, self.nmovies = ratings.shape

        # Init xi
        self.xis = np.zeros((self.nusers, 1))
        for user in xrange(self.nusers):
            self.xis[user] = gamma(ap, ap / b)

        # Init eta
        self.etas = np.zeros((self.nmovies, 1))
        for movie in xrange(self.nmovies):
            self.etas[movie] = gamma(cp, cp / d)

        # Init Beta
        self.betas = np.zeros((self.nmovies, self.ntopics))
        for movie, topic in product(xrange(self.nmovies), xrange(self.ntopics)):
            self.betas[movie, topic] = gamma(c, self.etas[movie])

        # Init theta
        self.thetas = np.zeros((self.nusers, self.ntopics))
        for user, topic in product(xrange(self.nusers), xrange(self.ntopics)):
            self.thetas[user, topic] = gamma(a, self.xis[user])

        user_indices, movie_indices = ratings.nonzero()
        self.nonzero_indices = zip(user_indices, movie_indices)

        # Init z
        self.zs = np.zeros((self.nusers, self.nmovies, self.ntopics))
        for user, movie in self.nonzero_indices:
            p = np.multiply(self.thetas[user, :], self.betas[movie, :])
            p /= p.sum()

            self.zs[user, movie, :] = np.random.multinomial(self.ratings[user, movie], p)

        def sample(self):
            """ One iteration of Gibbs sampling. """
            # Sample thetas
            rand_users = range(self.nusers)
            shuffle(rand_users)
            for user in rand_users:
                rand_topics = range(self.ntopics)
                shuffle(rand_topics)
                for topic in rand_topics:
                    self.thetas[user, topic] = gamma(a + np.sum(self.zs[user, :, topic]),
                                                     self.xis[user] + np.sum(self.betas[:, topic]))

            # Sample betas



def main():
    ratings = create_user_movie_matrix()
    bpf = BayesianPoissonFactorization(0.3, 0.3, 1.0, 0.3, 0.3, 1.0, 10,
                                       ratings)

if __name__ == '__main__':
    main()