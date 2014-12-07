import numpy as np
from random import shuffle
from itertools import product
from parseData import create_user_movie_matrix, getMeta
from scipy.stats import poisson
from scipy.stats import gamma as gammafun


def gamma(shape, rate, size=None):
    return np.random.gamma(shape, 1.0 / rate, size)

def gammapdf(x, shape, rate):
    return gammafun.pdf(x, shape, 1.0/rate)


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
                self.thetas[user, topic] = gamma(self.a + np.sum(self.zs[user, :, topic]),
                                                 self.xis[user] + np.sum(self.betas[:, topic]))
        print "Done with thetas"

        # Sample betas
        rand_movies = range(self.nmovies)
        shuffle(rand_movies)
        for movie in rand_movies:
            rand_topics = range(self.ntopics)
            shuffle(rand_topics)
            for topic in rand_topics:
                self.betas[movie, topic] = gamma(self.a + np.sum(self.zs[:,movie,topic]),
                                                 self.etas[movie] + np.sum(self.thetas[:,topic]))
        print "Done with betas"

        # Sample Xis
        rand_users = range(self.nusers)
        shuffle(rand_users)
        for user in rand_users:
            self.xis[user] = gamma(self.ap + self.ntopics*self.a, self.b + self.thetas[user,:].sum())
        print "Done with xis"

        # Sample Etas
        rand_movies = range(self.nmovies)
        shuffle(rand_movies)
        for movie in rand_movies:
            self.etas[movie] = gamma(self.cp + self.ntopics*self.c, self.d + self.betas[movie,:].sum())
        print "Done with etas"

        # Sample Z's
        shuffle(self.nonzero_indices)
        for user, movie in self.nonzero_indices:
            p = np.multiply(self.thetas[user, :], self.betas[movie, :])
            p /= p.sum()

            self.zs[user, movie, :] = np.random.multinomial(self.ratings[user, movie], p)
        print "Done with zs"

        print self.compute_ll()

    def compute_ll(self):
        ll = 0

        for user, movie in self.nonzero_indices:
            assert self.ratings[user,movie] > 0
            ll += np.log(poisson.pmf(self.ratings[user,movie], np.dot(self.thetas[user,:], self.betas[movie,:])))

        for user in xrange(self.nusers):
            try:
                assert gammapdf(self.xis[user], self.ap, self.ap/self.b) > 0
            except AssertionError:
                print self.xis[user], self.ap, self.ap/self.b, gammapdf(self.xis[user], self.ap, self.ap/self.b)
                raise
            ll += np.log(gammapdf(self.xis[user], self.ap, self.ap/self.b))
            for topic in xrange(self.ntopics):
                assert gammapdf(self.thetas[user,topic], self.a, self.xis[user]) > 0
                ll += np.log(gammapdf(self.thetas[user,topic], self.a, self.xis[user]))

        for movie in xrange(self.nmovies):
            assert gammapdf(self.etas[movie], self.cp, self.cp/self.d) > 0
            ll += np.log(gammapdf(self.etas[movie], self.cp, self.cp/self.d))
            for topic in xrange(self.ntopics):
                assert gammapdf(self.betas[movie,topic], self.c, self.etas[movie]) > 0
                ll += np.log(gammapdf(self.betas[movie,topic], self.c, self.etas[movie]))

        return ll

def main():
    ratings = create_user_movie_matrix()
    bpf = BayesianPoissonFactorization(0.3, 0.3, 1.0, 0.3, 0.3, 1.0, 10,
                                       ratings)
    bpf.sample()

if __name__ == '__main__':
    main()