import numpy as np
import scipy as sp
from parseMovies import parseMovies
from parseData import create_user_movie_matrix
from parseData import getMeta
from random import randint
from itertools import izip


class GibbsSampler(object):

    def __init__(self, numTopics, alpha, beta, gamma, kappa):
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa

        info = getMeta()

        self.user_movies = create_user_movie_matrix()
        self.user_indices, self.movie_indices = self.user_movies.nonzero()

        self.CountMT = np.zeros( (info["movies"], numTopics) )
        self.CountRUT = np.zeros( (5, info["users"], numTopics) )
        self.CountUT = np.zeros( (info["users"], numTopics) )
        self.topic_assignments = np.zeros((info["users"], info["movies"]))

        for userid, movieid in izip(self.user_indices, self.movie_indices):
            topic = randint(0, numTopics - 1)
            self.CountMT[movieid, topic] += 1
            rating = self.user_movies[userid, movieid]
            self.CountRUT[rating, userid, topic] += 1
            self.CountUT[userid, topic] += 1
            self.topic_assignments[userid, movieid] = topic

        print "Finished initialization"

    def run(self, numIters):
        for currIter in xrange(numIters):
            for userid, movieid in izip(self.user_indices, self.movie_indices):
                topic_probs = np.zeros(self.numTopics)

                # Unassign previous topic
                prev_topic = self.topic_assignments[userid, movieid]
                rating = self.user_movies[userid, movieid]
                self.CountMT[movieid, prev_topic] -= 1
                self.CountUT[userid, prev_topic] -= 1
                self.CountRUT[rating, userid, prev_topic] -= 1

                # Get probability distribution for (user, movie) over topics
                for topic in self.numTopics:
                    topic_probs.append(self.getTopicProb(topic, userid, movieid, rating))

                # Normalize
                topic_probs = topic_probs / topic_probs.sum()

                # Sample new topic
                new_topic = np.random.multinomial(1, topic_probs)

                self.CountMT[movieid, new_topic] += 1
                self.CountUT[userid, new_topic] += 1
                self.CountRUT[rating, userid, new_topic] += 1
            print "Finished iteration %d" % currIter

    def getTopicProb(self, topic, userid, movieid, rating):
        p_mt = float(self.CountMT[movieid, topic] + self.gamma) / (self.CountMT[movieid, :].sum() + self.numTopics * self.gamma)
        p_ut = float(self.CountUT[userid, topic] + self.alpha) / (self.CountMT[userid, :].sum() + self.numTopics * self.alpha)
        p_rut = float(self.CountRUT[rating, userid, topic] + self.kappa) / (self.CountRUT[rating, userid, :].sum() + self.numTopics * self.kappa)
        return p_mt * p_ut * p_rut

# def gibbsSampler(numTopics, numIters):
#     movie_data = parseMovies()
#     user_movies = create_user_movie_matrix()
#     info = getMeta()
#     CountMT = np.zeros( (info["movies"], numTopics) )
#     CountRUT = np.zeros( (5, info["users"], numTopics) )
#     CountUT = np.zeros( (info["users"], numTopics) )
#     idx_row, idx_col = user_movies.nonzero()
#     idx = zip(idx_row, idx_col)
#     for userid, movieid in idx:
#         topic = randint(0, 18)
#         CountMT[movieid, topic] += 1
#         rate = user_movies[userid,movieid]
#         CountRUT[rate, userid, topic] += 1
#         CountUT[userid, topic] += 1
#     for currIter in xrange(numIters):



if __name__ == "__main__":
    numTopics = 19
    numIters = 100
    gibbsSampler(numTopics, numIters)