import numpy as np
import scipy as sp
from parseMovies import parseMovies
from parseData import create_user_movie_matrix
from parseData import getMeta
from random import randint, shuffle
from itertools import izip


class GibbsSampler(object):

    def __init__(self, numTopics, alpha, beta, gamma):
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        info = getMeta()

        self.user_movies = create_user_movie_matrix()
        user_indices, movie_indices = self.user_movies.nonzero()
        self.user_movie_indices = zip(user_indices, movie_indices)

        self.CountMT = np.zeros( (info["movies"], numTopics) )
        self.CountRUT = np.zeros( (6, info["users"], numTopics) )  # ratings 1-5 and 0
        self.CountUT = np.zeros( (info["users"], numTopics) )
        self.topic_assignments = np.zeros((info["users"], info["movies"]))

        # Normalization factors
        self.CountT = np.zeros(numTopics)
        self.CountU = np.zeros(info["users"])
        self.CountRU = np.zeros((6, info["users"]))

        for userid, movieid in self.user_movie_indices:
            topic = randint(0, numTopics - 1)
            self.CountMT[movieid, topic] += 1
            rating = self.user_movies[userid, movieid]
            self.CountRUT[rating, userid, topic] += 1
            self.CountUT[userid, topic] += 1
            self.topic_assignments[userid, movieid] = topic

            self.CountT[topic] += 1
            self.CountU[userid] += 1
            self.CountRU[rating, userid] += 1

    def run(self, numIters):
        for currIter in xrange(numIters):
            shuffle(self.user_movie_indices)
            for userid, movieid in self.user_movie_indices:

                # Unassign previous topic
                prev_topic = self.topic_assignments[userid, movieid]
                rating = self.user_movies[userid, movieid]
                self.CountMT[movieid, prev_topic] -= 1
                self.CountUT[userid, prev_topic] -= 1
                self.CountRUT[rating, userid, prev_topic] -= 1

                # Unassign normalization factors
                self.CountT[prev_topic] -= 1
                self.CountU[userid] -= 1
                self.CountRU[rating, userid] -= 1

                # Get probability distribution for (user, movie) over topics
                topic_probs = self.getTopicProb(userid, movieid, rating)

                # Normalize
                topic_probs = topic_probs / sum(topic_probs)

                # Sample new topic
                new_topic = np.random.choice(numTopics, 1, p=topic_probs)

                self.topic_assignments[userid, movieid] = new_topic
                # Update new topic assignments
                self.CountMT[movieid, new_topic] += 1
                self.CountUT[userid, new_topic] += 1
                self.CountRUT[rating, userid, new_topic] += 1

                # Assign normalization factors
                self.CountT[new_topic] += 1
                self.CountU[userid] += 1
                self.CountRU[rating, userid] += 1

            print "Finished iteration %d" % currIter

    def getTopicProb(self, userid, movieid, rating):
        p_mt = (self.CountMT[movieid, :] + self.gamma) / (self.CountT + self.numTopics * self.gamma)
        p_ut = (self.CountUT[userid, :] + self.alpha) / (self.CountU[userid] + self.numTopics * self.alpha)
        p_rut = (self.CountRUT[rating, userid, :] + self.gamma) / (self.CountRU[rating, userid] + self.numTopics * self.gamma)
        return p_mt * p_ut * p_rut


if __name__ == "__main__":
    numTopics = 19
    numIters = 100
    alpha = 0.1
    beta = 0.01
    gamma = 0.1
    sampler = GibbsSampler(numTopics, alpha, beta, gamma)
    sampler.run(10)