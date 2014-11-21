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

        print "Finished initialization"

    def run(self, numIters):
        for currIter in xrange(numIters):
            prev_user = None
            # shuffle(self.user_movie_indices)
            for userid, movieid in self.user_movie_indices:
                topic_probs = np.zeros(self.numTopics)

                # Unassign previous topic
                prev_topic = self.topic_assignments[userid, movieid]
                rating = self.user_movies[userid, movieid]
                self.CountMT[movieid, prev_topic] -= 1
                self.CountUT[userid, prev_topic] -= 1
                self.CountRUT[rating, userid, prev_topic] -= 1

                try:
                    assert self.CountMT[movieid, prev_topic] >= 0
                except AssertionError:
                    print movieid, userid, prev_topic
                    continue

                assert self.CountUT[userid, prev_topic] >= 0
                assert self.CountRUT[rating, userid, prev_topic] >= 0


                # Unassign normalization factors
                self.CountT[prev_topic] -= 1
                self.CountU[userid] -= 1
                self.CountRU[rating, userid] -= 1

                # Get probability distribution for (user, movie) over topics
                for topic in xrange(self.numTopics):
                    topic_probs[topic] = self.getTopicProb(topic, userid, movieid, rating)

                # Normalize
                topic_probs = topic_probs / sum(topic_probs)

                # Sample new topic
                try:
                    new_topic = np.random.multinomial(1, topic_probs)
                except Exception:
                    print "Topic probs:", topic_probs
                    print "Topic sum:", sum(topic_probs[:-1])
                    print "User: %d, Movie: %d" % (userid, movieid)
                    raise

                # Update new topic assignments
                self.CountMT[movieid, new_topic] += 1
                self.CountUT[userid, new_topic] += 1
                self.CountRUT[rating, userid, new_topic] += 1

                # Assign normalization factors
                self.CountT[new_topic] += 1
                self.CountU[userid] += 1
                self.CountRU[rating, userid] += 1

                if prev_user != userid:
                    print "User: %d" % userid
                prev_user = userid

            print "Finished iteration %d" % currIter

    def getTopicProb(self, topic, userid, movieid, rating):
        p_mt = float(self.CountMT[movieid, topic] + self.gamma) / (self.CountT[topic] + self.numTopics * self.gamma)
        p_ut = float(self.CountUT[userid, topic] + self.alpha) / (self.CountU[userid] + self.numTopics * self.alpha)
        p_rut = float(self.CountRUT[rating, userid, topic] + self.gamma) / (self.CountRU[rating, userid] + self.numTopics * self.gamma)
        return p_mt * p_ut * p_rut


if __name__ == "__main__":
    numTopics = 19
    numIters = 100
    alpha = 0.1
    beta = 0.01
    gamma = 0.1
    sampler = GibbsSampler(numTopics, alpha, beta, gamma)
    sampler.run(10)