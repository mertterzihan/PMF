import numpy as np
import scipy as sp
from parseMovies import parseMovies
from parseData import create_user_movie_matrix
from parseData import getMeta
from random import randint, shuffle, sample
from itertools import izip
import math
from collections import defaultdict
from matplotlib.mlab import PCA
from matplotlib import pyplot as plt


class GibbsSampler(object):

    def __init__(self, numTopics, alpha, beta, gamma):
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.info = getMeta()

        self.user_movies = create_user_movie_matrix()
        user_indices, movie_indices = self.user_movies.nonzero()
        self.user_movie_indices = zip(user_indices, movie_indices)

        self.CountMT = np.zeros( (self.info["movies"], numTopics) )
        self.CountRUT = np.zeros( (6, self.info["users"], numTopics) )  # ratings 1-5 and 0
        self.CountUT = np.zeros( (self.info["users"], numTopics) )
        self.topic_assignments = np.zeros((self.info["users"], self.info["movies"]))

        # Normalization factors
        self.CountT = np.zeros(numTopics)
        self.CountU = np.zeros(self.info["users"])
        self.CountRU = np.zeros((6, self.info["users"]))

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
                # self.CountT[prev_topic] -= 1
                self.CountU[userid] -= 1
                self.CountRU[rating, userid] -= 1

                # Get probability distribution for (user, movie) over topics
                topic_probs = self.getTopicProb(userid, movieid, rating)

                # Normalize
                topic_probs = topic_probs / sum(topic_probs)

                # Sample new topic
                new_topic = np.random.choice(self.numTopics, 1, p=topic_probs)

                self.topic_assignments[userid, movieid] = new_topic
                # Update new topic assignments
                self.CountMT[movieid, new_topic] += 1
                self.CountUT[userid, new_topic] += 1
                self.CountRUT[rating, userid, new_topic] += 1

                # Assign normalization factors
                # self.CountT[new_topic] += 1
                self.CountU[userid] += 1
                self.CountRU[rating, userid] += 1

            print "Finished iteration %d" % currIter
            #print self.logLike()

    def logLike(self):
        # phi = self.calcPhi()
        kappa = self.calcKappa()
        ll = 0

        for userid, movieid in self.user_movie_indices:
            topic = self.topic_assignments[userid, movieid]
            rating = self.user_movies[userid, movieid]
            # ll += math.log(phi[movieid, topic]) + math.log(kappa[rating, userid, topic])
            ll += math.log(kappa[rating, userid, topic])
        return ll

    def calcPhi(self):
        phi = (self.CountMT + self.beta)
        norm = (self.CountT + self.info["movies"] * self.beta)

        for topic in xrange(self.numTopics):
            phi[:, topic] /= norm[topic]

        return phi

    def calcKappa(self):
        kappa = (self.CountRUT + self.gamma)
        norm = (self.CountRU + self.numTopics * self.gamma)

        for rating, userid in zip(xrange(6), xrange(self.info["users"])):
            kappa[rating, userid, :] /= norm[rating, userid]

        return kappa

    def getTopicProb(self, userid, movieid, rating):
        # p_mt = (self.CountMT[movieid, :] + self.beta) / (self.CountT + self.info["movies"] * self.beta)
        p_ut = (self.CountUT[userid, :] + self.alpha) / (self.CountU[userid] + self.numTopics * self.alpha)
        p_rut = (self.CountRUT[rating, userid, :] + self.gamma) / (self.CountRU[rating, userid] + self.numTopics * self.gamma)
        return p_ut * p_rut

    def genMostLikelyTopic(self):
        phi = self.calcPhi()
        movies = parseMovies()
        topics = defaultdict(list)
        for movieid in xrange(self.info["movies"]):
            top_topic = np.argsort(phi[movieid, :])[-1]

            topics[top_topic].append(movies[movieid][0])
        return topics

    def genMostLikelyMovies(self):
        movies = parseMovies()
        phi = self.calcPhi()
        for topic in sample(xrange(self.numTopics), 10):
            top_movies = np.argsort(phi[:, topic])
            print "Topic: %d" % topic
            print "\n".join("%s: %.4f" % (movies[movieid][0], phi[movieid, topic]) for movieid in top_movies[-10:])
            print ""

    def visualizePCA(self):
        phi = self.calcPhi()
        movies = parseMovies()
        pca = PCA(phi)

        indices = sample(xrange(len(movies)), 50)

        x_axis = pca.Y[indices, 0]
        y_axis = pca.Y[indices, 1]

        plt.scatter(x_axis, y_axis)
        for idx, x, y in izip(indices, x_axis, y_axis):
            plt.annotate(movies[idx][0].encode('ascii', 'ignore'), (x, y))
        plt.show()
        return pca

if __name__ == "__main__":
    numTopics = 20
    numIters = 10
    alpha = 0.1
    beta = 0.01
    gamma = 0.9
    sampler = GibbsSampler(numTopics, alpha, beta, gamma)
    
    sampler.run(numIters)
    # sampler.genMostLikelyMovies()
    sampler.visualizePCA()
    topics = sampler.genMostLikelyTopic()
    for topicid in topics:
        print "Topic: %d" % topicid
        print "\n".join(title for title in topics[topicid][:10])
        print ""
