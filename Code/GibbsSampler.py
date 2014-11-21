import numpy as np 
import scipy as sp 
from parseMovies import parseMovies
from parseData import create_user_movie_matrix
from parseData import getMeta
from random import randint

def gibbsSampler(numTopics, numIters):
	movie_data = parseMovies()
	user_movies = create_user_movie_matrix()
	info = getMeta()
	CountMT = np.zeros( (info["movies"], numTopics) )
	CountRUT = np.zeros( (5, info["users"], numTopics) )
	CountUT = np.zeros( (info["users"], numTopics) )
	idx_row, idx_col = user_movies.nonzero()
	idx = zip(idx_row, idx_col)
	for userid, movieid in idx:
		topic = randint(0, 18)
		CountMT[movieid, topic] += 1
		rate = user_movies[userid,movieid]
		CountRUT[rate, userid, topic] += 1
		CountUT[userid, topic] += 1
	for currIter in xrange(numIters):
		


if __name__ == "__main__":
	numTopics = 19
	numIters = 100
	gibbsSampler(numTopics, numIters)