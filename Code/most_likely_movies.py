from parseMovies import parseMovies
import numpy as np
import sys
import argparse


def topic_distribution(fname, collection_name):
    npfile = np.load(fname)
    return npfile[collection_name].mean(axis=0)


def print_most_likely_movies(fname, collection_name):
        topic_dist = topic_distribution(fname, collection_name)
        movies = parseMovies()

        for topic in xrange(topic_dist.shape[1]):
            top_movies = np.argsort(topic_dist[:, topic])
            print "Topic: %d" % topic
            print "\n".join("%s: %.4f" % (movies[movieid][0], topic_dist[movieid, topic]) for movieid in top_movies[-10:])
            print ""


def main():
    """
    Usage:
        python most_likelymovies.py result_lda/result.npz phi_collection
        python most_likelymovies.py result_poisson/result.npz beta_collection > tmp.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Numpy file name")
    parser.add_argument("collection_name", help="Collection of topic distributions")
    args = parser.parse_args()
    print_most_likely_movies(args.fname, args.collection_name)

if __name__ == '__main__':
    main()
