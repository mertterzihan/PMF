import numpy as np


def parseMovies():
    data_folder = '../ml-100k'

    reader = open(data_folder+'/u.item')

    movies = list()

    for line in reader:
        line = line[:-1]
        data = line.split('|')
        topics = np.asarray(data[4:], dtype=np.int8)
        movies.append( (data[1], topics) )
    return movies