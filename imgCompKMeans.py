#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
#  author: Zoe Zinn (Dec 2023)
# purpose: K-Means compression on an image

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import pickle
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

import argparse


def parseArguments():
    parser = argparse.ArgumentParser(description='KMeans compression of images')

    parser.add_argument('--imageFileName', action='store',
                        dest='imageFileName', default="", required=True,
                        help='input image file')
    
    parser.add_argument('--k', action='store',
                        dest='k', default="", type=int, required=True,
                        help='number of clusters')

    parser.add_argument('--outputFileName', action='store',
                        dest='outputFileName', default="", required=True,
                        help='output imagefile name')

    return parser.parse_args()


def main():
    parms = parseArguments()

    img = imread(parms.imageFileName)
    img_size = img.shape

    # Reshape it to be 2-dimension
    # in other words, its a 1d array of pixels with colors (RGB)
    X = img.reshape(img_size[0] * img_size[1], img_size[2])


    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=parms.k, init ='random', max_iter=100,  n_init=1 )
    kmeans.fit(X)

    labels = kmeans.predict(X)


    # -- replace colors in the image with their respective centroid
    X_compressed = kmeans.cluster_centers_[kmeans.labels_]

    del X      # save memory by deleting X when it is no longer needed
    del labels # save memory by deleting labels when no longer needed


    # save modified image
    # Reshape to have the same dimension as the original image
    X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])

    # Save the new compressed image
    plt.imsave(parms.outputFileName, X_compressed.astype(np.uint8))


if __name__ == '__main__':
    main()
    exit(0)
