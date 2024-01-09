import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_labeled_images(categories, path_to_categories):
    labels = []
    all_images = []
    for i in range(len(categories)):
        images = os.listdir(path_to_categories+'/' + categories[i])
        for image in images:
            labels.append(i)
            all_images.append(cv2.imread(path_to_categories+'/'+categories[i]+'/'+image))
    return all_images, labels

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        return descriptors.astype(np.float32)
    else:
        return np.array([])

def create_bag_of_words(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    return kmeans

def compute_histogram(image, kmeans):
    features = extract_sift_features(image)
    if features is not None:
        labels = kmeans.predict(features)
        histogram, _ = np.histogram(labels, bins=range(kmeans.n_clusters + 1))
        histogram = histogram.astype(float)
        histogram /= histogram.sum()
        return histogram
    else:
        return None