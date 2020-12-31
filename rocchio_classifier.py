import math
import sys


class RocchioClassifier:
    def __init__(self, train_set):
        self.training_set = train_set
        self.class_centroids = {}
        self.training()

    def training(self):
        class_size = {}
        for doc_name, document_vector in self.training_set.items():
            doc_class = document_vector[-1]
            if doc_class not in self.class_centroids.keys():
                self.class_centroids[doc_class] = document_vector[0:-1]
                class_size[doc_class] = 1
            else:
                self.class_centroids[doc_class] = [self.class_centroids[doc_class][i] + document_vector[i]
                                                   for i in range(len(document_vector) - 1)]
                class_size[doc_class] += 1
        for c in self.class_centroids.keys():
            for i in range(len(self.class_centroids[c])):
                self.class_centroids[c][i] /= float(class_size[c])

    @staticmethod
    def euclidean_dist(vec1, vec2):
        if len(vec1) != len(vec2):
            print('Error. Vectors of different size')
            print(vec1)
            print(vec2)
            exit(0)

        return sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))])**0.5

    @staticmethod
    def cosine_similarity(vec1, vec2):
        divider = ((sum(x**2 for x in vec1))**0.5) * (sum(x**2 for x in vec2)**0.5)
        distance = sum(x * y for x, y in zip(vec1, vec2))/divider
        return 1 - distance

    def predict(self, vector,euclidean_dist):
        winner_class = -1
        lowest_distance = sys.float_info.max
        for class_name, class_vector in self.class_centroids.items():
            if euclidean_dist == 1:
                distance = self.euclidean_dist(vector, class_vector)
            else:
                distance = self.cosine_similarity(vector, class_vector)
            if distance < lowest_distance:
                winner_class = class_name
                lowest_distance = distance

        return winner_class
