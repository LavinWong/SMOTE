import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


class Smote(object):
    def __init__(self, N=50, k=5, r=2):
        #Initialize
        #Increase present
        self.N = N
        #Nearest K samples
        self.k = k
        # self.r decide the distance function
        self.r = r
        # self.newcase count creat cases
        self.newcase = 0

    def fit(self, dataset):
        #Initialize
        self.data = dataset
        # self.total_num is total rows of minority class
        #self.num_feat is number of minority class features.
        self.total_num, self.num_feat = self.data.shape

        # If N less than 100%
        #The created samples are less than the original minority samples
        if (self.N < 100):
            # Random selesct N*T/100 samples
            np.random.shuffle(self.data)
            self.total_num = int(self.N * self.total_num / 100)
            self.data = self.data[0:self.total_num, :]
            # Set N to 100%
            self.N = 100

        # Check the total num is less than nearest samples or not
        if (self.total_num <= self.k):
            # If yes then update the k be total num - 1
            self.k = self.total_num - 1

        # Let N be a multiple of 100
        N = int(self.N / 100)
        # Save created samples
        self.synthetic = np.zeros((self.total_num * N, self.num_feat))

        # Set nearest k, r=1 is Manhattan distance, r=2 is Euclidean distance
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.data)

        # Go all the samples
        for i in range(len(self.data)):
            # find the nearest samples
            nnarray = neighbors.kneighbors(self.data[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]

            # Build new samples
            self.build_sample(N, i, nnarray)

        # return result
        return self.synthetic

    # Build the new samples
    def build_sample(self, N, i, nnarray):
        for j in range(N):
            # attrs save new sample attributes
            attrs = []
            # select one sample from k smaples to create the new sample
            cre_sample = random.randint(0, self.k - 1)

            # calculate different
            diff = self.data[nnarray[cre_sample]] - self.data[i]
            # random select number between 0 to 1
            gap = random.uniform(0, 1)
            # put new sample in self.synthetic
            self.synthetic[self.newcase] = self.data[i] + gap * diff

            # Count the number of create cases
            self.newcase += 1

