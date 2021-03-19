'''
IMPORTANT NOTES
Below you can plot k-distances graph for five different datasets.
By default, uncommented is dataset B.
If you want to test other datasets, comment every section with B and uncomment
others.
A. my_test0.csv - my 2D test dataset with 8 elements to check if Plus
version is working properly
B. my_test1.csv - my 2D dataset with 450 elements, on which it is
easy to see how cosine similarity works
C. complex9.csv - artificial 2D dataset with 3031 elements from
https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/complex9.arff
D. cluto-t7-10k.csv - artificial 2D dataset with 10'000 elements from
https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff
E. letter.csv - artificial 16D dataset with 20'000 elements from
https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/real-world/letter.arff

In case of cosine similarity measure program only plots k-distances plot
for normalized forms of vectors and extracts value of epsilon for cosine
similarity, instead of plotting k-distances plots with nearest neighbours
with respect to cosine similarity.
'''


# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn import preprocessing

# importing files
# A. Uncomment below for A dataset
# NOTE - in this case it is not really important or valid to see the elbow
# because it is only a test where we set fixed Eps value by ourselves to see if
# plus version works properly
#input_file = 'my_test0.csv'

# B. Uncomment below for B dataset
#input_file = 'my_test1.csv'

# C. Uncomment below for C dataset
#input_file = 'complex9.csv'

# D. Uncomment below for D dataset
#input_file = 'cluto-t7-10k.csv'

# E. Uncomment below for E dataset
input_file = 'letter.csv'

df = pd.read_csv(input_file, delimiter=',')
df.drop(df.columns[-1], axis = 1, inplace = True)
X = [list(row) for row in df.values]
X = np.array(X)
X = preprocessing.normalize(X)

# ENTER - below you enter the number of nearest neighbours you want algorith to find
no_of_min_points = 32
neigh = NearestNeighbors(n_neighbors=no_of_min_points)
nbrs = neigh.fit(X)

# below you plot the k-distances plot in order
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,no_of_min_points-1]
plt.plot(distances)

# different dataset need different limits of y axis to clearly see the elbow

# A. Uncomment below fo y limit for dataset A
# NOTE - in this case it is not really important or valid to see the elbow
#plt.ylim((0,1))

# B. Uncomment below for y limit for dataset B
#plt.ylim((0,0.025))

# C. Uncomment below for y limit for dataset C
#plt.ylim((0,0.004))

# D. Uncomment below for y limit for dataset D
#plt.ylim((0,0.002))

# E. Uncomment below for y limit for dataset E
#plt.ylim((0,0.25))

plt.show()

# ENTER - below enter the elbow value - it will be calculated to an Eps value that
# you then put into the program with cosine similarity measure
eps_elbow = 0.125
print((((eps_elbow)**2)-2)/-2)