# DBSCAN+ w.r.t. cosine similarity, DBSCAN+ & TI-DBSCAN+ w.r.t. Euclidean on normalized vectors

**Aim of the project**
The aim of this project was to implement 3 different algorithms - DBSCAN+ w.r.t. cosine similarity, DBSCAN+ & TI-DBSCAN+ w.r.t. Euclidean on normalized vectors in Python and in C-like language and compare the results. In this repo you will find python version of the algorithm, C++ version of my colleague is under: https://github.com/flyeyesport/dbscan

The “plus” (+) version modifies the classical one by assigning border points to possibly many clusters and not to the first one as is the case of the classical DBSCAN algorithm. TI-DBSCAN version uses the triangle inequality property (TI) that reduces the number of potential candidates to be members of epsilon-neighborhood of a given point. The results of TI-DBSCAN and DBSCAN are the same, but the first version is much faster. We tested our implementations of the algorithms on 5 different datasets:

• Dataset A. my_test0.csv - our 2D test dataset with 8 elements to check if Plus version is working properly

• Dataset B. my_test1.csv - our 2D dataset with 450 elements, on which it is easy to see how cosine similarity works and it also includes cluster of 4 elements that is built of 3 border points of another cluster

• Dataset C. complex9.csv - artificial 2D dataset with 3031 elements from https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/complex9.arff (converted from .arff to .csv file)

• Dataset D. cluto-t7-10k.csv - artificial 2D dataset with 10'000 elements from https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff (converted from .arff to .csv file and “noise” labels changed to “-1”)

• Dataset E. letter.csv - artificial 16D dataset with 20'000 elements from https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/real-world/letter.arff (converted from .arff to .csv file and labels changed from A-Z to 1-26 numbers respectively)

**Description of a form of INPUT and OUTPUT data**

General assumptions about input data:

•	Our programs accept input data as files in csv format.

•	In all input csv files the first row is a header row with the names of columns of data fields in the rest of the rows.

•	The last column in input csv files is always a column with ground truth values. These are numbers of clusters to which our algorithms should assign corresponding data points. Of course depending on the dataset, and used algorithm parameters our algorithms may produce different results. It is also possible that our algorithms will divide the input dataset into the same clusters as defined by values in the ground truth column, but number of clusters will be different than in ground truth. In other words the contents of each cluster will be the same as expected, but the labels of the clusters will be different or in different order.

•	Ground truth labels from the last column are always numbers.

•	DBSCAN+ version of the algorithm may assign border points to more than one cluster. In the PYTHON version of the algorithm ground true labels are represented always by only one number, but then in predicted labels, it can assign border points to many clusters represented by a list of these clusters. 

General assumptions about output data:

PYTHON version

•	All OUT files are in .csv format, all STAT files are in .txt format

•	All files have headers

•	OUT files have 4 fixed headers: index, number of calculations, point type, Cids (cluster ids) and one variable header – number of dimensions (in our case it is either 2D data so only “d1,d2” or 16D data so “d1,…,d16”)

•	STAT files have variable headers depending on the type of file (if it is either about cosine similarity, Euclidean on normalized forms or TI-DBSCAN+)

**User guide**

Whole Python project was prepared with the use of Python version 3.8.3.
Python version consists of 4 programs:

•	Heuristic_of_epsilon.py

•	DBSCAN_plus_cosine_sim.py

•	DBSCAN_plus_euclidean_norm_forms.py

•	TI_DBSCAN_plus_euclidean_norm_forms.py

It is best to start with “Heuristic_of_epsilon.py” for choosing the right parameters for the program. In this program, there is a note about how to use it and what datasets are included. The most important thing is that there are 5 datasets A,B,C,D and E (described in point 1 of this document), and if you would like to see the results for a given dataset, you need to uncomment section responsible for this dataset and comment other ones. In “Heuristic_of_epsilon.py” program there are 2 such places – first while importing the dataset (below for example by default uncommented is dataset B):
 

And a second one about choosing a right y axis limit to clearly see the elbow plot:

 

Note: with the first file there are only 8 values to prove that plus version of the algorithm is working, so the elbow plot would not make much sense here.
In programs you will also see places, where you need to enter certain value. This are commented and are starting with “ENTER”. In “Heuristic_of_epsilon.py” there are 2 such places – first one is while choosing the number of nearest neighbors:
 

And a second one is at the very end and it is only of use for program with cosine similarity measure. After plotting the elbow plot, you choose the value of interest on the plot (from the elbow) and enter it in “eps_elbow”. The console will print you the value of Epsilon you should choose for cosine similarity measure program.
Important note: The program itself is plotting only k-distances plot with respect to Euclidean distance, on vectors that we normalized, from which value of epsilon for cosine similarity measure is extracted. It is not plotting k-distances plot with respect to cosine similarity measure.
 
For program with Euclidean on normalized forms versions, you simply choose value from the elbow without writing anything else into program.
In 3 other programs: DBSCAN_plus_cosine_sim.py, DBSCAN_plus_euclidean_norm_forms.py, TI_DBSCAN_plus_euclidean_norm_forms.py there are also notes on the beginning how to use them, but in all these programs there are 4 such places when you need to comment / uncomment section corresponding to the wanted file. All these are placed in MAIN function at the end of the program. First one is about reading the input file (below you will see the picture for the program with cosine similarity measure but this will look almost the same for the other two):

 

Second one is about writing into OUT file:
 

Third one is about writing everything, except total runtime, to STAT file:
 

And the last one is about writing total runtime to STAT file:
 

In these programs you can also enter the data of the parameters, known from the Heuristic_of_epsilon file. You can do it also in MAIN function:

 

Optional - You can also decide if you want to plot the results for 2D data or not by commenting / uncommenting the last block of MAIN function:
 


