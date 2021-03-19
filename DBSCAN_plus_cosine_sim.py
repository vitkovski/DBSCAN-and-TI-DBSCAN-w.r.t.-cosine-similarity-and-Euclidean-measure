'''
IMPORTANT NOTE
In this program in MAIN function you can import 5 different datasets.
By default, uncommented is dataset B.
If you want to test other datasets, comment every section with B and uncomment
others (it is needed while importing, writing to OUT and STAT).
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

In the MAIN function you can also enter Eps and MinPts value and decide if you
want to plot results for 2D datasets or not.
'''



# importing time library to start calculating total time
import time
start_time_total = time.time()

# importing other necessary libraries
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import comb
counter = 0

# function to compute cosine similarity
def cos_sim(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0

    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    global counter
    counter +=1
    return sumxy/math.sqrt(sumxx*sumyy)



# function to find nearest neighbours 
def find_neighbours(db, cosine, p, e):
    """
    return the indices of all points within epsilon of p w.r.t. cosine sim
    """
    return  [idx for idx, q in enumerate(db) if cosine(p,q) >= e]

# additional function for Rand helping to flatten nested lists
def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
            
            
# function to calculate Rand index
def rand_index_score(true, predicted):
    
    true2 = true[:] 
    
    if (any(isinstance(el, list) for el in predicted)):
        for idx1,q1 in enumerate(predicted):
            for idx2,q2 in enumerate(true2):
                if isinstance(q1, list) and q2 in q1 and idx1 == idx2:
                    
                    true2[idx2] = predicted[idx1]
                    
                    
                elif isinstance(q1, list) and q2 not in q1 and idx1 == idx2:
                    
                    predicted[idx1] = q1[0]
                    
                else:
                    true2[idx2] = true2[idx2]
                    predicted[idx1] = predicted[idx1]
    
    predicted = list(flatten(predicted))
    true2 = list(flatten(true2))
    predicted = [int(x) for x in predicted if str(x) != 'nan']
    A = np.c_[(true2, predicted)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(true2))
    
    tp_plus_fp = comb(np.bincount(true2), 2).sum()
    tp_plus_fn = comb(np.bincount(predicted), 2).sum()
    fp = tp_plus_fp - tp
    
    fn = tp_plus_fn - tp
    tn = comb(len(true), 2) - tp - fp - fn
    
    no_of_points_pairs = tp + fp + fn + tn
    
    rand_index = (tp + tn) / (tp + fp + fn + tn)
    return tp, tn, no_of_points_pairs, rand_index



# DBSCAN+ function
def dbscan(data, min_pts, eps, dist_func=cos_sim):
    """
    Run the DBSCAN+ clustering algorithm
    """
    C = 0                                   # cluster counter
    all_clusters = {}                       # dictionary to hold all of the clusters
    all_clusters.setdefault('border', [])   # initialization of a border key in dict
    visited = np.zeros(len(data))           # check to see if we visited this point
    cluster_id = [0]*len(data)              # list of cluster ids of all points
    point_type = [0]*len(data)              # list of point types of all points
    no_of_neighbours = [0]*len(data)        # list of no of neighbours of all points
    avg_no_border_clusters = 0
    result2 = {}                            # additional dictionary for calculations
    '''
    Algorithm goes through the data point by point, finding the epsilon
    neighbourhood of a point and check if it is a core point. If it is, 
    then check all of its neighbors and add them to the cluster that core point
    is in. Repeat until all points have been visited. One point may be a border
    point of many different clusters.
    '''
    for idx, point in enumerate(data): 
        if visited[idx] == 1:    
           continue
        visited[idx] = 1
       
        # in below's value all of point's neighbours are stacked
        neighbors = find_neighbours(data, dist_func, point, eps)
        no_of_neighbours[idx] = len(neighbors)
        
        # if it is not a core point, it is either noise or border    
        if len(neighbors) < min_pts:
            all_clusters.setdefault('noise', []).append(idx)
            cluster_id[idx] = -1
            point_type[idx] = -1
        # else there is a new cluster    
        else:
            C += 1
            all_clusters.setdefault(C, []).append(idx)
            cluster_id[idx] = C
            point_type[idx] = 1
            
            # then there is a check if in neighbours of the point there is
            # already a point that is in 'border' cluster. If it is then extend
            # actual cluster of such point
            check = any(item in all_clusters['border'] for item in neighbors)
            if check is True:
                bord = set(neighbors) & set(all_clusters['border'])
                all_clusters[C].extend(list(bord))

            # if a point was previously marked as noise, add it to the cluster
            # and mark it as a border point                
            for q in neighbors:
                if cluster_id[q] == -1:
                    cluster_id[q] = C
                    point_type[q] = 0
                    all_clusters['noise'].remove(q)
                    all_clusters.setdefault('border', []).append(q)
                    all_clusters[C].append(q)
                if visited[q] == 1:
                    continue
                visited[q] = 1
                
                # extend the search for every point in neighbours
                q_neighbors = find_neighbours(data, dist_func, data[q, :], eps)
                no_of_neighbours[q] = len(q_neighbors)
                if len(q_neighbors) >= min_pts:
                    neighbors.extend(q_neighbors)  # extend the search
                    point_type[q] = 1
                else:
                    all_clusters.setdefault('border', []).append(q)
                    point_type[q] = 0
                all_clusters[C].append(q)
                cluster_id[q] = C
            
            
            # to get the statistics of average no of clusters to which border points
            # belong there is a new of reverse dictionary, where point's ids are keys
            # and lists of clusters to which they belong are the values
            common = set(all_clusters[C]) & set(all_clusters['border']) 
            common = list(common)
            result = {}
            for m in common:
               result.setdefault(m, [])
            for k, v in all_clusters.items():
                for x, value in enumerate(common):
                    if value in v:
                        if k != 'border':
                            result[value].append(k)            
            for x, y in result.items():
                if len(y) > 1:
                    cluster_id[x] = y
            result2.update(result)
            if len(result2.keys()) != 0:
                avg_no_border_clusters = len(sum(result2.values(),[]))/len(result2.keys())
            else:
                avg_no_border_clusters = 1

    return all_clusters, no_of_neighbours, point_type, cluster_id, avg_no_border_clusters

      
if __name__ == "__main__":
    
    # measure the time of reading the input file
    start_time_input = time.time()
    
    # A. Uncomment below for A dataset
    input_file = 'my_test0.csv'
    
    # B. Uncomment below for B dataset
    #input_file = 'my_test1.csv'
    
    # C. Uncomment below for C dataset
    #input_file = 'complex9.csv'
    
    # D. Uncomment below for D dataset
    #input_file = 'cluto-t7-10k.csv'
    
    # E. Uncomment below for E dataset
    #input_file = 'letter.csv'
    
    df = pd.read_csv(input_file, delimiter=',')
    end_time_input = time.time()
    elapsed_time_input = end_time_input - start_time_input
    
    # keep the true labels for RAND in separate value and drop them from
    # original data
    true_labels = df.iloc[:,-1:]
    df.drop(df.columns[-1], axis = 1, inplace = True)
    
    # data will be kept in X
    X = [list(row) for row in df.values]
    X = np.array(X)

    
    # ENTER - below enter the parameters got from Heuristic_of_epsilon.py
    Eps = 0.7
    MinPts = 4
    
    # measure the time of clustring
    start_time_cluster = time.time()     
    all_clusters, no_of_neighbours, point_type, cluster_id, \
        avg_no_border_clusters = dbscan(X, MinPts, Eps, cos_sim)
    end_time_cluster = time.time()
    elapsed_time_cluster = end_time_cluster - start_time_cluster
    
    
    # create columns for OUT csv file
    if len(X[1]) == 16:
        first_part = [[index, value[0], value[1], value[2], value[3], value[4], \
                    value[5], value[6], value[7], value[8], value[9], \
                         value[10], value[11], value[12], value[13], value[14], \
                             value[15]] for index, value in enumerate(X)]
        out = pd.DataFrame(first_part, columns = ['index', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16'])
    elif len(X[1]) == 2:
        first_part = [[index, value[0], value[1]]for index, value in enumerate(X)]
        out = pd.DataFrame(first_part, columns = ['index', 'd1', 'd2'])
    else: print("Valid number of dimensions are only 2 and 16")
    
    second_part = [value for index, value in enumerate(no_of_neighbours)]
    third_part = [value for index, value in enumerate(point_type)]
    fourth_part = [value for index, value in enumerate(cluster_id)]
    
    out["number of calculations"] = pd.Series(second_part)
    out["point type"] = pd.Series(third_part)
    out["Cids (may be more than one)"] = pd.Series(fourth_part)
    
    # measure the time of writing to OUT
    start_time_write_out = time.time()  
    
      
    # A. Uncomment below for A dataset
    out.to_csv('OUT_DBSCAN+_cosine_my_test0_D2_R8_m4_e0_7', index = False)
    
    # B. Uncomment below for B dataset
    #out.to_csv('OUT_DBSCAN+_cosine_my_test1_D2_R450_m5_e0_9998875', index = False)
    
    # C. Uncomment below for C dataset
    #out.to_csv('OUT_DBSCAN+_cosine_complex9_D2_R3031_m4_e0_999998', index = False)
    
    # D. Uncomment below for D dataset
    #out.to_csv('OUT_DBSCAN+_cosine_cluto_t7_10k_D2_R10000_m4_e0_9999995', index = False)
    
    # E. Uncomment below for E dataset
    #out.to_csv('OUT_DBSCAN+_cosine_letter_D16_R20000_m32_e0_9922', index = False)

    
    end_time_write_out = time.time()
    elapsed_time_write_out = end_time_write_out - start_time_write_out
    
     
    # below are Rand index modifications
    # noise points are treated as a special cluster with label 0 (don't confuse
    # it with border points - in labels there are no zeros)    
    true_labels = [int(row) for row in true_labels.values]
    true_labels_rand = [0 if x == -1 else x for x in true_labels]
    cluster_id_rand = [0 if x == -1 else x for x in cluster_id]
    
    
    # calculate the Rand parameters and Rand index
    tp, tn, no_of_points_pairs, \
    rand_index = rand_index_score(true_labels_rand, cluster_id_rand)
      
  
          
    # creating columns of STAT file
    stat = pd.DataFrame()
    stat["Name of the input file"] = pd.Series(input_file)
    stat["Number of dimensions of a point"] = pd.Series(len(X[1]))
    stat["Number of points in the input file"] = pd.Series(len(X))
    stat["Eps"] = pd.Series(Eps)
    stat["MinPts"] = pd.Series(MinPts)
    stat["Runtime of reading the INPUT"] = pd.Series(elapsed_time_input)
    stat["Runtime of clustering"] = pd.Series(elapsed_time_cluster)
    stat["Runtime of writing to OUT"] = pd.Series(elapsed_time_write_out)
    stat["Total runtime"] = pd.Series()
      
    temp_discovered_clusters = list(all_clusters.keys())
    if 'border' in temp_discovered_clusters:
        temp_discovered_clusters.remove('border')
    if 'noise' in temp_discovered_clusters:
        temp_discovered_clusters.remove('noise')
    no_discovered_clusters = len(temp_discovered_clusters)

    stat["Number of discovered clusters"] = pd.Series(no_discovered_clusters)
    
    no_noise_points = [x for x in point_type if x == -1]
    no_core_points = [x for x in point_type if x == 1]
    no_border_points = [x for x in point_type if x == 0]
    
    stat["Number of discovered noise points"] = pd.Series(len(no_noise_points))
    stat["Number of discovered core points"] = pd.Series(len(no_core_points))
    stat["Number of discovered border points"] = pd.Series(len(no_border_points))
    stat["Average number of calculations of cos_sim"] = pd.Series(counter/len(X))
    stat["Average number of clusters to which border points are assigned"] = pd.Series(avg_no_border_clusters)
    stat["Rand's TP"] = pd.Series(tp)
    stat["Rand's TN"] = pd.Series(tn)
    stat["Rand's number of points pairs"] = pd.Series(no_of_points_pairs)
    stat["Rand's index"] = pd.Series(rand_index)
    
    # measure the time of writing to STAT
    start_time_write_stat = time.time()  
    
    # A. Uncomment below for A dataset
    stat.to_csv('STAT_DBSCAN+_cosine_my_test0_D2_R8_m4_e0_7', index = False)
    
    # B. Uncomment below for B dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_my_test1_D2_R450_m5_e0_9998875', index = False)
    
    # C. Uncomment below for C dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_complex9_D2_R3031_m4_e0_999998', index = False)
    
    # D. Uncomment below for D dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_cluto_t7_10k_D2_R10000_m4_e0_9999995', index = False)
    
    # E. Uncomment below for E dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_letter_D16_R20000_m32_e0_9922', index = False)

    end_time_write_stat = time.time()
    elapsed_time_write_stat = end_time_write_stat - start_time_write_stat
        
    # measure the total runtime
    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    stat["Total runtime"] = pd.Series(elapsed_time_total)
    
    stat = stat.T
    
    # A. Uncomment below for A dataset
    stat.to_csv('STAT_DBSCAN+_cosine_my_test0_D2_R8_m4_e0_7', index = True, header = None, sep= '\t')
    
    # B. Uncomment below for B dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_my_test1_D2_R450_m5_e0_9998875', index = True, header = None, sep= '\t')
    
    # C. Uncomment below for C dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_complex9_D2_R3031_m4_e0_999998', index = True, header = None, sep= '\t')
    
    # D. Uncomment below for D dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_cluto_t7_10k_D2_R10000_m4_e0_9999995', index = True, header = None, sep= '\t')
    
    # E. Uncomment below for E dataset
    #stat.to_csv('STAT_DBSCAN+_cosine_letter_D16_R20000_m32_e0_9922', index = True, header = None, sep= '\t')


    
    # if you would like not to plot the results for 2D data, comment this section
    for key,values in all_clusters.items():
        plt.plot(X[values][:, 0], X[values][:, 1], 'o', markersize=5, label=key)
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.axis('equal')
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=5)
    plt.show()
    
    '''   
    print("Elapsed time input was %g seconds" % (elapsed_time_input))
    print("Elapsed time cluster was %g seconds" % (elapsed_time_cluster))
    print("Elapsed time write out was %g seconds" % (elapsed_time_write_out))
    print("Elapsed time write stat was %g seconds" % (elapsed_time_write_stat))
    print("Elapsed time was %g seconds" % (elapsed_time_total))
    '''



