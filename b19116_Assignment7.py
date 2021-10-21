#Smriti Srivastava
#b19116
#6388490594
#%%

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial
from scipy.spatial.distance import cdist

train = pd.read_csv('mnist-tsne-train.csv')
test = pd.read_csv('mnist-tsne-test.csv')

train_labels = train['labels']
test_labels = test['labels']
train.drop(['labels'] , axis = 1 , inplace = True)
test.drop(['labels'] , axis = 1 , inplace = True)
#%%
#Q1
print('Q1.')
print('Using K-means for clusterig of class')
K = 10
kmeans = KMeans(n_clusters=10)
#fitting the data k clusture
kmeans.fit(train)
#predicting the labels of train
kmeans_prediction = kmeans.predict(train)

#a
print('\na.')
#scatter plot for plotting data points in deifferent clusters
plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=kmeans_prediction , cmap=plt.cm.get_cmap( 'viridis',10))
#scatter plot to plotting the centre of the cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('K-means (K=10) clustering on the mnist-tsne training data')
plt.colorbar(plot)
plt.show()

#b
print('\nb.')
#function for purity score

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

#computing the purity score for training data
purity_train = purity_score(train_labels , kmeans_prediction)
print('Purity Score of training data = ', str(purity_train))

#c
print('\nc.')
#predicting the labels of test
kmeans_prediction_test = kmeans.predict(test)
#scatter plot for plotting data points in deifferent clusters
plot = plt.scatter(test["dimention 1"], test["dimention 2"],c=kmeans_prediction_test ,  cmap=plt.cm.get_cmap( 'viridis',10))
#scatter plot to plotting the centre of the cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='red')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('K-means (K=10) clustering on the mnist-tsne testing data')
plt.colorbar(plot)
plt.show()


#d
print('\nd')
#computing the purity score for testing data
purity_test = purity_score(test_labels , kmeans_prediction_test)
print('Purity Score of testing data = ', str(purity_test))

#%%
#Q2
print('\n\n\nQ2.')
print('Using GMM for clustering of class')
K = 10
gmm = GaussianMixture(n_components = K)
#fitting the data in the gmm clusters of k = 10
gmm.fit(train)
#predicting the labels of train
GMM_prediction_train = gmm.predict(train)
cluster_centre_gmm = pd.DataFrame(gmm.means_)

#a
print('\na')
#scatter plot for plotting data points in deifferent clusters
plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=GMM_prediction_train ,  cmap=plt.cm.get_cmap( 'viridis',10))
#scatter plot to plotting the centre of the cluster
plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], s=30, color="red")

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('GMM clustering on the mnist-tsne training data')
plt.colorbar(plot)
plt.show()


#b
print('\nb.')
#computing the purity score for train
gmm_purity_train = purity_score(train_labels , GMM_prediction_train)
print('Purity of training data = ', str(gmm_purity_train))

#c.
print('\nc.')
#predicting the labels of test
GMM_prediction_test = gmm.predict(test)

#scatter plot for plotting data points in deifferent clusters
plot = plt.scatter(test["dimention 1"], test["dimention 2"],c=GMM_prediction_test ,  cmap=plt.cm.get_cmap( 'viridis',10))
#scatter plot to plotting the centre of the cluster
plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], s=30, color="red")

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('GMM clustering on the mnist-tsne testing data')
plt.colorbar(plot)
plt.show()


#d
print('\nd.')
#computing the purity score for test
gmm_purity_test = purity_score(test_labels , GMM_prediction_test)
print('Purity of testing data = ', str(gmm_purity_test))

#%%
#Q3
print('\n\n\nQ3')
print('DBSCAN CLUSTRING')
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train)
DBSCAN_predictions = dbscan_model.labels_

#a
print('\na.')
#plotting train data in clusters using DBSCAN
plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=DBSCAN_predictions ,  cmap=plt.cm.get_cmap( 'viridis',10))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('DBSCAN clustering on the mnist-tsne training data')
plt.colorbar(plot)
plt.show()

#b
print('\nb.')

#computing purity of train after DBSCAN
db_purity_train = purity_score(train_labels , DBSCAN_predictions )
print('Purity Score of training data after DBSCAN = ', str(db_purity_train))


#c
print('\nc.')

#e = list(dbscan_model.core_sample_indices_)
#a = dbscan_model.components_

def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] =dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


DBSCAN_test_prediction = dbscan_predict(dbscan_model, np.array(test), metric=spatial.distance.euclidean)
#plotting test data in clusters using DBSCAN
plot = plt.scatter(test["dimention 1"], test["dimention 2"], c=DBSCAN_test_prediction ,  cmap=plt.cm.get_cmap( 'viridis',10))
plt.xlabel("x 1")
plt.ylabel("x 2")
plt.title("DBSCAN clustering on the mnist tsne test data ")
plt.colorbar(plot)
plt.show()


#d
print('\nd.')
#computing purity of test after DBSCAN
db_purity_test = purity_score(test_labels , DBSCAN_test_prediction)
print('Purity Score of testing data after DBSCAN = ' , str(db_purity_test))



#%%

#BONUS
print('BONUS')
print('\n\n\nPART A.')
k = [2,5,8,12,18,20]
print('KMean Clustering')
J = []
purity_train = []
purity_test = []
for j in range(len(k)):
    i = k[j]
    K = i
    print('K = ' , str(K))
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(train)
    kmeans_prediction = kmeans.predict(train)
    #scatter plot for plotting data points in deifferent clusters
    plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=kmeans_prediction , cmap=plt.cm.get_cmap( 'viridis',10))
    #scatter plot to plotting the centre of the cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('K-means clustering on the mnist-tsne training data')
    plt.colorbar(plot)
    plt.show()
    #Calculating Distortion
    J.append(sum(np.min(cdist(test,kmeans.cluster_centers_ , 'euclidean'),axis=1)) / train.shape[0]) 
    purity_train.append(purity_score(train_labels , kmeans_prediction))

    kmeans_prediction_test = kmeans.predict(test)
        #scatter plot for plotting data points in deifferent clusters
    plot = plt.scatter(test["dimention 1"], test["dimention 2"],c=kmeans_prediction_test , cmap=plt.cm.get_cmap( 'viridis',10))
    #scatter plot to plotting the centre of the cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('K-means clustering on the mnist-tsne testing data')
    plt.colorbar(plot)
    plt.show()
    purity_test.append(purity_score(test_labels , kmeans_prediction_test))

#creating dataframe of purity
data = {'K' : k, 'Purity Train' : purity_train , 'Purity Test': purity_test}
purity = pd.DataFrame(data, columns =['K','Purity Train', 'Purity Test'])
print(purity)
#plotting the elbow graph
plt.plot(k , J)
plt.xlabel('K')
plt.ylabel('Distortion measure (J)')
plt.title('K v.s Distortion')
plt.show()


print("Optimal value of K using K-Means =", 8)



print('\n\nGMM Clustering')
J = []
purity_train = []
purity_test = []
for j in range(len(k)):
    i = k[j]
    K = i
    print('K = ' , str(K))
    gmm = GaussianMixture(n_components = K)
    gmm.fit(train)
    GMM_prediction_train = gmm.predict(train)
    cluster_centre_gmm = pd.DataFrame(gmm.means_)
    #scatter plot for plotting data points in deifferent clusters
    plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=GMM_prediction_train ,  cmap=plt.cm.get_cmap( 'viridis',10))
    #scatter plot to plotting the centre of the cluster
    plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], s=30, color="red")
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GMM clustering on the mnist-tsne training data')
    plt.colorbar(plot)
    plt.show()
    
    #calculating distortion
    J.append(sum(np.min(cdist(test, gmm.means_, 'euclidean'),axis=1)) / train.shape[0])
    purity_train.append(purity_score(train_labels , GMM_prediction_train))
    GMM_prediction_test = gmm.predict(test)
    #scatter plot for plotting data points in deifferent clusters
    plot = plt.scatter(test["dimention 1"], test["dimention 2"],c=GMM_prediction_test ,  cmap=plt.cm.get_cmap( 'viridis',10))
    #scatter plot to plotting the centre of the cluster
    plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], s=30, color="red")
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GMM clustering on the mnist-tsne testing data')
    plt.colorbar(plot)
    plt.show()
    purity_test.append(purity_score(test_labels , GMM_prediction_test))

#creating dataframe of purity
data = {'K' : k, 'Purity Train' : purity_train , 'Purity Test': purity_test}
purity = pd.DataFrame(data, columns =['K','Purity Train', 'Purity Test'])
print(purity)
#plotting the elbow graph
plt.plot(k , J)
plt.xlabel('K')
plt.ylabel('Distortion measure (J)')
plt.title('K v.s Distortion')
plt.show()


print("Optimal value of K using GMM =", 12)



#%%
#PART B
print('\n\n\nPART B')
eps = [1, 5, 10]
min_samples = [10, 30, 50]

purity_train = []
purity_test = []
_e = []
_m = []
for e in eps:
    for m in min_samples:
        print('\n\n')
        print('eps = ', str(e) , ', min_samples = ' , str(m))
        _e.append(e)
        _m.append(m)
        dbscan_model=DBSCAN(eps=e, min_samples=m).fit(train)
        DBSCAN_predictions = dbscan_model.labels_
        
        plot = plt.scatter(train["dimention 1"], train["dimension 2"],c=DBSCAN_predictions ,  cmap=plt.cm.get_cmap( 'viridis',10))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('DBSCAN clustering on the mnist-tsne training data')
        plt.colorbar(plot)
        plt.show()
        #computing purity of train after DBSCAN
        purity_train.append(purity_score(train_labels , DBSCAN_predictions) )
        
        
        DBSCAN_test_prediction = dbscan_predict(dbscan_model, np.array(test), metric=spatial.distance.euclidean)

        plot = plt.scatter(test["dimention 1"], test["dimention 2"], c=DBSCAN_test_prediction ,  cmap=plt.cm.get_cmap( 'viridis',10))
        plt.xlabel("x 1")
        plt.ylabel("x 2")
        plt.title("DBSCAN clustering on the mnist tsne test data ")
        plt.colorbar(plot)
        plt.show()
        #computing purity of test after DBSCAN
        purity_test.append(purity_score(test_labels , DBSCAN_test_prediction))
        
#creating purity score table
data = {'esp': _e, 'Min_samples': _m ,  'Purity Train' : purity_train , 'Purity Test': purity_test}
purity = pd.DataFrame(data, columns =['esp','Min_samples','Purity Train', 'Purity Test'])
print(purity)