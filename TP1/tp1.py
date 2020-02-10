#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Turn interactive plotting off
plt.ioff()

#########################################################################
# 1 - Analyse descriptive des données

# read input text and put data inside a data frame
fruits = pd.read_table('data/fruit_data_with_colors.txt')
print(fruits.head())

# print nb of instances and features
print(fruits.shape)

# print feature types
print(fruits.dtypes)

# print balance between classes
print(fruits.groupby('fruit_name').size())

# plot correlation between attributes w.r.t. classification
from pandas.plotting import scatter_matrix
from matplotlib import cm

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
# TODO
y = fruits['fruit_label']

fig = plt.figure()
cmap = cm.get_cmap('gnuplot')

scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fig/fruits_scatter_matrix')
plt.close(fig)

# print histogram for each attribute with belonging to classes
for attr in feature_names:
    fig = plt.figure()
    pd.DataFrame({k: v for k, v in fruits.groupby('fruit_name')[attr]}).plot.hist(stacked=True)
    plt.suptitle(attr)
    plt.savefig('fig/fruits_histogram_'+attr)
    plt.close(fig)



#########################################################################
# 2 - Prétraitement

attr='mass'

# discretize with equal-intervaled bins
fig = plt.figure()
plt.subplot(211)
matplotlib.pyplot.xticks(fontsize=6)
pd.cut(fruits[attr],10).value_counts(sort=False).plot.bar()
plt.xticks(rotation=25)
# discretize with equal-sized bins
plt.subplot(212)
matplotlib.pyplot.xticks(fontsize=6)
# TODO: plot with qcut
pd.qcut(fruits[attr],10).value_counts(sort=False).plot.bar()
plt.xticks(rotation=25)
plt.suptitle('Histogram for '+attr+' discretized with equal-intervaled and equal-sized bins')
plt.savefig('fig/'+attr+'_histogram_discretization')
plt.close(fig)



#########################################################################
# 3 - Cluster

# # Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = fruits[feature_names]
y = fruits['fruit_label']
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# # kmeans
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# # Plot clusters
lst_kmeans = [KMeans(n_clusters=n) for n in range(3,6)]
titles = [str(x)+' clusters' for x in range(3,6)]
fignum = 1
for kmeans in lst_kmeans:
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    kmeans.fit(X_norm)
    labels = kmeans.labels_
    ax.scatter(X['mass'], X['width'], X['color_score'],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('mass')
    ax.set_ylabel('width')
    ax.set_zlabel('color_score')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    plt.savefig('fig/k-means_'+str(2+fignum)+'_clusters')
    fignum = fignum + 1
    plt.close(fig)

# # Plot the ground truth
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for label in fruits['fruit_name'].unique():
    ax.text3D(fruits.loc[fruits['fruit_name']==label].mass.mean(),
              fruits.loc[fruits['fruit_name']==label].width.mean(),
              fruits.loc[fruits['fruit_name']==label].color_score.mean(),
              label,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
ax.scatter(X['mass'], X['width'], X['color_score'], c=y, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('mass')
ax.set_ylabel('width')
ax.set_zlabel('color_score')
ax.set_title('Ground Truth')
ax.dist = 12
plt.savefig('fig/k-means_ground_truth')
plt.close(fig)


# Compute R-square, i.e. V_inter/V
from R_square_clustering import r_square
from purity import purity_score

# Plot elbow graphs for KMeans using R square and purity scores
lst_k=range(2,11)
lst_rsq = []
lst_purity = []
for k in lst_k:
    est=KMeans(n_clusters=k)
    est.fit(X_norm)
    lst_rsq.append(r_square(X_norm.to_numpy(), est.cluster_centers_,est.labels_,k))
    lst_purity.append(purity_score(y.to_numpy(), est.labels_))
    
fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.plot(lst_k, lst_purity, 'rx-')
plt.xlabel('k')
plt.ylabel('RSQ/purity score')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('fig/k-means_elbow_method')
plt.close()
    


# # hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage

lst_labels = map(lambda pair: pair[0]+str(pair[1]), zip(fruits['fruit_name'].values,fruits.index))
linkage_matrix = linkage(X_norm, 'ward')
fig = plt.figure()
dendrogram(
    linkage_matrix,
    color_threshold=0,
    labels=list(lst_labels),
)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.savefig('fig/hierarchical-clustering')
plt.close()


#########################################################################
# 4 - Classement

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Create Training and Test Sets and Apply Scaling
# by default test data represents 25%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression()

lst_classif = [dummycl, gmb, dectree, logreg]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']

for clf,name_clf in zip(lst_classif,lst_classif_names):
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of '+name_clf+' classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
    print(confusion_matrix(y_test, y_pred))

print('0 : ' + fruits[fruits['fruit_label'] == 0].iloc[0, 1])
print('1 : ' + fruits[fruits['fruit_label'] == 1].iloc[0, 1])
print('2 : ' + fruits[fruits['fruit_label'] == 2].iloc[0, 1])
print('3 : ' + fruits[fruits['fruit_label'] == 3].iloc[0, 1])

# print decision tree
import graphviz
dot_data = tree.export_graphviz(dectree, out_file=None, 
                                feature_names=feature_names,  
                                class_names=fruits['fruit_name'].unique(),  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render(directory='fig',filename='decision_tree')


# # Supervised learning
# # cross-validation
from sklearn.model_selection import cross_val_score

for clf,name_clf in zip(lst_classif,lst_classif_names):
    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




#########################################################################
# 5 - Classement et discrétisation

list_prefix = ['eqsized_bins_', 'eqintervaled_bins_']
nb_bin = 10
for prefix in list_prefix:
    print("###### Discretization with "+prefix+" ######")
    
    for attr in feature_names:
        if 'sized' in prefix:
            fruits[prefix+attr]=pd.qcut(fruits[attr],nb_bin)
        else:
            fruits[prefix+attr]=pd.cut(fruits[attr],nb_bin)
        # use pd.concat to join the new columns with your original dataframe
        fruits=pd.concat([fruits,pd.get_dummies(fruits[prefix+attr],prefix=prefix+attr)],axis=1)
        # now drop the original column (you don't need it anymore)
        fruits.drop(prefix+attr,axis=1, inplace=True)

    feature_names_bins = filter(lambda x: x.startswith(prefix) and x.endswith(']'), list(fruits))
    X_discret = fruits[feature_names_bins]
    print(X_discret.head())

    # TODO : compute accuracies using cross validation with 4 classifiers
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X_discret, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    


#########################################################################
# 6 - ACP et sélection de variables

from sklearn.decomposition import PCA

acp = PCA(svd_solver='full')
coord = acp.fit_transform(X_norm)

# nb of computed components
print(acp.n_components_) 

# explained variance scores
print(acp.explained_variance_ratio_)

# plot eigen values
n = np.size(X_norm, 0)
p = np.size(X_norm, 1)
eigval = float(n-1)/n*acp.explained_variance_
fig = plt.figure()
plt.plot(np.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.savefig('fig/acp_eigen_values')
plt.close(fig)

# print eigen vectors
print(acp.components_)
# lines: factors
# columns: variables

# print correlations between factors and original variables
sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((p,p))
for k in range(p):
    corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
print(corvar)
# lines: variables
# columns: factors


# plot instances on the first plan (first 2 factors)
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
for i in range(n):
    plt.annotate(y.values[i],(coord[i,0],coord[i,1]))
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.savefig('fig/acp_instances_1st_plan')
plt.close(fig)



from sklearn.model_selection import cross_val_score
from sklearn import metrics

lst_classif = [dummycl, gmb, dectree, logreg]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']
print('*** Results for first 2 factors of ACP ***')

for clf,name_clf in zip(lst_classif,lst_classif_names):
    scores = cross_val_score(clf, coord[:,:2], y, cv=5)
    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('*** Results for first 2 original variables ACP ***')

for clf,name_clf in zip(lst_classif,lst_classif_names):
    scores = cross_val_score(clf, X_norm.iloc[:, :2], y, cv=5)
    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # Variable selection
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFECV(estimator=estimator, cv=5)
selector.fit(X_norm, y)
print("Optimal number of features: %d" % selector.n_features_)
print(selector.ranking_)

print(X_norm.head())

X_feature_trimmed = selector.transform(X_norm)

# print('*** Results for the 2 best original variables ACP ***')
for clf,name_clf in zip(lst_classif,lst_classif_names):
    scores = cross_val_score(clf, X_feature_trimmed, y, cv=5)
    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
