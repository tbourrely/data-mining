#!/usr/bin/env python
# -*- coding: utf-8 -*-

#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis,corvar,plt):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    ax = plt.axes()

    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        # plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
        plt.annotate(df.columns[j],(corvar[j,x_axis],corvar[j,y_axis]))
        plt.plot(corvar[j,x_axis],corvar[j,y_axis], 'ro')
        ax.arrow(0, 0, corvar[j,x_axis], corvar[j,y_axis], head_width=0.05, head_length=0.05, length_includes_head=True)
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)



# print centroids associated with several countries
# lst_countries=[]
# centroid of the entire dataset
# est: KMeans model fit to the dataset
# print est.cluster_centers_
# for name in lst_countries:
#     num_cluster = est.labels_[y.loc[y==name].index][0]
#     print 'Num cluster for '+name+': '+str(num_cluster)
#     print '\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values)
#     print '\tcentroid: '+str(est.cluster_centers_[num_cluster])

