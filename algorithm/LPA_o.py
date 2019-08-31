import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import community
import math
import timeit
import numpy as np
import scipy as sp


def read_graph_from_file(path):
    # read edge-list from file
    graph = nx.read_edgelist(path, data = (('weight', float), ))

    # initial graph node's attribute 'label' with its id
    for node, data in graph.nodes(True):
        data['label'] = node

    return graph


def lpa(graph):
    def estimate_stop_cond():
        for node in graph.nodes():
            count = {}

            for neighbor in graph.neighbors(node):
                neighbor_label = graph.node[neighbor]['label']
                neighbor_weight = 1
                count[neighbor_label] = count.setdefault(neighbor_label, 0.0) + neighbor_weight

            count_items = count.items()
            count_items.sort(key = lambda x: x[1], reverse = True)

            labels = [k for k,v in count_items if v == count_items[0][1]]

            if graph.node[node]['label'] not in labels:
                return False

        return True

    loop_count = 0

    while True:
        loop_count += 1

        for node in graph.nodes():
            count = {}

            for neighbor in graph.neighbors(node):
                neighbor_label = graph.node[neighbor]['label']
                neighbor_weight = 1
                count[neighbor_label] = count.setdefault(neighbor_label, 0.0) + neighbor_weight

            # find out labels with maximum count
            count_items = count.items()
            count_items.sort(key = lambda x: x[1], reverse = True)

            labels = [(k, v) for k, v in count_items if v == count_items[0][1]]
            label = random.sample(labels, 1)[0][0]

            graph.node[node]['label'] = label

        if estimate_stop_cond() is True or loop_count >= 10:
            return

def revert_graph_info(graph):
    game_info = {}
    info = {}
    num=0
    result = {}

    for node, data in graph.nodes(True):
        info.setdefault(graph.node[node]['label'], []).append(game_info.get(node, node))

    for clazz in info:
        for label in info[clazz]:
            result[label] = num
        num=num+1

    return result


def calculate_NMI(clust1, clust2):

    #N matrix
    N = get_N_matrix(clust1, clust2)
    Ntotal = N.sum()

    #number of communities
    [Ca,Cb] = N.shape

    #upper equation - sum of i and j
    v_ij = 0
    for i in range(0, Ca):
        for j in range(0, Cb):

            #sum of N for i and j
            Ni = N[i,:].sum()
            Nj = N[:,j].sum()

            #to avoid log(0)
            if (N[i,j] != 0):
                #equation calculation
                v_ij = v_ij + (N[i,j] * math.log10( (N[i,j] * Ntotal) / (Ni * Nj) ))

    v_i = 0
    for i in range(0, Ca):
        Ni = N[i,:].sum()

        v_i = v_i + Ni * math.log10( Ni / Ntotal)

    v_j = 0
    for j in range(0, Cb):
        Nj = N[:,j].sum()

        v_j = v_j + Nj * math.log10( Nj / Ntotal)

    v = -2 * (v_ij / (v_i + v_j))
    return v


#gets the N matrix between two clusterings
def get_N_matrix(clust1, clust2):

    #the set of unique communities in clustering
    communitiesInClust1 = set(clust1.values())
    communitiesInClust2 = set(clust2.values())

    #initialize N matrix
    N = np.empty([len(communitiesInClust1), len(communitiesInClust2)])

    for i in communitiesInClust1:
        c1 = dict((key,value) for key, value in clust1.iteritems() if value == i).keys()

        for j in communitiesInClust2:
            c2 = dict((key,value) for key, value in clust2.iteritems() if value == j).keys()

            n=0

            for k in c2:
                for m in c1:
                    if k == m:
                        n = n + 1

            N[i,j] = n

    return N


if __name__ == '__main__':
    g = read_graph_from_file('../network/club.txt')

    #timer start c1
    startC1 = timeit.default_timer()

    lpa(g)

    #print run time for c1
    runtimeC1 = timeit.default_timer() - startC1

    cluster1 = revert_graph_info(g)
    # mod1 = community.modularity(cluster1, g)


    graph = nx.read_edgelist('../network/club.txt')

    #timer start c1
    startC2 = timeit.default_timer()

    #best partition calculation
    # cluster2 = community.best_partition(graph)

    #print run time for c1
    runtimeC2 = timeit.default_timer() - startC2

    # mod2 = community.modularity(cluster2, graph)



    graph = nx.read_edgelist('../network/club.txt')

    #timer start c1
    startC3 = timeit.default_timer()

    # tmp = community.generate_dendogram(graph)
    # cluster3 = community.partition_at_level(tmp, 0)

    #print run time for c1
    runtimeC3 = timeit.default_timer() - startC3

    # mod3 = community.modularity(cluster3, graph)

    # print "modularity:  1:%f;  2:%f;  3:%f" % (mod1,mod2, mod3)
    # nmi1 = calculate_NMI(cluster1, cluster2)
    # print "nmi between cluster1 and cluster 2: %.10f" % nmi1
    # nmi2 = calculate_NMI(cluster1, cluster3)
    # print "nmi between cluster1 and cluster 3: %.10f" % nmi2
    # nmi3 = calculate_NMI(cluster2, cluster3)
    # print "nmi between cluster2 and cluster 3: %.10f" % nmi3


    # print "Clustering Run Time: 1:%f; 2:%f; 3:%f" %(runtimeC1,runtimeC2,runtimeC3)