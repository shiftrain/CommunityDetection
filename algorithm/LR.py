# -*- coding: UTF-8 -*-

"""
Created on 18-3-5

@summary: LeaderRank 节点排序算法

@author: dreamhomes
"""
import networkx as nx
import string
import time
import collections
import numpy as np


# def leaderrank(G):
#     """
#     节点排序
#     :param graph:复杂网络图Graph
#     :return: 返回节点排序值
#     """
#     graph=G
#     # 节点个数
#     num_nodes = graph.number_of_nodes()
#     # 节点
#     nodes = graph.nodes()
#     # 在网络中增加节点g并且与所有节点进行连接
#     graph.add_node(num_nodes)
#     for node in nodes:
#         graph.add_edge(num_nodes, node)
#     # LR值初始化
#     LR = np.ones(num_nodes+1)
#     LR[num_nodes] = 0.0
#     # 迭代从而满足停止条件
#     while True:
#         tempLR = {}
#         for node1 in graph.nodes():
#             s = 0.0
#             for node2 in graph.nodes():
#                 if node2 in graph.neighbors(node1):
#                     s += 1.0 / graph.degree[node2] * LR[node2]
#             tempLR[node1] = s
#         # 终止条件:LR值不在变化
#         error = 0.0
#         for n in xrange(num_nodes+1):
#             error += abs(tempLR[n] - LR[n])
#         if error == 0.0:
#             break
#         LR = tempLR
#     # 节点g的LR值平均分给其它的N个节点并且删除节点
#     avg = LR[num_nodes] / num_nodes
#     LR.pop(num_nodes)
#     for k in LR:
#         LR[k] += avg
#
#     return LR
#

def leaderrank(graph):
    """
    节点排序
    :param graph:复杂网络图Graph
    :return: 返回节点排序值
    """
    # 节点个数
    num_nodes = graph.number_of_nodes()
    # 节点
    nodes = graph.nodes()
    # 在网络中增加节点g并且与所有节点进行连接
    graph.add_node(num_nodes)
    for node in nodes:
        graph.add_edge(num_nodes, node)
    # LR值初始化
    LR = np.ones(num_nodes+1)
    LR[num_nodes] = 0.0
    # 迭代从而满足停止条件
    iter_time=0
    while True:
        iter_time+=1
        tempLR = np.zeros(num_nodes+1)
        for node1 in graph.nodes():
            # s = 0.0
            x=[]
            for node2 in graph.neighbors(node1):
                x.append(LR[node2] / graph.degree[node2])
            tempLR[node1] = sum(x)
        # 终止条件:LR值不在变化
        e=tempLR-LR
        e = max(map(abs, e))
        if e<0.001:
            break
        LR = tempLR
    # 节点g的LR值平均分给其它的N个节点并且删除节点
    avg = LR[num_nodes] / num_nodes
    # LR.pop(num_nodes)
    for k in xrange(num_nodes):
        LR[k] += avg
    print iter_time
    print LR
    LRR=collections.defaultdict(int)
    for i in xrange(num_nodes):
        LRR[i]=LR[i]
    LRR = sort_by_value(LRR)
    return LRR


def load_graph(path,n):
    G=np.zeros((n,n))
    with open(path) as text:
        for line in text:
            # vertices = line.strip().split()
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            G[v_i][v_j] = 1
            G[v_j][v_i] = 1
    return G

def karate_club_graph(path,n):
    all_members = set(range(n))
    G = nx.Graph()
    G.add_nodes_from(all_members)
    # G.name = "Zachary's Karate Club"

    GG = load_graph(path,n)
    for x in xrange(n):
        for y in xrange(n):
            if GG[x][y]==1:
                G.add_edge(x, y)
    return G

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse = True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

if __name__ == "__main__":
    time_start = time.time()
    data_path = '../network/football.txt'
    maxnum = 0
    with open(data_path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            if v_i > maxnum or v_j > maxnum:
                maxnum = max(v_i, v_j)
    node_num = maxnum + 1

    G = karate_club_graph(data_path, node_num)
    print  leaderrank(G)
    time_end = time.time()
    print 'totally cost', time_end - time_start, 's'
