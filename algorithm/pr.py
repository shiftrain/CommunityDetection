# -*- coding: utf-8 -*-
#!/usr/bin/python
# 输入为一个*.txt文件，例如
# A B
# B C
# B A
# ...表示前者指向后者

import numpy as np
import collections
import time

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse = True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

def pagerank(path):
    # time_start = time.time()
    # 读入有向图，存储边
    f = open(path, 'r')
    edges = [line.strip('\n').split(' ') for line in f]
    # print(edges)

    # 根据边获取节点的集合
    nodes = []
    for edge in edges:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    # print(nodes)

    N = len(nodes)

    # 将节点符号（字母），映射成阿拉伯数字，便于后面生成A矩阵/S矩阵
    i = 0
    node_to_num = {}
    for node in nodes:
        node_to_num[node] = i
        i += 1
    for edge in edges:
        edge[0] = node_to_num[edge[0]]
        edge[1] = node_to_num[edge[1]]
    # print(edges)

    # 生成初步的S矩阵
    S = np.zeros([N, N])
    for edge in edges:
        S[edge[1], edge[0]] = 1
    # print(S)

    # 计算比例：即一个网页对其他网页的PageRank值的贡献，即进行列的归一化处理
    # sum_of_col = sum(S)
    for j in range(N):
        sum_of_col = sum(S[:,j])
        for i in range(N):
            if sum_of_col==0:
                S[i,j]=0
            else:
                S[i, j] /= sum_of_col
    # print(S)

    # 计算矩阵A
    alpha = 0.85
    A = alpha*S + (1-alpha) / N * np.ones([N, N])
    # print(A)

    # 生成初始的PageRank值，记录在P_n中，P_n和P_n1均用于迭代
    P_n = np.ones(N) / N
    P_n1 = np.zeros(N)

    e = 100000  # 误差初始化
    k = 0   # 记录迭代次数
    # print('loop...')

    while e > 0.00000001:   # 开始迭代
        P_n1 = np.dot(A, P_n)   # 迭代公式
        e = P_n1-P_n
        e = max(map(abs, e))    # 计算误差
        P_n = P_n1
        k += 1
        # print('iteration %s:'%str(k), P_n1)
    print k
    # print('final result:', P_n)
    pr=collections.defaultdict(int)
    for i in xrange(N):
        pr[i]=P_n[i]
    print pr
    maxpr=max(pr.itervalues())
    mp=[item[0] for item in pr.items() if item[1] == maxpr]
    print mp
    pr=sort_by_value(pr)
    print pr
    # time_end = time.time()
    # print 'totally cost', time_end - time_start, 's'

if __name__ == '__main__':
    time_start = time.time()
    pagerank('../network/club.txt')
    time_end = time.time()
    print 'totally cost', time_end - time_start, 's'
