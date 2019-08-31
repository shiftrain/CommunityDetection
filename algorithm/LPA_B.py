# -*- coding: utf-8 -*-
#!/usr/bin/python
import collections
import random
import networkx as nx
import  numpy as np
import matplotlib.pyplot as plt
import string
import time
import math
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
# np.set_printoptions(threshold=np.inf)
try:  # Python 3.x
    import urllib.request as urllib
except ImportError:  # Python 2.x
    import urllib


'''
paper : <<Near linear time algorithm to detect community structures in large-scale networks>>
'''

class LPA():
    
    def __init__(self, G, max_iter = 20):
        self._G = G
        self._n = len(G.node) #number of nodes
        self._max_iter = max_iter

        
    def can_stop(self):
        # all node has the label same with its most neighbor
        for i in range(self._n):
            node = self._G.node[i]
            label = node["label"]
            max_labels = self.get_max_neighbor_label(i)
            if(label not in max_labels):
                return False
        return True
        
    def get_max_neighbor_label(self,node_index):
        m = collections.defaultdict(int)
        for neighbor_index in self._G.neighbors(node_index):
            neighbor_label = self._G.node[neighbor_index]["label"]
            if neighbor_label!=-1:
                m[neighbor_label] += 1
        if m:
            max_v = max(m.itervalues())
            return [item[0] for item in m.items() if item[1] == max_v]
        else:
            return [-1]



    '''asynchronous update'''

    def populate_label(self):
        # random visit
        # visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
        # if iter_time==1:
        visitSequence=pr
        # else:
        #     visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
        for i in visitSequence:
            node = self._G.node[i]
            label = node["label"]
            # print i,
            #找到邻居节点中最多的标签，并找到这些标签所在节点的节点序号

            labels_num = []#所有邻居节点的序号#可以简化代码，暂时懒得
            count = 0
            for neighbor_index in self._G.neighbors(i):#找到邻居节点的序号并记录到label_num[]
                count += 1
                labels_num.append(neighbor_index)
            max_labels = self.get_max_neighbor_label(i) #找到最多的标签

            max_labels_num=[]#从所有邻居节点中找到有和相同的的标签的节点max_labels[],并记录节点序号到max_labels_num中
            for x in xrange(len(max_labels)):
                for y in xrange(count):
                    if self._G.node[labels_num[y]]["label"]==max_labels[x]:
                        max_labels_num.append(labels_num[y])

            #S计算,直接在标签传播的过程中计算，只计算需要用的
            # for j in max_labels_num:
            #     # flag[j]=1
            #     T1 = 0.0
            #     T2 = 0.0
            #     T_1 = 0.0
            #     T_2 = 0.0
            #     for k in xrange(N):
            #         if A[i][k] == 1 and A[j][k] == 1:
            #             T1 += 1
            #         # if A[i][k]==1 or A[j][k]==1:
            #         #     T2+=1
            #         if A[i][k] == 1:
            #             T_1 += 1
            #         if A[j][k] == 1:
            #             T_2 += 1
            #     T2 = min(T_1, T_2)
            #     # jacc=T1
            #     jacc = T1 / T2
            #     S[i][j] = jacc
            #     S[j][i] = jacc##

            S0=[]
            for x in xrange(len(max_labels_num)):
                S0.append(S[max_labels_num[x]][i])
            if S0==[]:
                continue
            Smax=max(S0)
            Smax_index=S0.index(Smax)
            max_label_index=max_labels_num[Smax_index]
            newLabel=self._G.node[max_label_index]["label"]
            node["label"]=newLabel
            delta=0.9
            S[i][max_label_index]*=delta
            S[max_label_index][i]*=delta

            # Smin=min(S0)
            # Smin_index=S0.index(Smin)
            # min_label_index=max_labels_num[Smin_index]
            # # G.remove_edge(i,min_label_index)
            # S[i][min_label_index]=0
            # S[min_label_index][i]=0


    def get_communities(self):
        communities = collections.defaultdict(lambda:list())
        for node in self._G.nodes(True):
            label = node[1]["label"]
            communities[label].append(node[0])
        return communities.values()

    def execute(self):
        # initial label
        for i in range(self._n):
            self._G.node[i]["label"] = i
        global iter_time
        iter_time = 0
        # populate label
        while (not self.can_stop() and iter_time < self._max_iter):
            iter_time += 1
            self.populate_label()
            # ml=collections.defaultdict(int)
            # for node in self._G.nodes:
            #     label=self._G.node[node]["label"]
            #     ml[label]+=1
            # min_v = max(ml.itervalues())
            # min_v=[item[0] for item in ml.items() if item[1] == min_v]
            # for node in min_v:
            #     # self._G.node[node]["label"]=-1
            #     for nei_node in self._G.adj[node].keys():
            #         S[node][nei_node]*=0.5
            #         S[nei_node][node]*=0.5


        print "iter_time:",iter_time
        return self.get_communities()

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
    global LABEL
    LABEL=[]
    global A
    A=GG
    global S
    S=np.zeros((n,n))
    global N
    N=n
    # global flag
    # flag=np.zeros(n)

    # for i in xrange(n):        #优化前，暴力遍历矩阵，计算Sjaccard
    #     for j in xrange(i,n):
    #         T1=0.0
    #         T2=0.0
    #         T_1=0.0
    #         T_2=0.0
    #         for k in xrange(n):
    #             if A[i][k]==1 and A[j][k]==1:
    #                 T1+=1
    #             # if A[i][k]==1 or A[j][k]==1:
    #             #     T2+=1
    #             if A[i][k]==1:
    #                 T_1+=1
    #             if A[j][k]==1:
    #                 T_2+=1
    #         T2=min(T_1,T_2)
    #         # jacc=T1
    #         jacc=T1/T2
    #         S[i][j]=jacc
    #         S[j][i]=jacc

    # with open(path) as text:   #优化后（速度提升400+%？？？2s—>0.128s），遍历所有边，但是要区分有向图和无向图，无向图要Sij=Sji
    #     for line in text:
    #         # vertices = line.strip().split()
    #         vertices = line.strip().split()
    #         i = string.atoi(vertices[0])
    #         j = string.atoi(vertices[1])
    #         T1 = 0.0
    #         T2 = 0.0
    #         T_1 = 0.0
    #         T_2 = 0.0
    #         for k in xrange(n):
    #             if A[i][k] == 1 and A[j][k] == 1:
    #                 T1 += 1
    #             # if A[i][k]==1 or A[j][k]==1:
    #             #     T2+=1
    #             if A[i][k] == 1:
    #                 T_1 += 1
    #             if A[j][k] == 1:
    #                 T_2 += 1
    #         T2 = min(T_1, T_2)
    #         # jacc=T1
    #         jacc = T1 / T2
    #         S[i][j] = jacc
    #         S[j][i] = jacc


    for i in xrange(n):   #BEST#巧用nx 计算Sjaccard
        a_list=G.adj[i].keys()
        for j in a_list:
            b_list=G.adj[j].keys()
            ret_list = list((set(a_list).union(set(b_list)))^(set(a_list)^set(b_list)))
            T1=float(len(ret_list))
            # T1=float(T1-float((G.degree[i]*G.degree[j])/n))
            T2=float(len(nx.shortest_path(G, source=i, target=j)))
            # T2=float(len(a_list)+len(b_list))
            # T2=float(pow(len(a_list)*len(b_list),0.5))
            # T3=float(min(len(a_list),len(b_list)))
            jacc=T1/T2
            # jacc=T1
            # jacc=jacc/float(len(nx.shortest_path(G, source=i, target=j)))
            # jacc=float(jacc-float((G.degree[i]*G.degree[j])/n))
            S[i][j]=jacc
            S[j][i]=jacc



    return G

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse = False)
    return [ backitems[i][1] for i in range(0,len(backitems))]

def pagerank(path):

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
    alpha = 0.9
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

    # print('final result:', P_n)
    pr=collections.defaultdict(int)
    for i in xrange(N):
        pr[i]=P_n[i]

    pr=sort_by_value(pr)

    return pr
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
        if e<0.01:
            break
        LR = tempLR
    # 节点g的LR值平均分给其它的N个节点并且删除节点
    avg = LR[num_nodes] / num_nodes
    # LR.pop(num_nodes)
    graph.remove_node(num_nodes)
    for k in xrange(num_nodes):
        LR[k] += avg
    print iter_time
    print LR
    LRR=collections.defaultdict(int)
    for i in xrange(num_nodes):
        LRR[i]=LR[i]
    LRR = sort_by_value(LRR)
    return LRR
def Q1(comm, G):
    # 边的个数
    edges = G.edges()
    m = len(edges)
    # print 'm',m
    # 每个节点的度
    du = G.degree()
    # print 'du',du
    # 通过节点对（同一个社区内的节点对）计算
    ret = 0.0
    for c in comm:
        for x in c:
            for y in c:
                # 边都是前小后大的
                # 不能交换x，y，因为都是循环变量
                if x <= y:
                    if (x, y) in edges:
                        aij = 1.0
                    else:
                        aij = 0.0
                else:
                    if (y, x) in edges:
                        aij = 1.0
                    else:
                        aij = 0
                # print x,' ',y,' ',aij
                tmp = aij - du[x] * du[y] * 1.0 / (2 * m)
                # print du[x],' ',du[y]
                # print tmp
                ret = ret + tmp
                # print ret
                # print ' '
    ret = ret * 1.0 / (2 * m)
    # print 'ret ',ret

    return ret


if __name__ == '__main__':
# def MAIN():

    data_path= '../network/football.txt'
    data_path2= '../network/football_comm.dat'
    maxnum = 0
    with open(data_path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            if v_i > maxnum or v_j > maxnum:
                maxnum = max(v_i, v_j)
    node_num = maxnum + 1
    time_start = time.time()
    global pr
    pr=pagerank(data_path)
    G = karate_club_graph(data_path, node_num)
    # pr=leaderrank(G)
    # pr=collections.defaultdict(int)
    # for i in xrange(node_num):
    #     pr[i]=1-1.0/G.degree[i]
    # pr=sort_by_value(pr)
    # print pr
    algorithm = LPA(G)
    communities = algorithm.execute()
    # while 1:
    #     flag = 0
    #     for community in communities:
    #         if len(community)<=1:
    #             flag+=1
    #     if flag==0:
    #         break
    #     else:
    #         G = karate_club_graph(data_path, node_num)
    #         algorithm = LPA(G)
    #         communities = algorithm.execute()
    #         continue


    time_end = time.time()
    print 'totally cost', time_end - time_start, 's'

    colors=cnames.values()

    i=0
    for community in communities:
        print  community
        # print len(community)
        i+=1
    print i

    for j in xrange(node_num):
        print G.node[j]["label"],

    print '\n',

    # for j in xrange(node_num):
    #     print G.degree[j],
    #
    # print '\n'
    # print flag
    # print G.adj[0].keys()

    # G = nx.read_gml('../network/football.gml')
    pos = nx.spring_layout(G)
    for ii in xrange(i):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=communities[ii],
                               node_color=colors[ii],
                               node_size=150,
                               alpha=0.8)
    nx.draw_networkx_edges(G, pos,
                           width=0.5, alpha=0.8, edge_color='y')
    nx.draw_networkx_labels(G,pos,font_color='#050505',font_size=8)
    plt.show()


    Q=Q1(communities,G)
    print "Q  :",Q


    A=[]
    for node in xrange(node_num):
        A.append(G.node[node]["label"])
    # B=[1,1,1,1,1,1,1,1,2,2,1,1,1,1,2,2,1,1,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2]
    B0=collections.defaultdict(int)
    with open(data_path2) as text:
        for line in text:
            vertices = line.strip().split()
            node=string.atoi(vertices[0])
            label = string.atoi(vertices[1])
            B0[node]=label
    # print B0.values()
    print "NMI:",metrics.normalized_mutual_info_score(A,B0.values())
    print "ARI:",metrics.adjusted_rand_score(A,B0.values())