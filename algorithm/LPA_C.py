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
            m[neighbor_label] += 1
        max_v = max(m.itervalues())
        return [item[0] for item in m.items() if item[1] == max_v]



    '''asynchronous update'''

    # def populate_label(self):
    #     # random visit
    #     visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
    #     for i in visitSequence:
    #         node = self._G.node[i]
    #         label = node["label"]
    #         max_labels = self.get_max_neighbor_label(i)
    #         max_labels2=max_labels
    #         if (label not in max_labels):
    #             m=[]
    #             for i in xrange(len(max_labels)):
    #                 m.append(self._G.degree[max_labels[i]])
    #             m=random.sample(m,len(m))
    #             max_labels_index=m.index(max(m))
    #             max_label=max_labels2[max_labels_index]
    #             newLabel = max_label
    #             node["label"] = newLabel
    def populate_label(self):
        #random visit
        visitSequence = random.sample(self._G.nodes(),len(self._G.nodes()))
        for i in visitSequence:
            node = self._G.node[i]
            label = node["label"]
            max_labels = self.get_max_neighbor_label(i)
            if(label not in max_labels):
                newLabel = random.choice(max_labels)
                node["label"] = newLabel

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
        iter_time = 0
        # populate label
        while (not self.can_stop() and iter_time < self._max_iter):
            self.populate_label()
            iter_time += 1
        print "iter_time:",iter_time
        return self.get_communities()


def load_graph(path,n):
    G=np.zeros((n,n))
    with open(path) as text:
        for line in text:
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

cnames = {
'aliceblue':            '#F0F8FF',
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
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
# 'black':                '#000000',
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
    time_start = time.time()

    data_path='../network/facebook_combined.txt'

    '''get_node_num'''
    maxnum=0
    with open(data_path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            if v_i>maxnum or v_j>maxnum:
                maxnum=max(v_i,v_j)
    nodo_num=maxnum+1



    G=karate_club_graph(data_path,nodo_num)

    algorithm=LPA(G)

    communities=algorithm.execute()



    time_end = time.time()
    print 'totally cost', time_end - time_start, 's'

    colors = cnames.values()

    i=0
    for community in communities:
        print  community
        # print len(community)
        i+=1
    print i

    # for j in xrange(nodo_num):
    #     print G.node[j]["label"],
    #
    # print '\n',
    #
    # for j in xrange(nodo_num):
    #     print G.degree[j],

    # pos = nx.spring_layout(G)
    # for ii in xrange(i):
    #     nx.draw_networkx_nodes(G, pos,
    #                            nodelist=communities[ii],
    #                            node_color=colors[ii],
    #                            node_size=150,
    #                            alpha=0.8)
    # nx.draw_networkx_edges(G, pos,
    #                        width=0.5, alpha=0.8, edge_color='y')
    # nx.draw_networkx_labels(G,pos,font_color='#050505',font_size=8)
    # plt.show()

    Q=Q1(communities,G)
    print Q