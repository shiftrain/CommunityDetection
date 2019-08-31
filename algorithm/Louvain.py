# coding=utf-8

import collections
import random
import networkx as nx
import  numpy as np
import matplotlib.pyplot as plt
import string
import time
import pos
from openpyxl import load_workbook
'''
    paper : <<Fast unfolding of communities in large networks>>
'''

def load_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            G[v_i][v_j] = 1.0
            G[v_j][v_i] = 1.0
    return G



class Vertex():

    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in  #结点内部的边的权重

class Louvain():

    def __init__(self, G):
        self._G = G
        self._m = 0 #边数量
        self._cid_vertices = {} #需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}   #需维护的关于结点的信息(结点编号，相应的Vertex实例)
        for vid in self._G.keys():
            self._cid_vertices[vid] = set([vid])
            self._vid_vertex[vid] = Vertex(vid, vid, set([vid]))
            self._m += sum([1 for neighbor in self._G[vid].keys() if neighbor>vid])

    def first_stage(self):
        mod_inc = False  #用于判断算法是否可终止
        visit_sequence = self._G.keys()
        visit_sequence=random.sample(visit_sequence,len(visit_sequence))
        # random.shuffle(visit_sequence)
        while True:
            can_stop = True #第一阶段是否可终止
            for v_vid in visit_sequence:
                v_cid = self._vid_vertex[v_vid]._cid
                k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin
                cid_Q = {}
                for w_vid in self._G[v_vid].keys():
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        tot = sum([sum(self._G[k].values())+self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        k_v_in = sum([v for k,v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        delta_Q = k_v_in - k_v * tot / self._m  #由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        cid_Q[w_cid] = delta_Q

                cid,max_delta_Q = sorted(cid_Q.items(),key=lambda item:item[1],reverse=True)[0]
                if max_delta_Q > 0.0 and cid!=v_cid:

                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
        return mod_inc

    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid,vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                for k,v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v/2.0
            cid_vertices[cid] = set([cid])
            vid_vertex[cid] = new_vertex

        G = collections.defaultdict(dict)
        for cid1,vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2,vertices2 in self._cid_vertices.items():
                if cid2<=cid1 or len(vertices2)==0:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    for k,v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight

        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G


    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(c)
        return communities

    def execute(self):
        iter_time = 1
        while True:
            iter_time += 1
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()



def load_graphx(path,n):
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

    zacharydat = load_graphx(path,n)

    for x in xrange(n):
        for y in xrange(n):
            if zacharydat[x][y]==1:
                G.add_edge(x, y)

    return  G


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
# 'yellow':               '#FFFF00',
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
# def MAIN():
    time_start = time.time()

    path_load = '../network/facebook_combined.txt'


    maxnum = 0
    with open(path_load) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = string.atoi(vertices[0])
            v_j = string.atoi(vertices[1])
            if v_i > maxnum or v_j > maxnum:
                maxnum = max(v_i, v_j)
    node_num = maxnum + 1

    G1 = load_graph(path_load)
    algorithm = Louvain(G1)
    communities = algorithm.execute()
    time_end = time.time()
    G = karate_club_graph(path_load, node_num)
    i=0
    print 'totally cost', time_end - time_start, 's'

    colors =cnames.values()

    for community in communities:
        print community
        i=i+1
    print  i

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
    #
    # plt.show()

    Q=Q1(communities,G)
    print Q