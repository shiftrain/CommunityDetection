import collections
import random
import networkx as nx
import  numpy as np
import matplotlib.pyplot as plt
import string
import time
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
        visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
        for i in visitSequence:
            node = self._G.node[i]
            label = node["label"]
            max_labels = self.get_max_neighbor_label(i)
            if (label not in max_labels):
                newLabel = random.choice(max_labels)
                node["label"] = newLabel
    # def populate_label(self,G):
    #     #random visit
    #     visitSequence = random.sample(G._G.nodes(),len(G._G.nodes()))
    #
    #     # max_labels = self.get_max_neighbor_label(1)
    #     for i in visitSequence:
    #         node = G._G.node[i]
    #         label = node["label"]
    #         max_labels = G.get_max_neighbor_label(i)
    #         if(label not in max_labels):
    #             newLabel = random.choice(max_labels)
    #             node["label"] = newLabel
    #     return G

    # def populate_label2(self, G):
    #     # random visit
    #     visitSequence = random.sample(G._G.nodes(), len(G._G.nodes()))
    #
    #     # max_labels = self.get_max_neighbor_label(1)
    #     for i in visitSequence:
    #         node = G._G.node[i]
    #         label = node["label"]
    #         max_labels = G.get_max_neighbor_label(i)
    #         if (label not in max_labels):
    #             newLabel = random.choice(max_labels)
    #             node["label"] = newLabel
    #     return G
    def get_communities(self):
        communities = collections.defaultdict(lambda:list())
        for node in self._G.nodes(True):
            label = node[1]["label"]
            communities[label].append(node[0])
        return communities.values()

    def execute(self):
        # initial label
        # for i in range(self._n):
        #     self._G.node[i]["label"] = i
        iter_time = 0
        # populate label
        while (not self.can_stop() and iter_time < self._max_iter):
            self.populate_label()
            iter_time += 1
        return self.get_communities()
    #
    # def execute(self):
    #     #initial label
    #     for i in range(self._n):
    #         self._G.node[i]["label"] = i
    #     iter_time = 0
    #     G1 = self.populate_label2(self)
    #     G2 = G1
    #     #populate label
    #     while(not self.can_stop() and iter_time<self._max_iter):
    #         if iter_time%2==0:
    #             G1 = self.populate_label2(G1)
    #         if iter_time%2!=0:
    #             G2 = self.populate_label2(G2)
    #         iter_time += 1
    #     return self.get_communities()

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


omg_len=0
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
    '''1'''
    dg = []
    omg = []
    for i in xrange(n):
        dg.append(G.degree[i])
    for i in xrange(n):
        G.node[i]["label"]=-1
    while True:
        max_dg_index=dg.index(max(dg))
        omg.append(max_dg_index)
        dg[max_dg_index]=0
        for j in xrange(n):
            if GG[max_dg_index][j]==1:
                dg[j]=0
        SUM=sum(dg)
        if SUM==0:
            break

    '''2'''
    # dg = []
    # omg = []
    # t = 10  # *3
    # for i in xrange(n):
    #     dg.append(G.degree[i])
    # for i in xrange(n):
    #     G.node[i]["label"] = -1
    # while t:
    #     max_dg_index = dg.index(max(dg))
    #     omg.append(max_dg_index)
    #     dg[max_dg_index] = 0
    #     t -= 1
    #


    '''3'''
    # k=5
    # dg=[0 for i in xrange(n)]
    # omg = []
    #
    # for i in xrange(n):
    #     for j in xrange(n):
    #         if GG[i][j]==1:
    #            dg[i]+=1
    #
    #
    #
    # for i in xrange(n):
    #     G.node[i]["label"] = -1
    #
    # for k in range(0,5):
    #     while True:
    #         for i in xrange(n):
    #             if dg[i] == k:
    #                 for j in xrange(n):
    #                     GG[i][j] = 0
    #                 dg = [0 for i in xrange(n)]
    #                 for p in xrange(n):
    #                     for q in xrange(n):
    #                         if GG[p][q] == 1:
    #                             dg[p] += 1
    #         flag=0
    #         for i in xrange(n):
    #             if dg[i]==k:
    #                 flag=1
    #         if flag==0:
    #             break
    # for i in xrange(n):
    #     if dg[i]!=0:
    #         omg.append(i)
    #
    # omg=random.sample(omg,len(omg))





    print omg
    omg_len=len(omg)
    for i in xrange(len(omg)):
        G.node[i]["label"]=omg[i]




    # for row, line in enumerate(zacharydat.split('\n')):
    #     thisrow = [int(b) for b in line.split()]
    #     for col, entry in enumerate(thisrow):
    #         if entry == 1:
    #             G.add_edge(row, col)
    return G

# def draw_networkx_edges(G, pos,
#                         edgelist=None,
#                         width=1.0,
#                         edge_color='k',
#                         style='solid',
#                         alpha=1.0,
#                         edge_cmap=None,
#                         edge_vmin=None,
#                         edge_vmax=None,
#                         ax=None,
#                         arrows=True,
#                         label=None,
#                         **kwds):
#
# def draw_networkx_nodes(G, pos,
#                         nodelist=None,
#                         node_size=300,
#                         node_color='r',
#                         node_shape='o',
#                         alpha=1.0,
#                         cmap=None,
#                         vmin=None,
#                         vmax=None,
#                         ax=None,
#                         linewidths=None,
#                         label=None,
#                         **kwds):
#
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


if __name__ == '__main__':
    time_start = time.time()
    # G = karate_club_graph('../network/football.txt',115)
    G={}
    algorithm={}
    communities={}
    n=5
    nodo_num=115
    data_path='../network/club.txt'
    for g in xrange(n):
        G[g]=karate_club_graph(data_path,nodo_num)

    for alg in xrange(n):
        algorithm[alg]=LPA(G[alg])

    for com in xrange(n):
        communities[com]=algorithm[com].execute()

    time_end = time.time()
    print 'totally cost', time_end - time_start, 's'

    colors=['tab:blue', 'tab:orange', 'tab:green',
      'tab:purple', 'tab:brown', 'tab:pink',
     'tab:gray', 'tab:olive', 'tab:cyan', '#800080','tab:red','#EE82EE','#D2B48C','#A0522D','#7FFF00']
    c=[]
    cnum=0
    for ite in xrange(n):
        i=0
        # if len(communities[ite])<=1:
        #     continue
        c.append(ite)
        cnum+=1
        for community in communities[ite]:
            print community
            i = i + 1
        print  i

    i=0
    for i in xrange(cnum):
        for j in xrange(nodo_num):
            print  ("%2d"%(G[c[i]].node[j]["label"])),
        print '\n',
    i=0
    j=0
    comlabel=[]
    for j in xrange(nodo_num):
        m = collections.defaultdict(int)
        for i in xrange(cnum):
            max_label = G[c[i]].node[j]["label"]
            m[max_label] += 1
        max_v = max(m.itervalues())
        max_labelx=random.choice([item[0] for item in m.items() if item[1] == max_v])
        comlabel.append(max_labelx)
    print '\n',
    print  comlabel
    G = karate_club_graph(data_path,nodo_num)
    for up in xrange(nodo_num):
        G.node[up]["label"] = comlabel[up]
    algorithm = LPA(G)
    communities = algorithm.get_communities()
    print '\n',

    i=0
    for community in communities:
        print  community
        i+=1
    print i

    pos = nx.spring_layout(G)
    for ii in xrange(i):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=communities[ii],
                               node_color=colors[ii],
                               node_size=288,
                               alpha=0.9)
    nx.draw_networkx_edges(G, pos,
                           width=0.5, alpha=0.8, edge_color='y')
    nx.draw_networkx_labels(G,pos)
    plt.show()

    # G = karate_club_graph('../network/club.txt', 34)
    # cmnt = collections.defaultdict(lambda: list())
    # for nd in xrange(34):
    #     label=G.node[nd]["label"]
    #     cmnt[label].append(G.node[0])
    # cmnt_value=cmnt.values()
    #
    # cmnt = collections.defaultdict(lambda: list())
    # for node in G.nodes(True):
    #     label = node[1]["label"]
    #     cmnt[label].append(node[0])
    #
    # for cmn in cmnt_value:
    #     print cmn




    # for node in community:
        #     node.

    # for i in range(0, 33):
    #     print G1.node[i]["label"],' ',
    # print '\n'
    #
    #
    # for i in range(0, 33):
    #     print G2.node[i]["label"],' ',
    # print '\n'
    #
    #
    # for i in range(0, 33):
    #     print G3.node[i]["label"],' ',
    # print '\n'
    #
    # for i in range(0, 33):
    #     print G4.node[i]["label"],' ',
    # print '\n'
    #
    # for i in range(0, 33):
    #     print G5.node[i]["label"],' ',
    # print '\n'

    # G=load_graph('../network/club.txt',34)
    # print G
    # nx.drawing(G)

    # nx.draw(G,pos = nx.spring_layout(G),node_color = colors,with_labels = True)
    # plt.show()

    # G = nx.Graph()
    # v = range(1, 8)
    # G.add_nodes_from(v)
    # line = file_color.read()
    # colors = (line.split(' '))
    # for i in range(len(colors)):
    #      colors[i] = int(colors[i])
    # for x in range(0, len(matrix)):
    #      for y in range(0, len(matrix)):
    #           if matrix[x][y] == '1':
    #             G.add_edge(x, y)


    #
    # nx.draw(G, pos=nx.spring_layout(G),with_labels=True, node_color=['r','b','y','w','k'])
    # plt.show()





