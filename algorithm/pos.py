import networkx as nx
import Louvain
import LPA_B
import matplotlib as plt
def pos(G):
        pos = nx.spring_layout(G)
        return pos

if __name__ == '__main__':
        plt.figure(2)
        Louvain.MAIN()
        LPA_B.MAIN()
