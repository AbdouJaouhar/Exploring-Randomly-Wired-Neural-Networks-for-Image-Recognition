import numpy as np
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib as mpl
mpl.use('agg')
import torch
import torch.nn as nn

import networkx as nx
from torchsummary import summary

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

plt.rcParams["figure.figsize"] = (20,10)
sns.set()
 
from WattsStrogatz import *



def plot_graph_one(graph):
    node_color = "#FFE057"
    edge_color = "#767CE8"
    font_size = 18
    G = nx.DiGraph()

    G.add_nodes_from(list(graph.nodes.keys()))
    for n, e in graph.edges.items():
        for u in e:
            G.add_edge(n, u)
    pos = nx.layout.spring_layout(G)

    node_sizes = [1000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [2 for i in range(2, M + 2)]
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=20, edge_color=edge_color, width=2, connectionstyle='arc3, rad=0.1')
    labels=nx.draw_networkx_labels(G,pos=pos,  font_size=font_size, font_color='w')


    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

def plot_graph(graph):
    node_color = "#FFE057"
    edge_color = "#767CE8"
    font_size = 18
    G = nx.DiGraph()

    G.add_nodes_from(list(graph.nodes.keys()))
    for n, e in graph.edges.items():
        for u in e:
            G.add_edge(n, u)
    pos = nx.layout.spring_layout(G)

    node_sizes = [1000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [2 for i in range(2, M + 2)]
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=20, edge_color=edge_color, width=2, connectionstyle='arc3, rad=0.1')
    labels=nx.draw_networkx_labels(G,pos=pos,  font_size=font_size, font_color='w')


    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    graph.ClockwiseRewiring()

    G = nx.DiGraph()

    G.add_nodes_from(list(graph.nodes.keys()))
    for n, e in graph.edges.items():
        for u in e:
            G.add_edge(n, u)
    pos = nx.layout.spring_layout(G, pos=pos, fixed=pos.keys())

    node_sizes = [1000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [2 for i in range(2, M + 2)]
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=20, edge_color=edge_color, width=2, connectionstyle='arc3, rad=0.1')
    labels=nx.draw_networkx_labels(G,pos=pos,  font_size=font_size, font_color='w')

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    graph.make_acyclic()

    G = nx.DiGraph()

    G.add_nodes_from(list(graph.nodes.keys()))
    for n, e in graph.edges.items():
        for u in e:
            G.add_edge(n, u)
    pos = nx.layout.spring_layout(G, pos=pos, fixed=pos.keys())

    node_sizes = [1000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [edge_color for i in range(2, M + 2)]
    edge_alphas = [(1 + i) / (M + 10) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=20, edge_color=edge_color,  width=3, connectionstyle='arc3, rad=0.1')
    labels=nx.draw_networkx_labels(G,pos=pos,  font_size=font_size, font_color='w')

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()