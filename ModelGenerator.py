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



class Node(nn.Module):
    def __init__(self, i, name=None):
        super(Node, self).__init__()
        self.i = i
        self.name = name
        self.output = None
        C_in, C_out = 32,32
        self.C_in = C_in
        self.C_out = C_out
        self.convolution = DepthwiseSeparableConv2d(C_in, C_out)
        self.batch_norma = nn.BatchNorm2d(C_out)
        self.activation = nn.ReLU()

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, C_in, C_out, stride = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, groups=C_in, stride=stride)
        self.pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class OpNode(nn.Module):
    def __init__(self, C_in, C_out,name=None,  stride = 1):
        super(OpNode, self).__init__()
        self.C_in, self.C_out = C_in, C_out
        self.convolution = DepthwiseSeparableConv2d(C_in, C_out, stride = stride)
        self.batch_norma = nn.BatchNorm2d(C_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norma(x)
        x = self.activation(x)
        self.output = x

        return x


class Stage(nn.Module):
    def __init__(self, graph, C_in, C_out):
        super(Stage, self).__init__()

        self.graph = graph
        
        convolutions = []
        batch_normas = []
        activations = []

        self.weights_inputs = nn.ModuleDict({str(node) : nn.ParameterDict({str(node_inp) : nn.Parameter(torch.Tensor([1]), requires_grad=True) for node_inp in graph.get_inputs_node(node)}) for node, _ in self.graph.nodes.items() if node != 0})
        
        self.order_to_compute = list(self.graph.edges.keys())[1:]

        self.operation_input = OpNode(C_in=C_in, C_out=C_out, stride=2)
        self.operations_interm = nn.ModuleDict({str(node) : OpNode(C_in=C_out, C_out=C_out, stride=1) for node in self.order_to_compute})
        # print(self.operations_interm)
        # print(self.params(self.operation_input))
        # print(sum([self.params(self.weights_inputs[str(node)]) for node in self.order_to_compute]))
        # print(sum([self.params(self.operations_interm[str(node)]) for node in self.order_to_compute]))
            
    def params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x):
        temp = self.operation_input(x)
        
        for node, nodes_inp in self.weights_inputs.items():
            out = torch.zeros_like(self.operation_input.output)
            for node_inp, w in nodes_inp.items():
                if node_inp == "0":
                    out += w*self.operation_input.output
                else:
                    out += w*self.operations_interm[str(node_inp)].output
            temp = self.operations_interm[str(node)](out)

        
        return self.operations_interm[str(len(self.graph)-1)].output


class ModelNAS(nn.Module):
    def __init__(self, N, K, P, number_layer, number_class):
        super(ModelNAS, self).__init__()

        self.number_class = number_class
        stages = []
        channels = [3,4,8,16,32,64,64]
        self.graphes = []
        for i in range(number_layer):
            graph = WattsStrogatz(N, K, P)
            self.graphes.append(graph)
            # plot_graph(graph)
            # print("\n\nEdges : ", graph.edges)
            # print("Inputs nodes : ", {node : graph.get_inputs_node(node) for node, _ in graph.nodes.items()})
            stages.append(Stage(graph, C_in = channels[i], C_out = channels[i+1]))
            # print("Nombre de paramètres : ", sum(p.numel() for p in stages[-1].parameters()))
            # print()

        self.stages = nn.ModuleList(stages)

        self.linear1 = nn.Linear(1024,100)
        self.linear2 = nn.Linear(100, number_class)

    def forward(self, x):

        for stage in self.stages:
            x = stage(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)

        if self.number_class > 2:
            x = nn.Softmax(dim=1)(x)
        else:
            x = nn.Sigmoid()(x)

        return x
