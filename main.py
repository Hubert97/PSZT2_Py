# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from xml.dom import minidom
import math
import numpy as np

from numpy.ma import corrcoef

from parse import parse
import matplotlib.pyplot as plt

class Node_Class:
    def __init__(self,id,x,y,n):
        self.id = id
        self.pos_x = float(x)
        self.pos_y = float(y)
        self.no = n
    id = []
    no = 0;
    pos_x = 0
    pos_y = 0

class Link_Class:
    def __init__(self,id_begin,id_end,Node_Class):
        self.id_begin=id_begin
        self.id_end=id_end
        delta_y = 0
        delta_x = 0
        for i in Node_Class:
            if i.id == self.id_begin:
                delta_x = i.pos_x
                delta_y = i.pos_y

        for i in Node_Class:
            if i.id == self.id_end:
                delta_x = abs(delta_x - i.pos_x)
                delta_y = abs(delta_y - i.pos_y)
        self.distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    id_begin = []
    id_end = []
    distance = 0
    weight = 0

class map_class:
    Node_Class = []
    Link_Class = []
    Dataset = []
    cost = []


_no_of_input_layers=0



def import_data(map_class):
    print('Importing data')
    xmldoc = minidom.parse("Input_data/dataset.xml")
    Node_List = xmldoc.getElementsByTagName('node')
    _no_of_input_layers = len(Node_List)
    print("No of input Layers = ")
    print(_no_of_input_layers)
    print("check = ")
    iter_i = 0;
    for s in Node_List:
        tmpstr=s.toxml()
        iter_i += 1
        # print(tmpstr)
        cor_x = 0.0
        cor_y = 0.0
        r=parse("<node id={}<x>{}</x>{}<y>{}</y>{}/node>", tmpstr)
        cor_x = r[1]
        cor_y = r[3]
        # print(cor_x)
        # print(cor_y)
        map_class.Node_Class.append(Node_Class(s.attributes['id'].value,cor_x,cor_y,iter_i))
    print(len(map_class.Node_Class))
    Link_List=xmldoc.getElementsByTagName("link")
    print("No of links beetween nodes =  ")
    print( len(Link_List))
    for s in Link_List:
        # print(s.attributes['id'].value)
        r=parse("{}<source>{}</source>{}<target>{}</target>{}", s.toxml())
        source=r[1]
        destination=r[3]
        #print(source)
        #print(destination)
        map_class.Link_Class.append(Link_Class(source, destination, map_class.Node_Class))
    print(len(map_class.Link_Class))





def explore_path(input_matrix , trace_back, source, map_class,tier, cost_trace_back):
    new_tier = tier + 1
    new_trace_back = trace_back
    new_cost_trace_back = cost_trace_back
    for node in map_class.Node_Class:
        if source == node.id:
            new_trace_back.append(node.no)

    A = np.ones(18)
    B = np.array(-1)
    A = A.dot(B)
    for iter_b in range(0, len(new_trace_back)):
        A[iter_b] = new_trace_back[iter_b]
    input_matrix.append(A)      # wsadz nowa scierzke do macierzy

    iter_a = 0;
    print(new_tier)

    for link in map_class.Link_Class:

        if link.id_begin == source: # jesli link begin jest roowny nodowi z ktorego wychodzimy
            chk_flag = 0
            dst_id = -1
            for node in map_class.Node_Class:
                if link.id_end == node.id:
                    dst_id = node.no
            for t in new_trace_back:
                if t == dst_id or t == -1:  # jesli w traceback nie ma linku o id rownym end
                    chk_flag += 1

            if chk_flag == 0:
                explore_path(input_matrix, new_trace_back, link.id_end, map_class, new_tier, new_cost_trace_back)
                iter_a += 1
        elif link.id_end == source:
            chk_flag = 0
            dst_id = -1
            for node in map_class.Node_Class:
                if link.id_begin == node.id:
                    dst_id = node.no
            for t in new_trace_back:
                if t == dst_id or t == -1:
                    chk_flag += 1

            if chk_flag == 0:
                explore_path(input_matrix, new_trace_back, link.id_begin, map_class, new_tier, new_cost_trace_back)
                iter_a += 1




def start_exploring(input_matrix , trace_back, source, map_class,tier, cost_trace_back):
    for node in map_class.Node_Class:
        iter_a=0





def generate_dataset(map_class):
    print("Generating Dataset")
    MAX_ILOSC_DANYCH=2000
    ilosc_danych=0
    input_matrix  = []
    cost = []
    for iter_a in range (0,17):
        trace_back = []
        explore_path(input_matrix, trace_back, map_class.Node_Class[iter_a].id, map_class,0,cost)
    print(input_matrix)
    print(len(input_matrix))
    print(map_class.cost)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Program Init')
    city_map = map_class();
    #Creating Neural Network
    import_data(city_map)
    generate_dataset(city_map)

    net = nn.Sequential()
    # When instantiated, Sequential stores a chain of neural network layers.
    # Once presented with data, Sequential executes each layer in turn, using
    # the output of one layer as the input for the next
    with net.name_scope():
        net.add(nn.Dense(17, activation="relu"))  # 1st layer (17 miast)
        net.add(nn.Dense(128, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(64, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(4))                       #output layer


