# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from xml.dom import minidom
import matplotlib.pyplot as plt

class Node_Class:
    id=[]
    pos_x=0
    pos_y=0

class Link_Class:
    id_begin=[]
    id_end=[]
    weight=0


no_of_input_layers=0

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def import_data():
    print('importing data')
    xmldoc = minidom.parse("Input_data/dataset.xml")
    Node_List=xmldoc.getElementsByTagName('node')
    no_of_input_layers = len(Node_List)
    print("No of input Layers = ")
    print( no_of_input_layers)


    Link_List=xmldoc.getElementsByTagName("link")
    print("No of links beetween nodes =  ")
    print( len(Link_List))
    for s in Link_List:
        print(s.attributes['id'].value)
        print(s.NodeList())













# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Program Init')
    #Creating Neural Network
    import_data()


    net = nn.Sequential()
    # When instantiated, Sequential stores a chain of neural network layers.
    # Once presented with data, Sequential executes each layer in turn, using
    # the output of one layer as the input for the next
    with net.name_scope():
        net.add(nn.Dense(17, activation="relu"))  # 1st layer (17 miast)
        net.add(nn.Dense(128, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(64, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(4))                       #output layer


