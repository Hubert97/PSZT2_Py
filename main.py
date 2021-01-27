# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from xml.dom import minidom
import math

from numpy.ma import corrcoef

from parse import parse
import matplotlib.pyplot as plt

class Node_Class:
    def __init__(self,id,x,y):
        self.id = id
        self.pos_x = float(x)
        self.pos_y = float(y)
    id = []
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


class LogisticRegression:
    step = 0
    n_of_inputs = 0
    decision_boundary = 0.5
    weights = []
    def __init__(self, step, n_of_inputs):
        self.step = step
        self.n_of_inputs = n_of_inputs
        for n in range(0, n_of_inputs):
            self.weights.append(0.0)


    def logisticFunction(self, path):
        result = 0.0
        for p, w in zip(path, self.weights):
            result += p * w
        return result


    def predict(self, path):
        return 1.0 / (1 + math.exp(-1 * self.logisticFunction(path)))


    def calculateCost(self, prediction, classification):
        return (1.0 - classification) * math.log(1.0 - prediction) - math.log(prediction) * classification


    def updateWeights(self, path, classification, prediction):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.step * path[i] * (prediction - classification)

    def train(self, path, classification):
        prediction = self.predict(path)
        #cost = self.cost(prediction, classification)
        self.updateWeights(path, prediction, classification)
        self.step *= 0.999

    def decide(self, path):
        return self.predict(path) >= self.decision_boundary

_no_of_input_layers=0


def classify(dist):
    dist /= 1000
    if dist < 200.0:
        return 3
    elif dist >= 200 and dist < 400:
        return 2
    elif dist >= 400 and dist < 600:
        return 1
    else:
        return 0



def import_data(map_class):
    print('importing data')
    xmldoc = minidom.parse("Input_data/dataset.xml")
    Node_List = xmldoc.getElementsByTagName('node')
    _no_of_input_layers = len(Node_List)
    print("No of input Layers = ")
    print(_no_of_input_layers)
    for s in Node_List:
        tmpstr=s.toxml()
        # print(tmpstr)
        cor_x = 0.0
        cor_y = 0.0
        r=parse("<node id={}<x>{}</x>{}<y>{}</y>{}/node>", tmpstr)
        cor_x = r[1]
        cor_y = r[3]
        # print(cor_x)
        # print(cor_y)
        map_class.Node_Class.append(Node_Class(s.attributes['id'].value,cor_x,cor_y))
    print(len(map_class.Node_Class))
    Link_List=xmldoc.getElementsByTagName("link")
    print("No of links beetween nodes =  ")
    print( len(Link_List))
    for s in Link_List:
        # print(s.attributes['id'].value)
        r=parse("{}<source>{}</source>{}<target>{}</target>{}", s.toxml())
        source=r[1]
        destination=r[3]
        # print(source)
        # print(destination)
        map_class.Link_Class.append(Link_Class(source, destination, map_class.Node_Class))
    print(len(map_class.Link_Class))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Program Init')
    city_map=map_class()
    #Creating Neural Network
    import_data(city_map)
    step = 3.0
    parityBit = LogisticRegression(step=step, n_of_inputs= _no_of_input_layers)
    lowerHalf = LogisticRegression(step=step, n_of_inputs=_no_of_input_layers)
    upperHalf = LogisticRegression(step=step, n_of_inputs= _no_of_input_layers)

    net = nn.Sequential()
    # When instantiated, Sequential stores a chain of neural network layers.
    # Once presented with data, Sequential executes each layer in turn, using
    # the output of one layer as the input for the next
    with net.name_scope():
        net.add(nn.Dense(17, activation="relu"))  # 1st layer (17 miast)
        net.add(nn.Dense(128, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(64, activation="relu"))  # 2nd hidden layer
        net.add(nn.Dense(4))                       #output layer


