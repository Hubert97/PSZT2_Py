# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from xml.dom import minidom
import math
import numpy as np
import random

from numpy.ma import corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from parse import parse
import matplotlib.pyplot as plt


_no_of_input_layers = 0


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

class map_class:
    Node_Class = []
    Link_Class = []
    Dataset = []
    cost = []


class LogisticRegressionMod:
    step = 0
    n_of_inputs = 0
    decision_boundary = 0.5
    estimators = []
    def __init__(self, step, n_of_inputs):
        self.n_of_inputs = n_of_inputs
        self.step = step
        self.estimators = [0] * (n_of_inputs + 1)


    def logisticSum(self, path):
        result = 0.0
        for p, w in zip(path, self.estimators):
            result += float(p) * w
        return result


    def predict(self, path):
        return 1.0 / (1.0 + math.exp(-1 * self.logisticSum(path)))


    def updateEstimators(self, path, classification, prediction):
        for i in range(len(self.estimators)):
            self.estimators[i] = self.estimators[i] - self.step * float(path[i]) * (prediction - classification)

    def train(self, path, classification):
        prediction = self.predict(path)
        self.updateEstimators(path, classification, prediction)
        self.step *= 0.99

    def decide(self, path):
        return self.predict(path) >= self.decision_boundary




def classify(dist):
    if dist < 3:
        return 3
    elif dist < 6:
        return 2
    elif dist < 9:
        return 1
    else:
        return 0



def import_data(map_class):
    print('Importing data')
    xmldoc = minidom.parse("Input_data/dataset.xml")
    Node_List = xmldoc.getElementsByTagName('node')
    global _no_of_input_layers
    _no_of_input_layers = len(Node_List)
    print("No of input Layers = ", _no_of_input_layers)
    #print(_no_of_input_layers)
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
    print("No of links beetween nodes =  ", len(Link_List))
    for s in Link_List:
        # print(s.attributes['id'].value)
        r=parse("{}<source>{}</source>{}<target>{}</target>{}", s.toxml())
        source=r[1]
        destination=r[3]
        #print(source)
        #print(destination)
        map_class.Link_Class.append(Link_Class(source, destination, map_class.Node_Class))
    #print(len(map_class.Link_Class))


def _sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return (sum)

def explore_path(input_matrix , trace_back, source, map_class,tier, cost_trace_back,costs):
    new_tier = tier + 1
    new_trace_back = trace_back.copy()
    new_cost_trace_back = cost_trace_back.copy()
    new_cost_trace_backA = cost_trace_back.copy()
    new_cost_trace_backB = cost_trace_back.copy()
    for node in map_class.Node_Class:
        if source == node.id:
            new_trace_back.append(node.no)

    A = np.ones(18)
    B = np.array(-1)
    A = A.dot(B)

    for iter_b in range(0, len(new_trace_back)):
        A[iter_b] = new_trace_back[iter_b]

    input_matrix.append(A)      # wsadz nowa scieżke do macierzy
    costs.append(_sum(new_cost_trace_back))

    iter_a = 0;


    for link in map_class.Link_Class:

        if link.id_begin == source: # jesli link begin jest rowny nodowi z ktorego wychodzimy
            chk_flag = 0
            dst_id = -1
            for node in map_class.Node_Class:
                if link.id_end == node.id:
                    dst_id = node.no
            for t in new_trace_back:
                if t == dst_id or t == -1:  # jesli w traceback nie ma linku o id rownym end
                    chk_flag += 1

            if chk_flag == 0:
                new_cost_trace_backA.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_end, map_class, new_tier, new_cost_trace_back,costs)
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
                new_cost_trace_backB.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_begin, map_class, new_tier, new_cost_trace_back,costs)
                iter_a += 1




def start_exploring(input_matrix , trace_back, map_class,tier, cost_trace_back,costs):

    for source in map_class.Node_Class:
        new_tier = 0
        new_trace_back = trace_back.copy()
        new_cost_trace_back = cost_trace_back.copy()

        for node in map_class.Node_Class:
            if source.id == node.id:
                new_trace_back.append(node.no)

        for link in map_class.Link_Class:
            if link.id_begin == source.id:    # jesli link begin jest roowny nodowi z ktorego wychodzimy
                new_cost_trace_back.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_end, map_class, new_tier, new_cost_trace_back,costs)
            elif link.id_end == source.id:
                new_cost_trace_back.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_begin, map_class, new_tier, new_cost_trace_back,costs)


def cut_path(path):
    #path = path.tolist()
    for i in range(len(path)):
        if path[i] == -1:
            path = path[:i]
            return path



def generate_dataset(map_class):
    print("Generating Dataset")
    MAX_ILOSC_DANYCH=2000
    ilosc_danych=0
    input_matrix  = []
    cost = []
    cost_traceback = []

    trace_back = []
    start_exploring(input_matrix, trace_back, map_class, 0 ,cost_traceback,cost)
    #print(input_matrix[1:100])
    #print(len(input_matrix))
    #print(cost[1:10])
    #print(cost)
    maaap = {0: 0, 1: 0, 2:0, 3:0}
    for c in cost:
        maaap[classify(c)] += 1
    print(maaap)
    #plt.plot(cost)
    #plt.show()
    return input_matrix, cost

def create_dataset(map_class, size):
    visit_matrix, cost_array = generate_dataset(map_class)
    print("Max Val : ",max(cost_array)," , Min Val :", min(cost_array))
    result_matrix = []
    result_cost = []
    for i in range(size):
        rand = random.randint(0, len(visit_matrix) - 1)
        result_matrix.append(visit_matrix.pop(rand))
        result_cost.append(cost_array.pop(rand))
    return result_matrix, result_cost


def divide_dataset(visit_matrix, cost_array, percentageToTrain):
    divide_index = int(percentageToTrain * len(visit_matrix))
    train_visit = visit_matrix[:divide_index]
    train_costs = cost_array[:divide_index]
    valid_visit = visit_matrix[divide_index:]
    valid_costs = cost_array[divide_index:]
    return train_visit, train_costs, valid_visit, valid_costs

def format_input(path):
    inputs = [0] * (_no_of_input_layers + 1)
    #print(_no_of_input_layers)
    #print(path)
    for city_id in path:
        #print(city_id)
        inputs[int(city_id)] = 1
    #print(inputs)
    return inputs

def classifyLogisticRegression(firstResult, secondResult):
    if firstResult:
        return int(secondResult == True) + 2
    else:
        return int(secondResult == True)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Program Init')
    city_map = map_class()

    #Creating Neural Network
    import_data(city_map)
    step = 1.0
    percentageToTrain = 0.8
    dataset_size = 15000
    parityBit = LogisticRegressionMod(step, _no_of_input_layers)
    lowerHalf = LogisticRegressionMod(step, _no_of_input_layers)
    upperHalf = LogisticRegressionMod(step, _no_of_input_layers)
    visit_matrix, cost_array = create_dataset(city_map, dataset_size)
    train_visit, train_costs, valid_visit, valid_costs = divide_dataset(visit_matrix, cost_array, percentageToTrain)
    for visit, cost in zip(train_visit, train_costs):
        visit = cut_path(visit)
        #print("Rozmiar tego guwna to ", len(visit))
        classification = classify(cost)
        formated_visit_array = format_input(visit)
        #print("A rozmiar tego gunwa to ", len(formated_visit_array))
        if classification <= 1:
            parityBit.train(formated_visit_array, 0)
            lowerHalf.train(formated_visit_array, classification)
        else:
            parityBit.train(formated_visit_array, 1)
            upperHalf.train(formated_visit_array, classification % 2)

    good = 0
    bad = 0
    for visit, cost in zip(valid_visit, valid_costs):
        visit = cut_path(visit)

        formated_visit_array = format_input(visit)
        predict1 = parityBit.decide(formated_visit_array)
        if predict1:
            predict2 = upperHalf.decide(formated_visit_array)
        else:
            predict2 = lowerHalf.decide(formated_visit_array)

        if classifyLogisticRegression(predict1, predict2) == classify(cost):
            good += 1
        else:
            bad += 1

    print("GOOD GUESSES:", good)
    print("BAD GUESSES:", bad)
    print("GOOD PERCENTAGE", good / (good+bad) * 100)




    # net = nn.Sequential()
    # # When instantiated, Sequential stores a chain of neural network layers.
    # # Once presented with data, Sequential executes each layer in turn, using
    # # the output of one layer as the input for the next
    # with net.name_scope():
    #     net.add(nn.Dense(17, activation="relu"))  # 1st layer (17 miast)
    #     net.add(nn.Dense(128, activation="relu"))  # 2nd hidden layer
    #     net.add(nn.Dense(64, activation="relu"))  # 2nd hidden layer
    #     net.add(nn.Dense(4))                       #output layer


