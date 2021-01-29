#!/usr/bin/env python3
import time

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, trainer
from xml.dom import minidom
import math
import numpy as np
import pandas as pd
import random

import sys
import getopt
from numpy.ma import corrcoef
from mxnet import nd, gluon, init, autograd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mxnet.gluon.data.vision.transforms import Normalize

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
        self.distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)*111
    id_begin = []
    id_end = []
    distance = 0

class map_class:
    Node_Class = []
    Link_Class = []
    Dataset = []
    cost = []




def classify(dist):
    if dist < 1450:
        return 3
    elif dist < 1800:
        return 2
    elif dist < 2120:
        return 1
    else:
        return 0


def classifynn(dist):
    tmp = nd.zeros((1, 4))
    if dist < 1400.75:
        tmp[0,3] = 1
    elif dist < 1800 and dist >=1400 :
        tmp[0,2] = 1
    elif dist < 2200 and dist >=1800:
        tmp[0,1] = 1
    else:
        tmp[0,0] = 1
    return tmp


def import_data(map_class):
    xmldoc = minidom.parse("Input_data/dataset.xml")
    Node_List = xmldoc.getElementsByTagName('node')
    global _no_of_input_layers
    _no_of_input_layers = len(Node_List)
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
        map_class.Node_Class.append(Node_Class(s.attributes['id'].value,cor_x,cor_y,iter_i))
    Link_List=xmldoc.getElementsByTagName("link")
    for s in Link_List:
        # print(s.attributes['id'].value)
        r=parse("{}<source>{}</source>{}<target>{}</target>{}", s.toxml())
        source=r[1]
        destination=r[3]
        map_class.Link_Class.append(Link_Class(source, destination, map_class.Node_Class))


def _sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return (sum)

def explore_path(input_matrix , trace_back, source, map_class,tier, cost_trace_back,costs):
    new_tier = tier + 1
    new_trace_back = trace_back.copy()
    new_cost_trace_back = cost_trace_back.copy()

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

        new_cost_trace_backA = cost_trace_back.copy()
        new_cost_trace_backB = cost_trace_back.copy()
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
                explore_path(input_matrix, new_trace_back, link.id_end, map_class, new_tier, new_cost_trace_backA,costs)
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
                explore_path(input_matrix, new_trace_back, link.id_begin, map_class, new_tier, new_cost_trace_backB,costs)
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
            new_cost_trace_backA = cost_trace_back.copy()
            new_cost_trace_backB = cost_trace_back.copy()
            if link.id_begin == source.id:    # jesli link begin jest roowny nodowi z ktorego wychodzimy
                new_cost_trace_backA.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_end, map_class, new_tier, new_cost_trace_backA,costs)
            elif link.id_end == source.id:
                new_cost_trace_backB.append(link.distance)
                explore_path(input_matrix, new_trace_back, link.id_begin, map_class, new_tier, new_cost_trace_backB,costs)


def cut_path(path):
    #path = path.tolist()
    for i in range(len(path)):
        if path[i] == -1:
            path = path[:i]
            return path



def generate_dataset(map_class):
    input_matrix  = []
    cost = []
    cost_traceback = []

    trace_back = []
    start_exploring(input_matrix, trace_back, map_class, 0 ,cost_traceback,cost)
    # maaap = {0: 0, 1: 0, 2:0, 3:0}
    # for c in cost:
    #     maaap[classify(c)] += 1
    # print(maaap)
    input_matrix = np.array(input_matrix)
    cost = np.array(cost)
    return input_matrix, cost

def create_dataset(map_class, size):
    visit_matrix, cost_array = generate_dataset(map_class)
    result_matrix = []
    result_cost = []
    visit_matrix = visit_matrix.tolist()
    cost_array = cost_array.tolist()
    for i in range(size):
        rand = random.randint(0, len(visit_matrix) - 1)
        result_matrix.append(visit_matrix.pop(rand))
        result_cost.append(cost_array.pop(rand))
    return np.array(result_matrix), np.array(result_cost)


def divide_dataset(visit_matrix, cost_array, percentageToTrain):
    divide_index = int(percentageToTrain * len(visit_matrix))
    train_visit = visit_matrix[:divide_index]
    train_costs = cost_array[:divide_index]
    valid_visit = visit_matrix[len(visit_matrix)-divide_index:]
    valid_costs = cost_array[len(visit_matrix)-divide_index:]
    return train_visit, train_costs, valid_visit, valid_costs

def format_input(path):
    inputs = [0] * (_no_of_input_layers + 1)
    #print(_no_of_input_layers)
    #print(path)
    for city_id in path:
        if city_id == -1:
            break
        inputs[int(city_id)] = 1
    #print(inputs)
    return np.array(inputs)

def classifyLogisticRegression(firstResult, secondResult):
    if firstResult:
        return int(secondResult == True) + 2
    else:
        return int(secondResult == True)


def classify_nn(output):
    tmp = output[0,:]
    chosen=max(tmp)
    int_i=4
    for z in range(len(tmp)):
        int_i -=1
        if tmp[z]==chosen:
            return z
    print("Err")
    return -1



def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return _sum(output.argmax(axis=1) == label.mean().asscalar())

def logict(city_map, dataset_size):

    matrix, cost = create_dataset(city_map, dataset_size)
    y = [classify(c) for c in cost]
    y = np.array(y)
    x = matrix

    for i in range(len(x)):
        x[i] = format_input(x[i])

    x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(solver='saga', C=0.33, random_state=None, multi_class='auto', n_jobs=-1, penalty='l1').fit(x_train, y_train)

    y_predicted = model.predict(x_test)
    print("TRAINING SCORE: ", model.score(x_train, y_train))
    print("TEST SCORE: ", model.score(x_test, y_test))

    #print(classification_report(y_test, y_predicted))
    return model.score(x_test, y_test)

def neural_network(city_map, dataset_size):
    visit_matrix, cost_array = create_dataset(city_map, dataset_size)

    train_visit, train_costs, valid_visit, valid_costs = divide_dataset(visit_matrix, cost_array, 0.5)
    print(len(visit_matrix))

    batch_size = len(train_visit)
    print(batch_size)
    net = nn.Sequential()
    # When instantiated, Sequential stores a chain of neural network layers.
    # Once presented with data, Sequential executes each layer in turn, using
    # the output of one layer as the input for the next
    with net.name_scope():
        net.add(nn.Dense(1280, activation="tanh"))  # 2nd hidden layer
        net.add(nn.Dense(640, activation="tanh"))  # 2nd hidden layer
        net.add(nn.Dense(4))  # output layer
    net

    softmax_cross_entropy = gluon.loss.L2Loss()

    # softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    net.initialize(init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    epochs = 400
    procent = 0
    for epoch in range(epochs):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()

        for data, label in zip(train_visit, train_costs):
            # forward + backward
            with autograd.record():
                x = nd.ones((1, 18))
                x[0, :] = data[:]
                output = net(x)
                # tmp = 1/_sum(output[0,:])
                # output = output * tmp
                # print(output)
                # print (classifynn(label))
                loss = softmax_cross_entropy(output, classifynn(label))
            loss.backward()
            # update parameters
            trainer.step(batch_size / 1000)

            # calculate training metrics
            # train_loss += _sum(loss.mean().asscalar())
            train_acc += loss

        for data, label in zip(valid_visit, valid_costs):
            x = nd.ones((1, 18))
            x[0, :] = data[:]  # jebane gowno
            output = net(x)
            valid_acc += softmax_cross_entropy(output, classifynn(label))
        # calculate validation accuracy
        procent += 1
        if procent > epochs / 100:
            procent = 0
            print("Epoch %d  in %.1f sec" % (epoch, time.time() - tic))
            print("Trainint ERR")
            print(train_acc)
            print("Validation Error")
            print(valid_acc)
            print("********************************")
    trainer.save_states("NN_data.nn")
    #            print("Epoch %d: train acc %.3f, test acc %.3f, in %.1f sec" % (
    #                epoch, train_acc / len(train_visit),
    #                valid_acc / len(valid_visit), time.time() - tic))
    good = 0
    bad = 0
    for data, label in zip(valid_visit, valid_costs):
        x = nd.ones((1, 18))
        x[0, :] = data[:]  # jebane gowno
        output = net(x)

        if classify_nn(output) == classify(label):
            good += 1
        else:
            # print("Predicted", classify_nn(output), " In fact it is ", classify(label))
            bad += 1

    print("Good:", good)
    print("bad: ", bad)
    print("succes rate: ", good / (good + bad) * 100, "%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    city_map = map_class()
    import_data(city_map)
    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hn:s:i:t:o:")
        except getopt.GetoptError:
            print ('main.py -n [initial dataset size] -s [step between tests] -i [iterations on each test] -t [number of tests] -o [output file]')
            sys.exit(2)
        for o, a in opts:
            if o == "-n":
                initial_dataset_size = int(a)
            elif o == "-s":
                step_between_tests = int(a)
            elif o == "-i":
                iterations = int(a)
            elif o == "-t":
                n_of_tests = int(a)
            elif o == "-o":
                filename = a
            else:
                print('Usage: main.py -n [initial dataset size] -s [step between tests] -i [iterations on each test] -t [number of tests] -o [output file]')
                sys.exit(0)


        datasize = initial_dataset_size
        results = []
        while n_of_tests > 0:
            iter_counter = 0
            scores = []
            while iter_counter < iterations:
                scores.append(logict(city_map, datasize))
                iter_counter += 1
            results.append((datasize, np.array(scores).mean()))
            datasize += step_between_tests
            n_of_tests -= 1

        df = pd.DataFrame(results,columns=["Dataset size", "Efficiency on test set"])
        df.to_csv(filename, index=None)
    else:
        print("Using default mode, to see help on use {} -h", sys.argv[0])
        dataset_size = 1200


        logict(city_map, dataset_size)

        #neural_network(city_map, dataset_size)









