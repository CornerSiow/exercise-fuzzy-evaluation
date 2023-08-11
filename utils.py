#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:09:31 2023

@author: Corner Siow
"""

import math
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Node:        
    def __init__(self, w, index):
        self.w = w
        self.neighbourNodes = []
        self.fromNodes = []
        self.toNodes = []
        self.index = index
    def addNeighbour(self, node):
        if node not in self.neighbourNodes:
            self.neighbourNodes.append(node)
    def addFromNode(self, node):
        if node not in self.fromNodes:
            self.fromNodes.append(node)
    def addToNode(self, node):
        if node not in self.toNodes:
            self.toNodes.append(node)            
    def updateWeight(self, d, learning_rate):
        self.w += learning_rate * (d - self.w)
    def updateNeighboardWeight(self, d, learning_rate):
        for node in self.neighbourNodes:
            node.updateWeight(d, learning_rate)

def linkNode(fromA:Node, toB:Node):
    fromA.addNeighbour(toB)
    toB.addNeighbour(fromA)
    fromA.addToNode(toB)
    toB.addFromNode(fromA)

def hasNearestNode(nodeList, point, threshold):    
    bestNode = None
    bestDist = math.inf
    for node in nodeList:
        dist = distance.cosine(node.w, point)
        if dist <= threshold and dist < bestDist:
            bestDist = dist
            bestNode = node
            
    return bestNode

def isWithinDirection(directionList, point):
    for direction in directionList:        
        if np.dot(direction, point) < 0:
            return False
    return True
    
def displayGraph(nodeList, features, title=''):
    plt.figure()
    plt.title(title)
    plt.xlabel('hand position x')
    plt.ylabel('hand position y')
    plt.scatter(features[:,0], features[:,1])
    for node in nodeList:
        for n in node.neighbourNodes:     
            plt.plot([node.w[0],n.w[0]], [node.w[1],n.w[1]])
        plt.scatter(node.w[0], node.w[1])  
        plt.annotate(node.index, (node.w[0], node.w[1]))
    plt.show()

def searchChange(nodeList, features, threshold, display = False):
    requiredChange = {}
    if display:
      plt.figure()
      plt.scatter(features[:,0], features[:,1])
    for nodeIndex, node in enumerate(nodeList):
        directionList = []    
        for n in node.fromNodes:                    
            direction = (node.w - n.w)
            direction /= np.linalg.norm(direction)    
            directionList.append(direction)
        
        avgDirection = np.zeros(len(node.w))
        for d in directionList:
            avgDirection += d
            point = d  + node.w
            if display:  
              plt.plot([node.w[0], point[0]],[node.w[1], point[1]],c="#0000FF")
        avgDirection /= len(directionList)
        
        targetPoint = avgDirection  + node.w      
        if display:  
          plt.scatter(targetPoint[0], targetPoint[1], marker='x', c="#FF0000")              
          plt.scatter(node.w[0], node.w[1], marker='*', c="#FFFF00")       
    
        bestNode = None
        bestDist = math.inf
        for i, point in enumerate(features):
            if isWithinDirection(directionList, point - node.w):         
                if distance.cosine(point, node.w) < threshold:
                    dist = distance.euclidean(point, targetPoint)
                    if dist < bestDist:
                        bestDist = dist
                        bestNode = Node(point,i)
        requiredChange[nodeIndex] = bestNode
        
    for k in requiredChange:
        node = requiredChange[k]
        nodeList[k].w = node.w
        nodeList[k].index = node.index
    return nodeList

def getFeatures(data):
    for v in data:
      temp = []
      t = [i for i in range(33)]
      # try to show the right hand for the first index
      t[0] = 15
      t[15] = 0
      for i in t:
        temp = temp + [v[i].x, v[i].y]
      features.append(temp)
    features = np.asarray(features)
    return features

