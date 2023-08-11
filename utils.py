#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:09:31 2023

@author: Corner Siow
"""

import math
from scipy.spatial import distance

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



def hasNearestNode(nodeList, point, threshold):    
    bestNode = None
    bestDist = math.inf
    for node in nodeList:
        dist = distance.cosine(node.w, point)
        if dist <= threshold and dist < bestDist:
            bestDist = dist
            bestNode = node
            
    return bestNode
