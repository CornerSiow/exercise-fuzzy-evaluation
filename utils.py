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
import skfuzzy as fuzz

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
    features = []
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

def generateSimilarityFuzzyMembership(node:Node, features, threshold):    
    A = node.w
   
    centerPointResult = []
    plt.figure()
    plt.subplot(2, 1, 2)
    maxDist = 0
    for n_index, n in enumerate(node.neighbourNodes):
        B = n.w       
        if distance.cityblock(B, A) > maxDist:
            maxDist = distance.cityblock(B, A)
        
        dir_AB = (B - A)
        dir_AB /= np.linalg.norm(dir_AB) 

        # Obtains the data between node A and Node B   
        temp = []
        for point in features:
            if distance.cosine(A, point) >= threshold and distance.cosine(B, point) >= threshold:
                continue
            if np.dot(dir_AB, point-A) < 0 or np.dot(-dir_AB, point-B) < 0:
                continue
            temp.append(point)
        temp = np.asarray(temp) 
        
        
        #split to three group
        group = [[],[],[],[]]     
        for d in temp:
            _dist = distance.cosine(A,d)
            if _dist < (threshold / 4):
                group[0].append(d)
            elif _dist < (threshold / 4 * 2):
                group[1].append(d)
            elif _dist < (threshold / 4 * 3):
                group[2].append(d)
            else: 
                group[3].append(d)      
               
        
        plt.title("The data between the essential Pose and its neighbour nodes")
        plt.xlabel("hand position x")
        plt.ylabel("hand position y")
        colorList = ["#6060FF","#AAAA00","#00AAAA", "#00FF00"]
        for i in range(4):
            group[i] = np.asarray(group[i])
            plt.scatter(group[i][:,0], group[i][:,1], c=colorList[i], marker='.')
        
        if n_index == 0:
          plt.scatter(A[0], A[1], marker='^', c="#FF0000", label='Essential Pose')   
          plt.scatter(B[0], B[1], marker='v', c="#FF0000", label='Neighbour Nodes')
        else:
          plt.scatter(A[0], A[1], marker='^', c="#FF0000")   
          plt.scatter(B[0], B[1], marker='v', c="#FF0000")
        
        
        # The closest to Node A
        m1 = group[0].mean(axis=0)        
        # The closest to Node B
        m4 = group[3].mean(axis=0)        
        # In Transit Between Node A and Node B
        m2 = group[1].mean(axis=0)
        m3 = group[2].mean(axis=0)
        
        size = 12
        if n_index == 0:
          plt.scatter([m1[0],m2[0],m3[0],m4[0]] , [m1[1],m2[1],m3[1],m4[1]] , marker='*', c="#000000", label='Average Point of each group') 
        else:
          plt.scatter([m1[0],m2[0],m3[0],m4[0]] , [m1[1],m2[1],m3[1],m4[1]] , marker='*', c="#000000") 
        plt.annotate("1", (m1[0] , m1[1]), fontsize=size)
        plt.annotate("2", (m2[0] , m2[1]), fontsize=size)
        plt.annotate("3", (m3[0] , m3[1]), fontsize=size)
        plt.annotate("4", (m4[0] , m4[1]), fontsize=size)
                
        centerPointResult.append([distance.cityblock(m1, A),distance.cityblock(m2, A),distance.cityblock(m3, A),distance.cityblock(m4, A)])
    plt.legend()
    centerPointResult = np.asarray(centerPointResult)    
  
    dist1 = max(centerPointResult[:,0])
    dist2 = max(centerPointResult[:,1])
    dist3 = max(centerPointResult[:,2])
    dist4 = max(centerPointResult[:,3])
    
    x_similarity = np.arange(0, maxDist, 0.1)
    similarity_hi = fuzz.trapmf(x_similarity, [0, 0, dist1, dist2])
    similarity_md = fuzz.trapmf(x_similarity, [dist1, dist2, dist3, dist4])
    similarity_lo = fuzz.trapmf(x_similarity, [dist3, dist4, maxDist, maxDist])
    
    return x_similarity, similarity_hi, similarity_md, similarity_lo

def getCurrentPoseIndex(point, node_list, fuzzy_members):
  bestIndex = None
  bestScore = 0
  for i in range(len(fuzzy_members)):
    membership = fuzzy_members[i]
    node = node_list[i]
    dist = distance.cityblock(node.w, point)
    dist = min(dist, max(membership['x_axis']))
    score = fuzz.interp_membership(membership['x_axis'], membership['m_hi'], dist)
    if score > bestScore:
      bestScore = score
      bestIndex = i
  return bestIndex



def getCenterScore(m1,m2,m3):
    rulesShape = [[0,0,80],[0,80,100],[80,100,100]]
    ruleScore = [m1,m2,m3]
    
    sumTemp = [0,0]
    for i, (a1, a2, a3) in enumerate(rulesShape):
        M = ruleScore[i]
        LA = (M * M * a2 - M * M * a1)/2
        LC = (3*a1+2*M*a2-2*M*a1)/3
            
        MA = M*a3-M*M*a3-M*a1+a1*M*M
        MC = (a1+2*M*a2-M*a1+a3-M*a3)/2
        
        RA = (M*M*a3 - M*M*a2)/2
        RC = ((3*a3-2*M*a3+2*M*a2))/3
        # print(LA,MA,RA)
            
        sumTemp[0] += LC * LA
        sumTemp[0] += MC * MA
        sumTemp[0] += RC * RA
        sumTemp[1] += LA + MA + RA
    
    return sumTemp[0]/sumTemp[1]

def obtainsMembershipScores(features, node_list,fuzzy_members):
  currentPoseIndex = getCurrentPoseIndex(features[0],node_list,fuzzy_members)
  temporalList = [currentPoseIndex]
  scoreForEachPose = [0] * len(node_list)
  # hi, md, lo
  maxScore = [0,0,0]
  # for speed within each node
  frameCount = 0
  speedList = []
  for frame, point in enumerate(features):
    membership = fuzzy_members[currentPoseIndex]
    node = node_list[currentPoseIndex]

    dist = distance.cityblock(node.w, point)
    dist = min(dist, max(membership['x_axis']))
    hiScore = fuzz.interp_membership(membership['x_axis'], membership['m_hi'], dist)
    mdScore = fuzz.interp_membership(membership['x_axis'], membership['m_md'], dist)
    loScore = fuzz.interp_membership(membership['x_axis'], membership['m_lo'], dist)
    if hiScore > maxScore[0]:
      maxScore[0] = hiScore
    if mdScore > maxScore[1]:
      maxScore[1] = mdScore
    if loScore > maxScore[2]:
      maxScore[2] = loScore
    if mdScore > hiScore:
      frameCount += 1
    if loScore > mdScore:
      if maxScore[0] < scoreForEachPose[currentPoseIndex] or scoreForEachPose[currentPoseIndex] == 0:
        scoreForEachPose[currentPoseIndex] = maxScore[0]
      currentPoseIndex = getCurrentPoseIndex(point,all_nodes,fuzzy_members)
      temporalList.append(currentPoseIndex)
      speedList.append(frameCount)
      maxScore = [0,0,0]
      frameCount = 0
  # for the last node:
  if maxScore[0] < scoreForEachPose[currentPoseIndex] or scoreForEachPose[currentPoseIndex] == 0:
        scoreForEachPose[currentPoseIndex] = maxScore[0]
  speedList = np.asarray(speedList)
  temporalList = np.asarray(temporalList)
  scoreForEachPose = np.asarray(scoreForEachPose)
  return temporalList, speedList, scoreForEachPose
