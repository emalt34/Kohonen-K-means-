# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:13:02 2019

@author: Emilie Altman
"""

import numpy as np
import csv
import random

#get data from csv file
def getData():
    
    with open('dataset_noclass.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
            
        rows = []
        allData = []
        
        for row in readCSV:
            for i in range(len(row)):
                #see if data is a number and pass it if not
                try:
                    float(row[i])
                except:
                    continue
                #append float to rows
                rows.append(float(row[i]))
                
            #check if rows is empty - had nums  
            if rows != []: 
                #add row of data to allData
                allData.append(rows)
            #clear rows
            rows = []
         
    return allData


#num is how many weights in weight vector
def makeWeights(numWeights, numNodes):
  
    allWeights = []
    #make a weight vector for each cluster
    for n in range(numNodes):
        weightVector = []
        
        for i in range(numWeights):
            randFloat = random.uniform(0,1)
            weightVector.append(round(randFloat, 2))
            
        allWeights.append(weightVector)
   
    return allWeights; 

#caluclates the sum squared error
def sumSqrError(cluster, center):
    sumSqErr = 0
    for point in cluster:
        sumSqPt = 0
        for j in range(len(point)):
            sumSqPt = sumSqPt + (point[j] - center[j])**2
        sumSqErr = sumSqErr + sumSqPt
    return sumSqErr
 

#finds euclidian distance between each weight and current data pattern
#inputs is 1D array of input pattern
#allWeights is a 2D array of weight vectors
def euclidDist(inputs, allWeights):
    
    weightSums = []
   
    #each weight vector
    for weight in allWeights:
        euclid = 0
        #calculate distance with summation
        for i in range(len(inputs)):
            euclid = euclid + (inputs[i] - weight[i])**2
        #make decimal 2 places
        #euclid = round(euclid, 2)
        #append distance for weight vector to list
        weightSums.append(euclid)
        
    return weightSums


#update weights
#winW is closest ('winning') weight vector
#inputs is input vector
def updateWeight(inputs, winW, learnRate):
    
    for i in range(len(winW)):
        #change weights
        change = learnRate * (inputs[i] - winW[i])
        #print("w change" , change)
        winW[i] = winW[i] + change
    return winW, change


def kohonen(data, numOutNode):
    
    #make weights - want same size as pattern
    weights = makeWeights(len(data[0]), numOutNode)
    learn = 0.5
    
    for epoch in range(10):
        #gradually decrease learning rate
        learn = learn - 0.05
        #print('epoch', epoch+1)
        
        for i in range(len(data)):
            
            inputs = data[i]
            #calculate distances between inputs and weight vectors
            distances = euclidDist(inputs, weights)
            #winner is node closest to data
            minDist =  min(distances)
            #index of "winning" node
            winNode = distances.index(minDist)
            #update weights of winner
            newWeight, change = updateWeight(inputs, weights[winNode], learn)
            weights[winNode] = newWeight
            
            #print('up weights', weights)
   
    return weights

        
def clustAvrg(cluster, size): 
    cluster = np.array(cluster)
    #summing along the columns
    sumClust = np.sum(cluster, axis=0)
    #dividing by size of cluster
    avrg = sumClust/size
    #convert back to list for return
    return list(avrg)

    
#use for k means and to divide clusters after kohonen
#will not update weights for kohonen and only does 1 itteration    
def kMeans(data, weights, epochNum, kMean=True):
    
    #weights = makeWeights(len(data[0]), numOutNode)
    
    for epoch in range(epochNum): 
        clust1 = []
        clust2 = []
        for i in range(len(data)):
            inputs = data[i]
            #calculating closest which node the data is closest to
            distances = euclidDist(inputs, weights)
            minDist =  min(distances)
            winNode = distances.index(minDist)
            
            #if winNode is 0 then data point is closer to cluster 1 
            if winNode == 0:
                clust1.append(inputs)
            else: #if it's 1 - closer to cluster 2
                clust2.append(inputs)
            
        avrgC1 = clustAvrg(clust1, len(clust1))
        avrgC2 = clustAvrg(clust2, len(clust2))
       
        if kMean == True:
            weights = [avrgC1, avrgC2]
     
    #calculates sum squared error          
    SSE1 = sumSqrError(clust1, weights[0])
    SSE2 = sumSqrError(clust2, weights[1])
    totalErr = SSE1 + SSE2
    
    #divides by number of data points in clusters
    SSEdiv1 = SSE1/len(clust1)
    SSEdiv2 = SSE2/len(clust2)
    divTotal = totalErr/1000
    
    
    print("Sum squared error cluster1: ", SSEdiv1)
    print("Sum squared error cluster2: ", SSEdiv2)
    print("Total sum squared error: ", divTotal)
    
    return weights 

def main():
    data = getData()
    print("KOHONEN")
    kohoW = kohonen(data, 2)  
    kMeans(data, kohoW, 1, kMean=False)

    print("\nKMEANS")
    weights = makeWeights(len(data[0]), 2)
    kMeans(data, weights, 10)
    
main()    
    

