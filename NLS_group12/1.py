from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors

M=500
N=375
d=2

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)
    lines = plt.plot(y,x)
    # plt.axis([-20, 20, -20, 20])
    plt.setp(lines, color='r', linewidth=2.0)
    # plt.show()

def readDataSetWhole(fileName):
    f = open(fileName,"r")
    fl =f.readlines()[0:M]
    dSet = [[0 for x in range(d)] for y in range(M)] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = float(lines[j])
        i=i+1
    f.close()
    return dSet

def readDataSetTraining(fileName):
    f = open(fileName,"r")
    fl =f.readlines()[0:N]
    dSet = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = (lines[j])
        i=i+1
    f.close()
    return dSet


def readDataSetTesting(fileName):
    f = open(fileName,"r")
    fl =f.readlines()[N:M]
    dSet = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = (lines[j])
        i=i+1
    f.close()
    return dSet

def computeCovariance(dSet,mean):
    cov_mat = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;
    for i in range(d):
        for j in range(d):
            cov_mat[i][j]=0
            for k in range(len(dSet)):
                cov_mat[i][j] += (float(dSet[k][i])-float(mean[i]))*(float(dSet[k][j])-float(mean[j]))
            cov_mat[i][j] /= N-1
    return cov_mat

def calcW(mean,sigmaInverse):
    w = [mean[0]*sigmaInverse,mean[1]*sigmaInverse]
    return w

def calcW0(mean,sigmaInverse,prior):
    w0 = pow(mean[0],2) + pow(mean[1],2)
    w0 *= (sigmaInverse)
    w0 /= -2
    w0 += log(prior)
    return w0

def calcG(w,w0,x):
    g=(w[0]*float(x[0]))+(w[1]*float(x[1]))+w0
    return g

def getRange():
    X1=readDataSetWhole("Class1.txt")
    X2=readDataSetWhole("Class2.txt")
    X3=readDataSetWhole("Class3.txt")

    xmin=ymin=sys.maxsize
    xmax=ymax=-sys.maxsize

    for i in range(M):
        if X1[i][0] > xmax :
            xmax=X1[i][0]
        if X2[i][0] > xmax :    
            xmax=X2[i][0]
        if X3[i][0] > xmax :    
            xmax=X3[i][0]
        if X1[i][0] < xmin :    
            xmin=X1[i][0]
        if X2[i][0] < xmin :    
            xmin=X2[i][0]
        if X3[i][0] < xmin :    
            xmin=X3[i][0]

        if X1[i][1] > ymax :
            ymax=X1[i][1]
        if X2[i][1] > ymax :
            ymax=X2[i][1]
        if X3[i][1] > ymax :
            ymax=X3[i][1]
        if X1[i][1] < ymin :    
            ymin=X1[i][1]
        if X2[i][1] < ymin :    
            ymin=X2[i][1]
        if X3[i][1] < ymin :    
            ymin=X3[i][1]
    dSet = [[0 for x in range(2)] for y in range(2)]
    dSet[0][0]=xmin
    dSet[0][1]=xmax
    dSet[1][0]=ymin
    dSet[1][1]=ymax
    return dSet


def main():
    sigma = 0.0
    #########
    #Class 1#
    #########
    x1=0
    x2=0
    print ("Reading from file")
    X_class1=readDataSetTraining("Class1.txt")
    for i in range(N):
        x1=x1+float(X_class1[i][0])
        x2=x2+float(X_class1[i][1])
    #Calculating Mean
    mean_x_class1 = [x1/N, x2/N];
    print(mean_x_class1[0]," --- ", mean_x_class1[1])
    #Calculating Covariance
    cov_mat_class1 = computeCovariance(X_class1,mean_x_class1)
    for i in range(d):
        sigma += cov_mat_class1[i][i]
    


    #########
    #Class 2#
    #########
    x1=0
    x2=0
    #Reading from file
    X_class2=readDataSetTraining("Class2.txt")
    for i in range(N):
        x1=x1+float(X_class2[i][0])
        x2=x2+float(X_class2[i][1])
    #Calculating Mean
    mean_x_class2 = [x1/N, x2/N];
    print(mean_x_class2[0]," --- ", mean_x_class2[1])
    #Calculating Covariance
    cov_mat_class2 = computeCovariance(X_class2,mean_x_class2)
    for i in range(d):
        sigma += cov_mat_class2[i][i]


    #########
    #Class 3#
    #########
    x1=0
    x2=0
    #Reading from file
    X_class3=readDataSetTraining("Class3.txt")
    for i in range(N):
        x1=x1+float(X_class3[i][0])
        x2=x2+float(X_class3[i][1])
    #Calculating Mean
    mean_x_class3 = [x1/N, x2/N];
    print(mean_x_class3[0]," --- ", mean_x_class3[1])
    #Calculating Covariance
    cov_mat_class3 = computeCovariance(X_class3,mean_x_class3)
    for i in range(d):
        sigma += cov_mat_class3[i][i]

    print("sigma = ", sigma, "\n")
    sigma /= 6; 
    print("sigma = ", sigma, "\n")

    

    #CASE - A (All covaraince matrix are equal to sigma^2 I)

    cov_matrix_a = [[float(sigma),0],[0,float(sigma)]]

    prior_class1 = (0.75*500)/(0.75*500 + 0.75*500 + 0.75*500)
    prior_class2 = prior_class3 = prior_class1

    sigma_inverse = 1 / (cov_matrix_a[0][0]);

    
    #Calc w and w0
    #For Class 1
    w1 = calcW(mean_x_class1,sigma_inverse)
    w01 = calcW0(mean_x_class1,sigma_inverse,prior_class1)
    #For Class 2
    w2 = calcW(mean_x_class2,sigma_inverse)
    w02 = calcW0(mean_x_class2,sigma_inverse,prior_class2)
    #For Class 3
    w3 = calcW(mean_x_class3,sigma_inverse)
    w03 = calcW0(mean_x_class3,sigma_inverse,prior_class3) 
    

    #Allocating Class to each point
    #For Class 1
    c1_1=c1_2=c1_3=0
    X1=readDataSetTesting("Class1.txt")
    for i in range (M-N):
        g1=calcG(w1,w01,X1[i])
        g2=calcG(w2,w02,X1[i])
        g3=calcG(w3,w03,X1[i])
        if g1==max(g1,g2,g3):
            c1_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c1_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c1_3+=1
            # print("3")
    print("c1_1 = ", c1_1, "c1_2 = ", c1_2, "c1_3 = ", c1_3)
    print ("End of Class 1")
    #For Class 2
    c2_1=c2_2=c2_3=0
    X2=readDataSetTesting("Class2.txt")
    for i in range (M-N):
        g1=calcG(w1,w01,X2[i])
        g2=calcG(w2,w02,X2[i])
        g3=calcG(w3,w03,X2[i])
        if g1==max(g1,g2,g3):
            c2_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c2_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c2_3+=1
            # print("3")
    print("c2_1 = ", c2_1, "c2_2 = ", c2_2, "c2_3 = ", c2_3)
    print ("End of Class 2")
    #For Class 3
    c3_1=c3_2=c3_3=0
    X3=readDataSetTesting("Class3.txt")
    for i in range (M-N):
        g1=calcG(w1,w01,X3[i])
        g2=calcG(w2,w02,X3[i])
        g3=calcG(w3,w03,X3[i])
        if g1==max(g1,g2,g3):
            c3_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c3_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c3_3+=1
            # print("3")
    print("c3_1 = ", c3_1, "c3_2 = ", c3_2, "c3_3 = ", c3_3)
    print ("End of Class 3")
    
    X=getRange()
    xmin=X[0][0]
    xmax=X[0][1]
    ymin=X[1][0]
    ymax=X[1][1]

    print ("xmin = ",xmin)
    print ("ymin = ",ymin)
    print ("xmax = ",xmax)
    print ("ymax = ",ymax)
    A = [[0 for x in range(2)] for y in range(2)]

    i=xmin
    while i<xmax :
        j=ymin
        while j<ymax:
            A[0]=i
            A[1]=j
            g1=calcG(w1,w01,A)
            g2=calcG(w2,w02,A)
            # g3=calcG(w3,w03,A)
            if g1==max(g1,g2):
                plt.plot(i,j,color='#f6668f',marker='s')
            elif g2==max(g1,g2):
                plt.plot(i,j,color='#33d7ff',marker='s')
            # elif g3==max(g1,g2,g3):
            #     plt.plot(i,j,color='#75f740',marker='s')
            j+=0.07
        i+=0.07

    X1=readDataSetTraining("Class1.txt")
    for i in range(N):
        plt.plot(X1[i][0],X1[i][1],'ro')

    X2=readDataSetTraining("Class2.txt")
    for i in range(N):
        plt.plot(X2[i][0],X2[i][1],'bo')

    # X3=readDataSetTraining("Class3.txt")
    # for i in range(N):
    #     plt.plot(X3[i][0],X3[i][1],'go')
        
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("figure112.png")
    plt.clf()

    i=xmin
    while i<xmax :
        j=ymin
        while j<ymax:
            A[0]=i
            A[1]=j
            g1=calcG(w1,w01,A)
            # g2=calcG(w2,w02,A)
            g3=calcG(w3,w03,A)
            if g1==max(g1,g3):
                plt.plot(i,j,color='#f6668f',marker='s')
            # # elif g2==max(g1,g2):
            #     plt.plot(i,j,color='#33d7ff',marker='s')
            elif g3==max(g1,g3):
                plt.plot(i,j,color='#75f740',marker='s')
            j+=0.07
        i+=0.07

    X1=readDataSetTraining("Class1.txt")
    for i in range(N):
        plt.plot(X1[i][0],X1[i][1],'ro')

    # X2=readDataSetTraining("Class2.txt")
    # for i in range(N):
    #     plt.plot(X2[i][0],X2[i][1],'bo')

    X3=readDataSetTraining("Class3.txt")
    for i in range(N):
        plt.plot(X3[i][0],X3[i][1],'go')
        
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("figure113.png")
    plt.clf()
    # plt.show()  

    i=xmin
    while i<xmax :
        j=ymin
        while j<ymax:
            A[0]=i
            A[1]=j
            # g1=calcG(w1,w01,A)
            g2=calcG(w2,w02,A)
            g3=calcG(w3,w03,A)
            # if g1==max(g1,g2):
            #     plt.plot(i,j,color='#f6668f',marker='s')
            if g2==max(g3,g2):
                plt.plot(i,j,color='#33d7ff',marker='s')
            elif g3==max(g2,g3):
                plt.plot(i,j,color='#75f740',marker='s')
            j+=0.07
        i+=0.07

    # X1=readDataSetTraining("Class1.txt")
    # for i in range(N):
    #     plt.plot(X1[i][0],X1[i][1],'ro')

    X2=readDataSetTraining("Class2.txt")
    for i in range(N):
        plt.plot(X2[i][0],X2[i][1],'bo')

    X3=readDataSetTraining("Class3.txt")
    for i in range(N):
        plt.plot(X3[i][0],X3[i][1],'go')
        
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("figure123.png") 
    plt.clf()

if __name__== "__main__":
  main()

  # 1611275.5956010565