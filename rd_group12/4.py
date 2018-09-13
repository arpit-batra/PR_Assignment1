from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt

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

def readDataSetTraining(fileName):
    f = open(fileName,"r")
    fl =f.readlines()
    N = int(ceil(0.75*len(fl)))
    # print(N)
    fl = fl[0:int(N)]
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
    fl =f.readlines()
    N = int(ceil(0.75*len(fl)))
    M = int(len(fl))
    # print(N, M)
    fl = fl[int(N):int(M)]
    # print(len(fl))
    dSet = [[0 for x in range(d)] for y in range(len(fl))] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = (lines[j])
        i=i+1
    f.close()
    return dSet

def computeMean(X_class,N):
    x1=0
    x2=0
    for i in range(N):
        x1=x1+float(X_class[i][0])
        x2=x2+float(X_class[i][1])
    #Calculating Mean
    mean_x_class = [x1/N, x2/N];
    return mean_x_class

def computeCovariance(dSet,mean):
    cov_mat = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;
    for i in range(d):
        for j in range(d):
            cov_mat[i][j]=0
            for k in range(len(dSet)):
                cov_mat[i][j] += (float(dSet[k][i])-float(mean[i]))*(float(dSet[k][j])-float(mean[j]))
            cov_mat[i][j] /= N-1
    return cov_mat

def calcW2(cov_matrix_inverse):
    w2 = [[0 for x in range(d)] for y in range(d)]
    for i in range(d):
        for j in range(d):
            w2[i][j] /= (-2)
            
    return w2

def calcW1(mean,cov_matrix_inverse):
    w = [((mean[0]*cov_matrix_inverse[0][0])+(mean[1]*cov_matrix_inverse[0][1])), (mean[0]*cov_matrix_inverse[1][0])+(mean[1]*cov_matrix_inverse[1][1])]
    # print(w12)
    return w;

def calcW0(mean,cov_matrix_inverse,prior,det):
    t = [((mean[0]*cov_matrix_inverse[0][0])+(mean[1]*cov_matrix_inverse[1][0])), ((mean[0]*cov_matrix_inverse[0][1])+(mean[1]*cov_matrix_inverse[1][1]))]
    w0 = ((t[0]*mean[0])+(t[1]*mean[1]))/(-1*2)
    w0 += log(prior)
    w0 -= (det/2)
    return w0

def calcG(w2,w1,w0,x):
    x1 = float(x[0])
    x2 = float(x[1])
    # print(x1," ", x2);
    temp = [( (x1*w2[0][0]) + (x2*w2[1][0]) ), ( (x1*w2[0][1]) + (x2*w2[1][1]) )]
    g = ( (temp[0]*x1) + (temp[1]*x2) )
    
    g += ( (w1[0]*x1) + (w1[1]*x2) )

    g += w0
    # g=(w[0]*float(x[0]))+(w[1]*float(x[1]))+w0
    return g

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)
    lines = plt.plot(y,x)
    # plt.axis([-20, 20, -20, 20])
    plt.setp(lines, color='r', linewidth=2.0)
    # plt.show()

def computeCaseCCovariance(cov_mat_class):
    avg=0.0;
    for i in range(d):
        for j in range(d):
            avg += cov_mat_class[i][j]
    
    cov_mat_class[0][0] = cov_mat_class[1][1] = avg/4
    cov_mat_class[1][0] = cov_mat_class[0][1] = 0
    return cov_mat_class

def calcDet(matrix):
    det = (matrix[0][0]*matrix[1][1]) - (matrix[0][1]*matrix[1][0])        
    return det

def inverseMatrix(cov_matrix_b):
    det = (cov_matrix_b[0][0]*cov_matrix_b[1][1]) - (cov_matrix_b[0][1]*cov_matrix_b[1][0])

    cov_matrix_b[0][1] = -1*cov_matrix_b[0][1]
    cov_matrix_b[1][0] = -1*cov_matrix_b[1][0]

    temp = cov_matrix_b[0][0]
    cov_matrix_b[0][0] = cov_matrix_b[1][1]
    cov_matrix_b[1][1] = temp;

    for i in range(d):
        for j in range(d):
            cov_matrix_b[i][j] /= det;
    return cov_matrix_b



def main():
    #########
    #Class 1#
    #########
    print ("Reading from file")
    X_class1=readDataSetTraining("class1.txt")
    #Calculating Mean
    mean_x_class1 = computeMean(X_class1,len(X_class1))
    print(mean_x_class1[0]," --- ", mean_x_class1[1])
    #Calculating Covariance
    cov_mat_class1 = computeCovariance(X_class1,mean_x_class1)
    det_class1 = calcDet(cov_mat_class1)
    cov_mat_class1 = inverseMatrix(cov_mat_class1)
    print(cov_mat_class1[0][0], "  ", cov_mat_class1[0][1], "\n", cov_mat_class1[1][0], "  ", cov_mat_class1[1][1])

    print("\n--------------------------\n");

    #########
    #Class 2#
    #########
    #Reading from file
    X_class2=readDataSetTraining("class2.txt")
    #Calculating Mean
    mean_x_class2 = computeMean(X_class2,len(X_class2))
    print(mean_x_class2[0]," --- ", mean_x_class2[1])
    #Calculating Covariance
    cov_mat_class2 = computeCovariance(X_class2,mean_x_class2)
    det_class2 = calcDet(cov_mat_class2)
    cov_mat_class2 = inverseMatrix(cov_mat_class2)
    print(cov_mat_class2[0][0], "  ", cov_mat_class2[0][1], "\n", cov_mat_class2[1][0], "  ", cov_mat_class2[1][1])
    
    print("\n--------------------------\n");

    #########
    #Class 3#
    #########
    x1=0
    x2=0
    #Reading from file
    X_class3=readDataSetTraining("class3.txt")
    #Calculating Mean
    mean_x_class3 = computeMean(X_class3,len(X_class3))
    print(mean_x_class3[0]," --- ", mean_x_class3[1])
    #Calculating Covariance
    cov_mat_class3 = computeCovariance(X_class3,mean_x_class3)
    det_class3 = calcDet(cov_mat_class3)
    cov_mat_class3 = inverseMatrix(cov_mat_class3)
    print(cov_mat_class3[0][0], "  ", cov_mat_class3[0][1], "\n", cov_mat_class3[1][0], "  ", cov_mat_class3[1][1])

    print("\n--------------------------\n");
    

    #CASE - C (All covaraince matrix are not equal but diagonal)

    prior_class1 = (0.75*500)/(0.75*500 + 0.75*500 + 0.75*500)
    prior_class2 = prior_class3 = prior_class1

    print(prior_class1, " ", prior_class2, " ", prior_class3)

    #Calc w2, w1 and w0
    #For Class 1
    w2_1 = calcW2(cov_mat_class1)
    w1_1 = calcW1(mean_x_class1,cov_mat_class1)
    w01 = calcW0(mean_x_class1,cov_mat_class1,prior_class1,det_class1)
    #For Class 2
    w2_2 = calcW2(cov_mat_class2)
    w1_2 = calcW1(mean_x_class2,cov_mat_class2)
    w02 = calcW0(mean_x_class2,cov_mat_class2,prior_class2,det_class2)
    #For Class 3
    w2_3 = calcW2(cov_mat_class3)
    w1_3 = calcW1(mean_x_class3,cov_mat_class3)
    w03 = calcW0(mean_x_class3,cov_mat_class3,prior_class3,det_class3)

    #Allocating Class to each point
    #For Class 1
    c1_1=c1_2=c1_3=0
    X1=readDataSetTesting("class1.txt")
    for i in range (len(X1)):
        g1=calcG(w2_1,w1_1,w01,X1[i])
        g2=calcG(w2_2,w1_2,w02,X1[i])
        g3=calcG(w2_3,w1_3,w03,X1[i])
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
    X2=readDataSetTesting("class2.txt")
    for i in range (len(X2)):
        g1=calcG(w2_1,w1_1,w01,X2[i])
        g2=calcG(w2_2,w1_2,w02,X2[i])
        g3=calcG(w2_3,w1_3,w03,X2[i])
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
    X3=readDataSetTesting("class3.txt")
    for i in range (len(X3)):
        g1=calcG(w2_1,w1_1,w01,X3[i])
        g2=calcG(w2_2,w1_2,w02,X3[i])
        g3=calcG(w2_3,w1_3,w03,X3[i])
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
    

    # x = np.linspace(-100, 100, 100)
    # y = np.linspace(-100, 100, 100)
    # X, Y = np.meshgrid(x,y)
    # F = (w2_12[0][0])*(X**2) + (w2_12[1][1])*(Y**2) + X*Y*(w2_12[0][1]+w2_12[1][0]) + w1_12[0]*X + w1_12[1]*Y + w0_12 
    # plt.contour(X,Y,F,[0],cmap=plt.get_cmap('autumn'))
    # F = (w2_13[0][0])*(X**2) + (w2_13[1][1])*(Y**2) + X*Y*(w2_13[0][1]+w2_13[1][0]) + w1_13[0]*X + w1_13[1]*Y + w0_13
    # plt.contour(X,Y,F,[0],cmap=plt.get_cmap('winter'))
    # F = (w2_23[0][0])*(X**2) + (w2_23[1][1])*(Y**2) + X*Y*(w2_23[0][1]+w2_23[1][0]) + w1_23[0]*X + w1_23[1]*Y + w0_23
    # plt.contour(X,Y,F,[0],cmap=plt.get_cmap('spring'))
    # # plt.show()    
    # x=[]
    # y=[]
    # f = open("Class1.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # i=0
    # for lines in fl:
    #     lines=lines.split();
    #     x.append(float(lines[0]))
    #     y.append(float(lines[1]))
    #     i+=1

    # # print(x)
    # plt.plot(x,y,'bs')

    # f = open("Class2.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # x=[]
    # y=[]
    # i=0
    # for lines in fl:
    #     lines=lines.split();
    #     x.append(float(lines[0]))
    #     y.append(float(lines[1]))
    #     i+=1

    # plt.plot(x,y,'ro')
    # # print(x)

    # f = open("Class3.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # x=[]
    # y=[]
    # i=0
    # for lines in fl:
    #     lines=lines.split();
    #     x.append(float(lines[0]))
    #     y.append(float(lines[1]))
    #     i+=1

    # plt.plot(x,y,'gp')

    # plt.show()




if __name__== "__main__":
  main()

  # 1611275.5956010565