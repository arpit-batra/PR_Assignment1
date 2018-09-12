from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)
    lines = plt.plot(y,x)
    # plt.axis([-20, 20, -20, 20])
    plt.setp(lines, color='r', linewidth=2.0)
    # plt.show()

def main():
    #Open the file back and read the contents
    N=375
    d=2

    
    #class 1
    f = open("Class1.txt","r")
    fl =f.readlines()[0:N]
    i=0
    x1=0
    x2=0
    X_class1 = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            X_class1[i][j] = (lines[j])
            # X_class1[i][j] = (lines[1])
            # print(X[i][j],end = " ");
            # print(X[i][j],end = " ");
            if j==0:
                x1 = x1+float(lines[0])
            elif j==1:
                x2 = x2+float(lines[1])
        # print()
        i=i+1
    f.close()

    print(x1," --- ", x2)
    mean_x_class1 = [x1/N, x2/N];
    print(mean_x_class1[0]," --- ", mean_x_class1[1])

    cov_mat_class1 = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;

    for i in range(d):
        for j in range(d):
            cov_mat_class1[i][j]=0
            for k in range(len(X_class1)):
                cov_mat_class1[i][j] += (float(X_class1[k][i])-float(mean_x_class1[i]))*(float(X_class1[k][j])-float(mean_x_class1[j]))
            
            cov_mat_class1[i][j] /= N-1
            
            print(cov_mat_class1[i][j], end=" ")
        print()

    print("\n--------------------------\n");

    # class 2
    f = open("Class2.txt","r")
    fl =f.readlines()[0:N]
    i=0
    x1=0
    x2=0
    X_class2 = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            X_class2[i][j] = (lines[j])
        
            if j==0:
                x1 = x1+float(lines[0])
            elif j==1:
                x2 = x2+float(lines[1])
        i=i+1
    f.close()

   
    print(x1," --- ", x2)
    mean_x_class2 = [x1/N, x2/N];
    print(mean_x_class2[0]," --- ", mean_x_class2[1])

    cov_mat_class2 = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;

    for i in range(d):
        for j in range(d):
            cov_mat_class2[i][j]=0
            for k in range(len(X_class2)):
                cov_mat_class2[i][j] += (float(X_class2[k][i])-float(mean_x_class2[i]))*(float(X_class2[k][j])-float(mean_x_class2[j]))
            
            cov_mat_class2[i][j] /= N-1
           
            print(cov_mat_class2[i][j], end=" ")
        print()

    print("\n--------------------------\n");


    # class 2
    f = open("Class3.txt","r")
    fl =f.readlines()[0:N]
    i=0
    x1=0
    x2=0
    X_class3 = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            X_class3[i][j] = (lines[j])
            if j==0:
                x1 = x1+float(lines[0])
            elif j==1:
                x2 = x2+float(lines[1])
        i=i+1
    f.close()

    print(x1," --- ", x2)
    mean_x_class3 = [x1/N, x2/N];
    print(mean_x_class3[0]," --- ", mean_x_class3[1])

    cov_mat_class3 = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;

    for i in range(d):
        for j in range(d):
            cov_mat_class3[i][j]=0
            for k in range(len(X_class3)):
                cov_mat_class3[i][j] += (float(X_class3[k][i])-float(mean_x_class3[i]))*(float(X_class3[k][j])-float(mean_x_class3[j]))
            
            cov_mat_class3[i][j] /= N-1
            print(cov_mat_class3[i][j], end=" ")

        print()
    print("\n--------------------------\n");

    #CASE - B (All covaraince matrix are just equal)

    cov_matrix_b = [[0 for x in range(d)] for y in range(d)] # general covariance matrix of size 2x2;

    for i in range(d):
        for j in range(d):
            cov_matrix_b[i][j] = cov_mat_class1[i][j] + cov_mat_class2[i][j] + cov_mat_class3[i][j]
            cov_matrix_b[i][j] /= 3
            if(i!=j):
                cov_matrix_b[i][j] *= -1

    temp = cov_matrix_b[0][0]
    cov_matrix_b[0][0] = cov_matrix_b[1][1]
    cov_matrix_b[1][1] = temp;

    temp = (cov_matrix_b[0][0]*cov_matrix_b[1][1]) - (cov_matrix_b[0][1]*cov_matrix_b[1][0])
    # print (temp);
    # print (temp);
    for i in range(d):
        for j in range(d):
            cov_matrix_b[i][j] /= temp;

    print(cov_matrix_b[0][0]," ",cov_matrix_b[0][1]," \n",cov_matrix_b[1][0]," ",cov_matrix_b[1][1],)

    prior_class1 = (0.75*500)/(0.75*500 + 0.75*500 + 0.75*500)
    prior_class2 = prior_class3 = prior_class1

    print(prior_class1, " ", prior_class2, " ", prior_class3)

    # testing gij = wij.x + w0ij
    
    #for  g12, wij = inverse of covariance matrx * (u1-u2)

    # u1-u2
    temp_w12 = [(mean_x_class1[0]-mean_x_class2[0]), (mean_x_class1[1]-mean_x_class2[1])]
    print(temp_w12)
    w12 = [((temp_w12[0]*cov_matrix_b[0][0])+(temp_w12[1]*cov_matrix_b[0][1])), (temp_w12[0]*cov_matrix_b[1][0])+(temp_w12[1]*cov_matrix_b[1][1])]
    print(w12)

    t = [((mean_x_class1[0]*cov_matrix_b[0][0])+(mean_x_class1[0]*cov_matrix_b[1][0])), ((mean_x_class1[0]*cov_matrix_b[0][1])+(mean_x_class1[0]*cov_matrix_b[1][1]))]
    w012 = ((t[0]*mean_x_class1[0])+(t[1]*mean_x_class1[1]))


    t = [((mean_x_class2[0]*cov_matrix_b[0][0])+(mean_x_class2[0]*cov_matrix_b[1][0])), ((mean_x_class2[0]*cov_matrix_b[0][1])+(mean_x_class2[0]*cov_matrix_b[1][1]))]
    w012 = w012 - ((t[0]*mean_x_class2[0])+(t[1]*mean_x_class2[1]))

    w012 /= -2;

    print(w012);
           
    # testing for g12

    # f = open("Class1.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # for lines in fl:
    #     lines=lines.split();
    #     # print(lines[0]," ",lines[1])
    #     g12 = (float(lines[0])*w12[0]) + (float(lines[1])*w12[1]) 
    #     g12 += w012

        # print("g12 = ", g12);

    # def my_formula_12(x):
    #     return (w12[1]*x+w012)/(-1*w12[0])

    # graph(my_formula_12, range(-10, 30))

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

    # plt.show()


    #for  g12, wij = inverse of covariance matrx * (u1-u2)

    # u1-u2
    temp_w13 = [(mean_x_class1[0]-mean_x_class3[0]), (mean_x_class1[1]-mean_x_class3[1])]
    print(temp_w13)
    w13 = [((temp_w13[0]*cov_matrix_b[0][0])+(temp_w13[1]*cov_matrix_b[0][1])), (temp_w13[0]*cov_matrix_b[1][0])+(temp_w13[1]*cov_matrix_b[1][1])]
    print(w13)

    t = [((mean_x_class1[0]*cov_matrix_b[0][0])+(mean_x_class1[0]*cov_matrix_b[1][0])), ((mean_x_class1[0]*cov_matrix_b[0][1])+(mean_x_class1[0]*cov_matrix_b[1][1]))]
    w013 = ((t[0]*mean_x_class1[0])+(t[1]*mean_x_class1[1]))


    t = [((mean_x_class3[0]*cov_matrix_b[0][0])+(mean_x_class3[0]*cov_matrix_b[1][0])), ((mean_x_class3[0]*cov_matrix_b[0][1])+(mean_x_class3[0]*cov_matrix_b[1][1]))]
    w013 = w013 - ((t[0]*mean_x_class3[0])+(t[1]*mean_x_class3[1]))

    w013 /= -2;

    print(w013);
           
    # testing for g13

    f = open("Class1.txt","r")
    fl =f.readlines()[N:500]
    f.close
    for lines in fl:
        lines=lines.split();
        # print(lines[0]," ",lines[1])
        g13 = (float(lines[0])*w13[0]) + (float(lines[1])*w13[1]) 
        g13 += w013

        # print("g13 = ", g13);

    # def my_formula_13(x):
    #     return (w13[1]*x+w013)/(-1*w13[0])

    # graph(my_formula_13, range(-10, 30))

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

    # plt.plot(x,y,'ro')
    # # print(x)

    # plt.show()

    #for  g12, wij = inverse of covariance matrx * (u1-u2)

    # u1-u2
    temp_w23 = [(mean_x_class2[0]-mean_x_class3[0]), (mean_x_class2[1]-mean_x_class3[1])]
    print(temp_w23)
    w23 = [((temp_w23[0]*cov_matrix_b[0][0])+(temp_w23[1]*cov_matrix_b[0][1])), (temp_w23[0]*cov_matrix_b[1][0])+(temp_w23[1]*cov_matrix_b[1][1])]
    print(w23)

    t = [((mean_x_class2[0]*cov_matrix_b[0][0])+(mean_x_class2[0]*cov_matrix_b[1][0])), ((mean_x_class2[0]*cov_matrix_b[0][1])+(mean_x_class2[0]*cov_matrix_b[1][1]))]
    w023 = ((t[0]*mean_x_class2[0])+(t[1]*mean_x_class2[1]))


    t = [((mean_x_class3[0]*cov_matrix_b[0][0])+(mean_x_class3[0]*cov_matrix_b[1][0])), ((mean_x_class3[0]*cov_matrix_b[0][1])+(mean_x_class3[0]*cov_matrix_b[1][1]))]
    w023 = w023 - ((t[0]*mean_x_class3[0])+(t[1]*mean_x_class3[1]))

    w023 /= -2;

    print(w023);
           
    # testing for g13

    # f = open("Class3.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # for lines in fl:
    #     lines=lines.split();
    #     # print(lines[0]," ",lines[1])
    #     g23 = (float(lines[0])*w23[0]) + (float(lines[1])*w23[1]) 
    #     g23 += w023

        # print("g23 = ", g23);

    def my_formula_23(x):
        return (w23[1]*x+w023)/(-1*w23[0])

    graph(my_formula_23, range(-10, 30))

    x=[]
    y=[]
    f = open("Class2.txt","r")
    fl =f.readlines()[N:500]
    f.close
    i=0
    for lines in fl:
        lines=lines.split();
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        i+=1

    # print(x)
    plt.plot(x,y,'bs')

    f = open("Class3.txt","r")
    fl =f.readlines()[N:500]
    f.close
    x=[]
    y=[]
    i=0
    for lines in fl:
        lines=lines.split();
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        i+=1

    plt.plot(x,y,'ro')
    # print(x)

    plt.show()



if __name__== "__main__":
  main()

  # 1611275.5956010565