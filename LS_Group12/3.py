from array import *
from math import *
import numpy as np
import matplotlib.pyplot as plt

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
    avg1=0.0;
    for i in range(d):
        for j in range(d):
            cov_mat_class1[i][j]=0
            for k in range(len(X_class1)):
                cov_mat_class1[i][j] += (float(X_class1[k][i])-float(mean_x_class1[i]))*(float(X_class1[k][j])-float(mean_x_class1[j]))
            
            cov_mat_class1[i][j] /= N-1
            avg1 += cov_mat_class1[i][j]
            # print(cov_mat_class1[i][j], end=" ")
        # print()
    
    cov_mat_class1[0][0] = cov_mat_class1[1][1] = avg1/4
    cov_mat_class1[1][0] = cov_mat_class1[0][1] = 0

    # for inverse of mtrix switch left to right diagonal values which are same inverse sign of other elements of diagonal which are same
    # and divide by determinanT which is (avg/4)^2 - 0 = (avg/4)^2

    cov_mat_class1[0][0] /= pow((avg1/4),2) 
    cov_mat_class1[1][1] /= pow((avg1/4),2)


    print(cov_mat_class1[0][0], "  ", cov_mat_class1[0][1], "\n", cov_mat_class1[1][0], "  ", cov_mat_class1[1][1])

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
    avg2=0.0
    for i in range(d):
        for j in range(d):
            cov_mat_class2[i][j]=0
            for k in range(len(X_class2)):
                cov_mat_class2[i][j] += (float(X_class2[k][i])-float(mean_x_class2[i]))*(float(X_class2[k][j])-float(mean_x_class2[j]))
            
            cov_mat_class2[i][j] /= N-1
            avg2 += cov_mat_class2[i][j]
            # print(cov_mat_class2[i][j], end=" ")
        # print()

    cov_mat_class2[0][0] = cov_mat_class2[1][1] = avg2/4
    cov_mat_class2[1][0] = cov_mat_class2[0][1] = 0

    # for inverse of mtrix switch left to right diagonal values which are same inverse sign of other elements of diagonal which are same
    # and divide by determinanT which is (avg/4)^2 - 0 = (avg/4)^2

    cov_mat_class2[0][0] /= pow((avg2/4),2) 
    cov_mat_class2[1][1] /= pow((avg2/4),2)

    print(cov_mat_class2[0][0], "  ", cov_mat_class2[0][1],"\n", cov_mat_class2[1][0], "  ", cov_mat_class2[1][1])

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
    avg3=0.0
    for i in range(d):
        for j in range(d):
            cov_mat_class3[i][j]=0
            for k in range(len(X_class3)):
                cov_mat_class3[i][j] += (float(X_class3[k][i])-float(mean_x_class3[i]))*(float(X_class3[k][j])-float(mean_x_class3[j]))
            
            cov_mat_class3[i][j] /= N-1
            avg3 += cov_mat_class3[i][j]
            # print(cov_mat_class3[i][j], end=" ")

        # print()

    cov_mat_class3[0][0] = cov_mat_class3[1][1] = avg3/4
    cov_mat_class3[1][0] = cov_mat_class3[0][1] = 0

    # for inverse of mtrix switch left to right diagonal values which are same inverse sign of other elements of diagonal which are same
    # and divide by determinanT which is (avg/4)^2 - 0 = (avg/4)^2

    cov_mat_class3[0][0] /= pow((avg3/4),2) 
    cov_mat_class3[1][1] /= pow((avg3/4),2)

    print(cov_mat_class3[0][0], "  ", cov_mat_class3[0][1],"\n", cov_mat_class3[1][0], "  ", cov_mat_class3[1][1])

    print("\n--------------------------\n");

    

    #CASE - C (All covaraince matrix are not equal but diagonal)

    prior_class1 = (0.75*500)/(0.75*500 + 0.75*500 + 0.75*500)
    prior_class2 = prior_class3 = prior_class1

    print(prior_class1, " ", prior_class2, " ", prior_class3)

    # # testing gij = x(t)*w2_ij*x + w1_ij(t)*x + w0_ij
     
    # for g12, w2_ij = (inverse of covariance matrx 1 - inverse of covariance matrx 2)  matrix of 2*2
    w2_12 = [[0.0 for x in range(d)] for y in range(d)]
    
    w2_12[0][0] = cov_mat_class1[0][0]-cov_mat_class2[0][0]
    w2_12[1][1] = cov_mat_class1[1][1]-cov_mat_class2[1][1]
    w2_12[1][0] = w2_12[0][1] = 0

    print(w2_12[0][0], " ",  w2_12[0][1])
    print(w2_12[1][0], " ",  w2_12[1][1])

    # for w1_12 - matrix of 2*1
    w1_12 = [( (cov_mat_class1[0][0]*mean_x_class1[0])+(cov_mat_class1[0][1]*mean_x_class1[1]) ), ( (cov_mat_class1[1][0]*mean_x_class1[0])+(cov_mat_class1[1][1]*mean_x_class1[1]) )  ]
    w1_12[0] -= ( (cov_mat_class2[0][0]*mean_x_class2[0])+(cov_mat_class2[0][1]*mean_x_class2[1]) )
    w1_12[1] -= ( (cov_mat_class2[1][0]*mean_x_class2[0])+(cov_mat_class2[1][1]*mean_x_class2[1]) )

    
    # w0_12 - constant
    temp = [((mean_x_class1[0]*cov_mat_class1[0][0])+(mean_x_class1[1]*cov_mat_class1[1][0])), ( (mean_x_class1[0]*cov_mat_class1[0][1])+(mean_x_class1[1]*cov_mat_class1[1][1]) )]
    w0_12 = ((temp[0]*mean_x_class1[0])+(temp[1]*mean_x_class1[1]))/(-1*2)
    print(w0_12)
    temp = [((mean_x_class2[0]*cov_mat_class2[0][0])+(mean_x_class2[1]*cov_mat_class2[1][0])), ( (mean_x_class2[0]*cov_mat_class2[0][1])+(mean_x_class2[1]*cov_mat_class2[1][1]) )]
    w0_12 += ((temp[0]*mean_x_class2[0])+(temp[1]*mean_x_class2[1]))/(2)
    print(w0_12)

    avg1 /= 4;
    avg2 /= 4;

    print(avg1)
    print(avg2) 

    w0_12 -= (log(pow(avg1,2)/pow(avg2,2)))/2;
    print(w0_12)    
    w0_12 += log(prior_class1/prior_class2);

    print(w0_12)
    # testing for g12

    # f = open("Class2.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # for lines in fl:
    #     lines=lines.split();
    #     # print(lines[0]," ",lines[1])
    #     x1 = float(lines[0])
    #     x2 = float(lines[1])
    #     # print(x1, " ", x2)
    #     temp = [( (x1*w2_12[0][0]) + (x2*w2_12[1][0]) ), ( (x1*w2_12[0][1]) + (x2*w2_12[1][1]) )]
    #     g12 = ( (temp[0]*x1) + (temp[1]*x2) )
        
    #     g12 += ( (w1_12[0]*x1) + (w1_12[1]*x2) )

    #     g12 += w0_12
        
        # print("g12 = ", g12);


    # x = np.linspace(-45, 30, 100)
    # y = np.linspace(-20, 40, 100)
    # X, Y = np.meshgrid(x,y)
    # F = (w2_12[0][0])*(X**2) + (w2_12[1][1])*(Y**2) + X*Y*(w2_12[0][1]+w2_12[1][0]) + w1_12[0]*X + w1_12[1]*Y + w0_12 
    # plt.contour(X,Y,F,[0])
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

    # plt.show()

    print( "\n-------------------------------------------------------------------------------------------------------------", "\n")
    # for g13, w2_ij = (inverse of covariance matrx i - inverse of covariance matrx j)  matrix of 2*2
    w2_13 = [[0.0 for x in range(d)] for y in range(d)]
    
    w2_13[0][0] = cov_mat_class1[0][0]-cov_mat_class3[0][0]
    w2_13[1][1] = cov_mat_class1[1][1]-cov_mat_class3[1][1]
    w2_13[1][0] = w2_13[0][1] = 0

    print(w2_13[0][0], " ",  w2_13[0][1])
    print(w2_13[1][0], " ",  w2_13[1][1])

    # for w1_13 - matrix of 2*1
    w1_13 = [( (cov_mat_class1[0][0]*mean_x_class1[0])+(cov_mat_class1[0][1]*mean_x_class1[1]) ), ( (cov_mat_class1[1][0]*mean_x_class1[0])+(cov_mat_class1[1][1]*mean_x_class1[1]) )  ]
    w1_13[0] -= ( (cov_mat_class3[0][0]*mean_x_class3[0])+(cov_mat_class3[0][1]*mean_x_class3[1]) )
    w1_13[1] -= ( (cov_mat_class3[1][0]*mean_x_class3[0])+(cov_mat_class3[1][1]*mean_x_class3[1]) )

    
    # w0_13 - constant
    temp = [((mean_x_class1[0]*cov_mat_class1[0][0])+(mean_x_class1[1]*cov_mat_class1[1][0])), ( (mean_x_class1[0]*cov_mat_class1[0][1])+(mean_x_class1[1]*cov_mat_class1[1][1]) )]
    w0_13 = ((temp[0]*mean_x_class1[0])+(temp[1]*mean_x_class1[1]))/(-1*2)
    print(w0_13)
    temp = [((mean_x_class3[0]*cov_mat_class3[0][0])+(mean_x_class3[1]*cov_mat_class3[1][0])), ( (mean_x_class3[0]*cov_mat_class3[0][1])+(mean_x_class3[1]*cov_mat_class3[1][1]) )]
    w0_13 += ((temp[0]*mean_x_class3[0])+(temp[1]*mean_x_class3[1]))/(2)
    print(w0_13)

    avg3 /= 4;

    print(avg1)
    print(avg3) 

    w0_13 -= (log(pow(avg1,2)/pow(avg3,2)))/2;
    print(w0_13)    
    w0_13 += log(prior_class1/prior_class2);

    print(w0_13)

    
    # testing for g13

    # f = open("Class1.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # for lines in fl:
    #     lines=lines.split();
    #     # print(lines[0]," ",lines[1])
    #     x1 = float(lines[0])
    #     x2 = float(lines[1])
    #     # print(x1, " ", x2)
    #     temp = [( (x1*w2_13[0][0]) + (x2*w2_13[1][0]) ), ( (x1*w2_13[0][1]) + (x2*w2_13[1][1]) )]
    #     g13 = ( (temp[0]*x1) + (temp[1]*x2) )
        
    #     g13 += ( (w1_13[0]*x1) + (w1_13[1]*x2) )

    #     g13 += w0_13
        
        # print("g13 = ", g13);

    print( "\n-------------------------------------------------------------------------------------------------------------", "\n")
    
    # for g23, w2_ij = (inverse of covariance matrx i - inverse of covariance matrx j)  matrix of 2*2
    w2_23 = [[0.0 for x in range(d)] for y in range(d)]
    
    w2_23[0][0] = cov_mat_class2[0][0]-cov_mat_class3[0][0]
    w2_23[1][1] = cov_mat_class2[1][1]-cov_mat_class3[1][1]
    w2_23[1][0] = w2_23[0][1] = 0

    print(w2_23[0][0], " ",  w2_23[0][1])
    print(w2_23[1][0], " ",  w2_23[1][1])

    # for w2_23 - matrix of 2*1
    w1_23 = [( (cov_mat_class2[0][0]*mean_x_class2[0])+(cov_mat_class2[0][1]*mean_x_class2[1]) ), ( (cov_mat_class2[1][0]*mean_x_class2[0])+(cov_mat_class2[1][1]*mean_x_class2[1]) )  ]
    w1_23[0] -= ( (cov_mat_class3[0][0]*mean_x_class3[0])+(cov_mat_class3[0][1]*mean_x_class3[1]) )
    w1_23[1] -= ( (cov_mat_class3[1][0]*mean_x_class3[0])+(cov_mat_class3[1][1]*mean_x_class3[1]) )

    
    # w0_23 - constant
    temp = [((mean_x_class2[0]*cov_mat_class2[0][0])+(mean_x_class2[1]*cov_mat_class2[1][0])), ( (mean_x_class2[0]*cov_mat_class2[0][1])+(mean_x_class2[1]*cov_mat_class2[1][1]) )]
    w0_23 = ((temp[0]*mean_x_class2[0])+(temp[1]*mean_x_class2[1]))/(-1*2)
    print(w0_23)
    temp = [((mean_x_class3[0]*cov_mat_class3[0][0])+(mean_x_class3[1]*cov_mat_class3[1][0])), ( (mean_x_class3[0]*cov_mat_class3[0][1])+(mean_x_class3[1]*cov_mat_class3[1][1]) )]
    w0_23 += ((temp[0]*mean_x_class3[0])+(temp[1]*mean_x_class3[1]))/(2)
    print(w0_23)

    # avg3 /= 4;

    print(avg2)
    print(avg3) 

    w0_23 -= (log(pow(avg2,2)/pow(avg3,2)))/2;
    print(w0_23)    
    w0_23 += log(prior_class2/prior_class3);

    print(w0_23)

    
    # testing for g13

    # f = open("Class2.txt","r")
    # fl =f.readlines()[N:500]
    # f.close
    # for lines in fl:
    #     lines=lines.split();
    #     # print(lines[0]," ",lines[1])
    #     x1 = float(lines[0])
    #     x2 = float(lines[1])
    #     # print(x1, " ", x2)
    #     temp = [( (x1*w2_23[0][0]) + (x2*w2_23[1][0]) ), ( (x1*w2_23[0][1]) + (x2*w2_23[1][1]) )]
    #     g23 = ( (temp[0]*x1) + (temp[1]*x2) )
        
    #     g23 += ( (w1_23[0]*x1) + (w1_23[1]*x2) )

    #     g23 += w0_23
        
        # print("g23 = ", g23);


    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x,y)
    F = (w2_12[0][0])*(X**2) + (w2_12[1][1])*(Y**2) + X*Y*(w2_12[0][1]+w2_12[1][0]) + w1_12[0]*X + w1_12[1]*Y + w0_12 
    plt.contour(X,Y,F,[0],cmap=plt.get_cmap('autumn'))
    F = (w2_13[0][0])*(X**2) + (w2_13[1][1])*(Y**2) + X*Y*(w2_13[0][1]+w2_13[1][0]) + w1_13[0]*X + w1_13[1]*Y + w0_13
    plt.contour(X,Y,F,[0],cmap=plt.get_cmap('winter'))
    F = (w2_23[0][0])*(X**2) + (w2_23[1][1])*(Y**2) + X*Y*(w2_23[0][1]+w2_23[1][0]) + w1_23[0]*X + w1_23[1]*Y + w0_23
    plt.contour(X,Y,F,[0],cmap=plt.get_cmap('spring'))
    # plt.show()    
    x=[]
    y=[]
    f = open("Class1.txt","r")
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

    f = open("Class2.txt","r")
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

    plt.plot(x,y,'gp')

    plt.show()


if __name__== "__main__":
  main()

  # 1611275.5956010565