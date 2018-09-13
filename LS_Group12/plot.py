import numpy as np
import matplotlib.pyplot as plt

w2_12 = [[1,1],[1,1]]
w1_12 = [1,2]
w0_12 = -123.6968181881104

w=[1,1]
w0=0

x = np.linspace(10, 30, 400)
y = np.linspace(-15, 5	, 400)
X, Y = np.meshgrid(x,y)
# F = (w2_12[0][0])*(X**2) + w2_12[1][1]*(Y**2) + X*Y*(w2_12[0][1]+w2_12[1][0]) + w1_12[1]*X + w1_12[0]*Y + w0_12 
# F = (w[0]*X)+(w[1]*Y)+w0
F = ((X**2)+(Y**2))
plt.contour(X,Y,F,[0,1,2])
# plt.show()






# import numpy as np  
# import matplotlib.pyplot as plt  

# def graph(formula, x_range):  
# 	x = np.array(x_range)  
# 	y = formula(x)
# 	lines = plt.plot(y,x)
# 	# plt.axis([-20, 20, -20, 20])
# 	plt.setp(lines, color='r', linewidth=2.0)
# 	# plt.show()

# w12 = [2,-1]
# w012=3

# def my_formula(x):
#     return (w12[0]*x+w012)/(-1*w12[1])

# graph(my_formula, range(-10, 11))

# plt.title("A Sine Curve")
# plt.xlabel("x")
# plt.ylabel("sin(x)");


# N=375

# # x=[0.0 for i in range(N)]
# # y=[0.0 for i in range(N)]
x=[]
y=[]
f = open("Class1.txt","r")
fl =f.readlines()
f.close
i=0
for lines in fl:
    lines=lines.split();
    x.append(float(lines[0]))
    y.append(float(lines[1]))
    i+=1

# plt.plot(x,y,'ro') # blue squares

# x=[]
# y=[]

# f = open("Class2.txt","r")
# fl =f.readlines()[N:500]
# f.close
# i=0
# for lines in fl:
#     lines=lines.split();
#     x.append(float(lines[0]))
#     y.append(float(lines[1]))
#     i+=1

# plt.plot(x,y,'ro') #red circles
plt.axis([-200, 400, -200, 200])
plt.show()