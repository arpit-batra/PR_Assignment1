import numpy as np  
import matplotlib.pyplot as plt  

# def plot_line(w, c, min, max):
# 	y=[]
# 	i=0 
# 	j = min
# 	while j <max:
# 		y.append(((w[0]*j + w[1]*j) + c))
# 		j+=0.01 
# 	print(y);
	
# 	lines = plt.plot(y,x,0.01)
# 	plt.setp(lines, color='r', linewidth=2.0)

# w = [1,1]
# c=0
# plot_line(w, c, -100.0, 100.0)

N=500
l=1

x=[0.0 for i in range(N)]
y=[0.0 for i in range(N)]

f = open("Class1.txt","r")
fl =f.readlines()

f.close
i=0
for lines in fl:
    lines=lines.split();
    x[i] = float(lines[0])
    y[i] = float(lines[1])
    i+=1

plt.plot(x,y,'bs')

f = open("Class3.txt","r")
fl =f.readlines()
x=[0.0 for i in range(len(fl))]
y=[0.0 for i in range(len(fl))]
f.close
i=0
for lines in fl:
    lines=lines.split();
    x[i] = float(lines[0])
    y[i] = float(lines[1])
    i+=1

plt.plot(x,y,'ro')

plt.show()