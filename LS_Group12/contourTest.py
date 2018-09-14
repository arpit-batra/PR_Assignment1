from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors
import matplotlib.cm as cm
import matplotlib.mlab as mlab


xmin =  -9.1856
ymin =  -14.321
xmax =  28.653
ymax =  16.992

w11=-0.5104281345144983
w22=-0.873727190955755
w0=-3.9939410793687014

x = np.linspace(xmin,xmax,10000)
y = np.linspace(ymin,ymax,10000)
X, Y = np.meshgrid(x, y)
Z=(w11*X)+(w22*Y)+w0
print (Z)
# #plt.figure()
plt.contourf(X, Y, Z)
#plt.figure()
plt.show()   
