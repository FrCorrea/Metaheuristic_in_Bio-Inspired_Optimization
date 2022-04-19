import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import *
from numpy import meshgrid
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure("Lista de Exercicios 1",figsize=plt.figaspect(0.5))

# Função Schewefel

ax = fig.add_subplot(1,3,1, projection='3d')

def funcao_schwefel(x1,x2):
    return 418.9829*2 - x1*sin(sqrt(abs(x1))) - x2*sin(sqrt(abs(x2)))

x1 = linspace(-500,500)
x2 = linspace(-500,500)

x1,x2 = np.meshgrid(x1,x2)
results = funcao_schwefel(x1,x2)

ax.plot_surface(x1,x2,results,cmap='viridis')
ax.title.set_text("Função Schewefel")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Função Rastrigin

ax = fig.add_subplot(1,3,2, projection='3d')

def funcao_rastrigin(x,y):
    return 20 + pow(x, 2) + pow(y, 2) - 10 * (cos(2 * pi * x) + cos(2 * pi * y))

x = linspace(-5,5)
y = linspace(-5,5)

x,y = np.meshgrid(x,y)
results = funcao_rastrigin(x,y)

ax.plot_surface(x,y,results,cmap='viridis')
ax.title.set_text("Função Rastrigin")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Função Exponencial

ax = fig.add_subplot(1,3,3, projection='3d')

def funcao_exp(x,y):
    return x * exp(-(pow(x,2) + pow(y,2)))

x = linspace(-2,2)
y = linspace(-2,2)

x,y = np.meshgrid(x,y)
results = funcao_exp(x,y)

ax.plot_surface(x,y,results,cmap='viridis')
ax.title.set_text("Função Exponencial")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig("functionsPlot.png")
plt.show()
