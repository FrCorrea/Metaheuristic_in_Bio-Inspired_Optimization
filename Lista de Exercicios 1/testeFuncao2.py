import algGenetico as AG
import statistics as sta
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib import cm
import numpy as np
from numpy import *
from numpy import meshgrid
from mpl_toolkits.mplot3d.axes3d import Axes3D

# plot funcao_rastrigin

fig = plt.figure("Lista de Exercicios 1",figsize=plt.figaspect(1))
ax = fig.add_subplot(1,1,1, projection='3d')

def funcao_rastrigin(x,y):
    return 20 + pow(x, 2) + pow(y, 2) - 10 * (cos(2 * pi * x) + cos(2 * pi * y))

x1 = linspace(-5,5)
x2 = linspace(-5,5)

x1,x2 = np.meshgrid(x1,x2)
results = funcao_rastrigin(x1,x2)

ax.plot_surface(x1,x2,results,cmap='viridis')
ax.title.set_text("Função Rastrigin")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# x_min, x_max, y_min, y_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes, theta, *args
algoritmo_genetico = AG.AlgoritmoGenetico(-5, 5, -5, 5, 30, 5, 70, 50, 5, 0.7)
algoritmo_genetico.calcular_fitness("funcao_rastrigin")
x_data, y_data, z_data = [], [], []
# Teste da Seleção por torneio e Crossover no ponto
for i in range(algoritmo_genetico.num_geracoes):
    print("Geração {} = Fitness Maximo: {} Fitness Medio: {} Fitness Minimo: {}".format(i, max(algoritmo_genetico.fitness), sta.mean(algoritmo_genetico.fitness), min(algoritmo_genetico.fitness)))
    print("Resultado {}: {}".format(i, algoritmo_genetico.encontrar_filho_mais_apto()))
    if max(algoritmo_genetico.fitness) - min(algoritmo_genetico.fitness) < 0.001:
        break
    mais_apto = algoritmo_genetico.encontrar_filho_mais_apto()[0]
    x_data += [mais_apto[0]]
    y_data += [mais_apto[1]]
    z_data += [algoritmo_genetico.retorna_custo(mais_apto, "funcao_rastrigin")]
    nova_populacao = []
    while len(nova_populacao) < algoritmo_genetico.tam_populacao:
        # seleciona os pais
        pai = algoritmo_genetico.selecao_torneio_proporcional()
        mae = algoritmo_genetico.selecao_torneio_proporcional()
        # realiza o crossover dos pais para gerar os filhos
        filho_1, filho_2 = algoritmo_genetico.crossover_arithmetic(pai, mae)
        # realiza a mutação dos filhos e os adiciona à nova população
        algoritmo_genetico.mutar(filho_1)
        algoritmo_genetico.mutar(filho_2)
        nova_populacao.append(filho_1)
        nova_populacao.append(filho_2)
    # substitui a população antiga pela nova e realiza sua avaliação
    algoritmo_genetico.populacao = nova_populacao
    algoritmo_genetico.calcular_fitness("funcao_rastrigin")

graph = ax.scatter(x_data, y_data, z_data, c=[[1, 0, 0]])

def update_graph(num):
    if num < 50:
        graph._offsets3d = (x_data, y_data, z_data)
        ax.title.set_text('Iteração = {}'.format(num))

ani = matplotlib.animation.FuncAnimation(fig, update_graph)

plt.show()