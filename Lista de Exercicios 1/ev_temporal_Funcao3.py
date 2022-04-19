import algGenetico as AG
import statistics as sta
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib import cm
import numpy as np
from numpy import *
from numpy import meshgrid
from mpl_toolkits.mplot3d.axes3d import Axes3D

# plot funcao_schwefel
fig = plt.figure("Lista de Exercicios 1",figsize=plt.figaspect(1))
ax = fig.add_subplot(1,1,1)

ax.title.set_text("Função_Exponecial")
ax.set_xlabel('Iterações')
ax.set_ylabel('Fitness')

# x_min, x_max, y_min, y_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes, theta, *args
algoritmo_genetico = AG.AlgoritmoGenetico(-2, 2, -2, 2, 30, 5, 70, 50, 0.5, 0.7)
algoritmo_genetico.calcular_fitness("funcao_exp")
ev_maximo, ev_medio, ev_minimo, num_ite = [], [], [], []
# Teste da Seleção por torneio e Crossover no ponto
for i in range(algoritmo_genetico.num_geracoes):
    print("Geração {} = Fitness Maximo: {} Fitness Medio: {} Fitness Minimo: {}".format(i, max(algoritmo_genetico.fitness), sta.mean(algoritmo_genetico.fitness), min(algoritmo_genetico.fitness)))
    print("Melhor Individuo i={}: {}".format(i, algoritmo_genetico.encontrar_filho_mais_apto()))
    num_ite += [i]
    ev_maximo += [max(algoritmo_genetico.fitness)]
    ev_medio += [sta.mean(algoritmo_genetico.fitness)]
    ev_minimo += [min(algoritmo_genetico.fitness)]
    if max(algoritmo_genetico.fitness) - min(algoritmo_genetico.fitness) < 0.001: break
    nova_populacao = []
    while len(nova_populacao) < algoritmo_genetico.tam_populacao:
        # seleciona os pais
        pai = algoritmo_genetico.selecao_torneio()
        mae = algoritmo_genetico.selecao_torneio()
        # realiza o crossover dos pais para gerar os filhos
        filho_1, filho_2 = algoritmo_genetico.crossover_arithmetic(pai, mae)
        # realiza a mutação dos filhos e os adiciona à nova população
        algoritmo_genetico.mutar(filho_1)
        algoritmo_genetico.mutar(filho_2)
        nova_populacao.append(filho_1)
        nova_populacao.append(filho_2)
    # substitui a população antiga pela nova e realiza sua avaliação
    algoritmo_genetico.populacao = nova_populacao
    algoritmo_genetico.calcular_fitness("funcao_exp")

plt.plot(num_ite, ev_minimo, label="Fitness Minimo")
plt.plot(num_ite, ev_medio, label="Fitness Médio")
plt.plot(num_ite, ev_maximo, label="Fitness Maximo")

plt.legend()

plt.show()

