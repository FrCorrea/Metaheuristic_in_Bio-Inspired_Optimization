import random
from random import randint
import numpy as np
from numpy import *
import random

class AlgoritmoGenetico:
    def __init__(self, x_min, x_max, y_min, y_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes, theta, *args):

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.tam_populacao = tam_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes
        self.fitness = None
        self.theta = theta
        self.custo = None
        if len(args) == 1: self.alfa = args[0]
        # gera os individuos da população
        self._gerar_populacao()

    def _gerar_populacao(self):
        self.populacao = [[] for i in range(self.tam_populacao)]
        # preenche a população
        for individuo in self.populacao:
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            individuo.append(x)
            individuo.append(y)

    def funcao_schwefel(self, x1, x2):
        return 418.9829 * 2 - x1 * sin(sqrt(abs(x1))) - x2 * sin(sqrt(abs(x2)))

    def funcao_rastrigin(self, x, y):
        return 20 + pow(x, 2) + pow(y, 2) - 10 * (cos(2 * pi * x) + cos(2 * pi * y))

    def funcao_exp(self, x, y):
        return x * exp(-(pow(x, 2) + pow(y, 2)))

    def retorna_custo(self, individuo, funcao):
        if funcao == "funcao_schwefel":
            return self.funcao_schwefel(individuo[0], individuo[1])

        if funcao == "funcao_rastrigin":
            return self.funcao_rastrigin(individuo[0], individuo[1])

        if funcao == "funcao_exp":
            return self.funcao_exp(individuo[0], individuo[1])

    def calcular_fitness(self, funcao):
        self.fitness = []
        if funcao == "funcao_schwefel":
            for individuo in self.populacao:
                self.fitness.append(self.funcao_schwefel(individuo[0], individuo[1]))
            cost_sum = sum(self.fitness)
            for i, av in enumerate(self.fitness):
                self.fitness[i] = 1 / (1 + (av / cost_sum))

        if funcao == "funcao_rastrigin":
            for individuo in self.populacao:
                self.fitness.append(self.funcao_rastrigin(individuo[0], individuo[1]))
            cost_sum = sum(self.fitness)
            for i, av in enumerate(self.fitness):
                self.fitness[i] = 1 / (1 + (av / cost_sum))

        if funcao == "funcao_exp":
            for individuo in self.populacao:
                self.fitness.append(self.funcao_exp(individuo[0], individuo[1]))
            cost_sum = sum(self.fitness)
            print(cost_sum)
            for i, av in enumerate(self.fitness):
                self.fitness[i] = av / cost_sum

    # Estrategias de Seleção

    def selecao_torneio(self):
        # agrupa os individuos com as suas avaliações para gerar os participantes do torneio
        participantes_torneio = list(zip(self.populacao, self.fitness))
        # escolhe dois individuos aleatoriamente
        individuo_1 = participantes_torneio[randint(0, self.tam_populacao - 1)]
        individuo_2 = participantes_torneio[randint(0, self.tam_populacao - 1)]
        # retorna individuo com a maior avaliação, ou seja, o vencedor do torneio
        return individuo_1[0] if individuo_1[1] >= individuo_2[1] else individuo_2[0]

    # Roleta
    def weighted_random_roulette(self, participantes_torneio):
        max_value = sum(participante[1] for participante in participantes_torneio)
        pick, current = random.uniform(0, max_value), 0
        for index, participante in enumerate(participantes_torneio):
            current += participante[1]
            if current > pick:
                return index

    def selecao_torneio_proporcional(self):
        # agrupa os individuos com as suas avaliações para gerar os participantes do torneio
        participantes_torneio = list(zip(self.populacao, self.fitness))
        # escolhe dois individuos aleatoriamente
        individuo_1 = participantes_torneio[self.weighted_random_roulette(participantes_torneio)]
        individuo_2 = participantes_torneio[self.weighted_random_roulette(participantes_torneio)]
        # retorna individuo com a maior avaliação, ou seja, o vencedor do torneio
        return individuo_1[0] if individuo_1[1] >= individuo_2[1] else individuo_2[0]

    # Estrategias de Crossover

    def ajusta_restricao_dominio(self, individuo):
        if individuo[0] < self.x_min: individuo[0] = self.x_min
        if individuo[0] > self.x_max: individuo[0] = self.x_max
        if individuo[1] < self.y_min: individuo[1] = self.y_min
        if individuo[1] > self.y_max: individuo[1] = self.y_max

    def crossover_point(self, pai, mae):
        if randint(1,100) <= self.taxa_crossover:
            # caso o crossover seja aplicado os pais trocam dividem seus genes X, Y nos para os filhos
            filho_1 = [pai[0], mae[1]]
            filho_2 = [mae[0], pai[1]]
            # verifica se o filho gerado ainda está no dominio x, y
            self.ajusta_restricao_dominio(filho_1)
            self.ajusta_restricao_dominio(filho_2)
        else:
            # caso contrário os filhos são cópias exatas dos pais
            filho_1 = pai
            filho_2 = mae
        # retorna os filhos obtidos pelo crossover
        return [filho_1, filho_2]

    def crossover_arithmetic(self, pai, mae):
        if randint(1,100) <= self.taxa_crossover:
            # caso o crossover seja aplicado e feita a combinação linear dos vetores pai utilizando um valor alfa
            filho_1 = [(1 - self.alfa) * pai[0] + self.alfa * mae[0], (1 - self.alfa) * pai[1] + self.alfa * mae[1]]
            filho_2 = [self.alfa * pai[0] + (1 - self.alfa) * mae[0], self.alfa * pai[1] + (1 - self.alfa) * mae[1]]
            # verifica se o filho gerado ainda está no dominio x, y
            self.ajusta_restricao_dominio(filho_1)
            self.ajusta_restricao_dominio(filho_2)
        else:
            # caso contrário os filhos são cópias exatas dos pais
            filho_1 = pai
            filho_2 = mae
        # retorna os filhos obtidos pelo crossover
        return [filho_1, filho_2]

    def mutar(self, individuo):
        # caso a taxa de mutação seja atingida, ela é realizada
        if randint(1,100) <= self.taxa_mutacao:
            indice = randint(0, 1)
            individuo[indice] = individuo[indice] + random.gauss(0, self.theta)
        # verifica se na mutação o novo individuo ainda esta no dominio x, y
        self.ajusta_restricao_dominio(individuo)

    def encontrar_filho_mais_apto(self):
        # agrupa os individuos com suas avaliações para gerar os candidatos
        candidatos = zip(self.populacao, self.fitness)
        # retorna o candidato com a melhor avaliação, ou seja, o mais apto da população
        return max(candidatos, key=lambda elemento: elemento[1])


