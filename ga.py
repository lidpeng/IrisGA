import numpy
import random
#本文件为一些遗传算法使用到的函数

# 从权重字典转化为一维向量
def mat_trans(mat_pop_weights):
    pop_weights_vector = []
    for sol_idx in range(len(mat_pop_weights)):
        curr_vector = []
        for layer_key in mat_pop_weights[0].keys():
            vector_weights = numpy.reshape(mat_pop_weights[sol_idx][layer_key], newshape=(mat_pop_weights[sol_idx][layer_key].size))
            curr_vector.extend(vector_weights)
        pop_weights_vector.append(curr_vector)
    return numpy.array(pop_weights_vector)

# 从一维向量转换回权重字典
def vector_trans(vector_pop_weights, mat_pop_weights):
    all_weights = []
    #print(len(mat_pop_weights))
    for sol_idx in range(len(mat_pop_weights)):
        start = 0
        end = 0
        mat_weights = {}
        for layer_key in mat_pop_weights[0].keys():
            end = end + mat_pop_weights[sol_idx][layer_key].size
            curr_vector = vector_pop_weights[sol_idx, start:end]
            mat_layer_weights = numpy.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx][layer_key].shape))
            mat_weights[layer_key] = mat_layer_weights
            #mat_weights.append(mat_layer_weights)
            start = end
        all_weights.append(mat_weights)
    return all_weights


def select_mating_pool(pop, fitness, num_parents):
    # 选择num_parents个最大的fitness所在位置作为优势遗传
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -999
    return parents

def crossover(parents, offspring_size):
    #交叉
    offspring = numpy.empty(offspring_size)

    crossover_point = numpy.uint32(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

def mutation(offspring_crossover, mutation_percent,fitness):
    #变异
    #此处加了自己设计的一个环境，根据fitness进行相应的变异
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value*(1-fitness[idx])*10 #随机添加新value
    return offspring_crossover
