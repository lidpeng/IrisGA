import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNetExtend
from optimizer import *
import numpy as np
import sys, os
from sklearn.preprocessing import Normalizer
from ga import *
import time

start_time = time.time()
iris_data = pd.read_csv('Iris.csv')
#对标签进行one-hot独热编码
iris_data =iris_data.join(pd.get_dummies(iris_data.Species))
del iris_data["Species"]
#对数据集进行随机打乱，并划分训练集与测试集
index = np.random.permutation(len(iris_data))
# 用生成的乱的索引就能将其打乱了
iris_data = iris_data.iloc[index ,:]

use_norm = False
split_rate = 0.8 #训练集测试集比例
#划分训练集/测试集
if use_norm:
    train_x= Normalizer().fit_transform(iris_data.iloc[:int(iris_data.shape[0]*split_rate), 1:-3])
    test_x = Normalizer().fit_transform(iris_data.iloc[int(iris_data.shape[0]*split_rate):, 1:-3])
    train_y= Normalizer().fit_transform(iris_data.iloc[:int(iris_data.shape[0]*split_rate), -3:])
    test_y = Normalizer().fit_transform(iris_data.iloc[int(iris_data.shape[0]*split_rate):, -3:])
else:
    train_x= iris_data.iloc[:int(iris_data.shape[0]*split_rate), 1:-3]
    test_x = iris_data.iloc[int(iris_data.shape[0]*split_rate):, 1:-3]
    train_y= iris_data.iloc[:int(iris_data.shape[0]*split_rate), -3:]
    test_y = iris_data.iloc[int(iris_data.shape[0]*split_rate):, -3:]
sys.path.append(os.pardir)

#optimizer = Momentum(lr=0.01)

optimizer = SGD(lr=0.1)
# 读入数据
(x_train, t_train), (x_test, t_test) = \
    (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


###ga_params
sol_per_pop = 8
num_parents_mating = 4
num_generations = 100
mutation_percent = 20 #变异比例
fitness_all_gen = []

###net_params
iters_num = 20000
train_size = x_train.shape[0]
batch_size = 2
learning_rate = 0.1
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
#选择是否使用遗传算法
use_ga = True


iter_per_epoch = 10  # max(train_size / batch_size, 1)
max_acc_step = 0
flag = 0

initial_pop_params = []
network = MultiLayerNetExtend(input_size=4, hidden_size_list=[8], output_size=3)
for i in range(sol_per_pop):
    initial_pop_params.append(MultiLayerNetExtend(input_size=4, hidden_size_list=[8], output_size=3).params)

#new_paras = vector_trans(mat_trans(initial_pop_weights),initial_pop_nets)
#print(mat_trans(initial_pop_weights).shape)
#print(vector_trans(mat_trans(initial_pop_weights),initial_pop_weights))

for i in range(iters_num):
    vec_weights = mat_trans(initial_pop_params)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(t_batch.shape)

    params = network.params
    if use_ga:
        #使用遗传算法
        fitness_per_gen = []
        #获得该代中所有物种的fitness
        for pop_param in initial_pop_params:
            #network.params = pop_param
            network.change_weight(pop_param)
            fitness = network.accuracy(x_test, t_test)
            #可以选择用训练集或者验证集来作为fitness的标注，验证集学习到的会更具有泛化性

            fitness_per_gen.append(fitness)
        #print(fitness_per_gen[0])
        fitness_all_gen.append(max(fitness_per_gen)) #将该代最优fitness存储
        #print('max_fitness = ',max(fitness_per_gen))

        max_fitness_idx = numpy.where(fitness_per_gen == numpy.max(fitness_per_gen))
        max_fitness_idx = max_fitness_idx[0][0]

        parents = select_mating_pool(vec_weights,fitness_per_gen.copy(),num_parents_mating)
        #print('parents = ',parents)
        #选择优势个体

        offspring_crossover = crossover(parents,offspring_size= (vec_weights.shape[0]-parents.shape[0],vec_weights.shape[1]))
        #print('Crossover = ', offspring_crossover)
        #基因交叉

        offspring_mutation = mutation(offspring_crossover,mutation_percent=mutation_percent,fitness = fitness_per_gen)
        #print("Mutation = ",offspring_mutation)
        #基因突变

        vec_weights[0:parents.shape[0], :] = parents
        vec_weights[parents.shape[0]:, :] = offspring_mutation
        initial_pop_params = vector_trans(vec_weights, initial_pop_params)
    else:
        #当不使用遗传算法时，使用梯度下降优化
        #通过误差反向传播法求梯度
        grad = network.gradient(x_batch, t_batch)
        optimizer.update(params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        if use_ga:
            network.change_weight(initial_pop_params[max_fitness_idx])

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_loss = network.loss(x_train, t_train)
        test_loss = network.loss(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_acc>=max(test_acc_list):
            if flag == 0:    #记录第一次达到100%的epochs
                max_acc_step = i

            if test_acc == 1:
                flag = 1

        print('iter:', i, 'acc:', train_acc, test_acc)
        print('loss:', i, 'loss:', train_loss, test_loss)

    # if not (test_acc) > max(test_acc_list):
    #     # flag_train = True   #防止网络被过度训练
    #     # break
    #     improved_str = ''
    #     pass

plt.plot(train_loss_list)
plt.savefig('train_loss_list_node8.png')
plt.show()
plt.plot(test_acc_list)
plt.savefig('test_acc_list_node8.png')
plt.show()
plt.plot(test_loss_list)
plt.savefig('test_loss_list_node8.png')
plt.show()
plt.plot(train_acc_list)
plt.savefig('train_acc_list_node8.png')
plt.show()
print('best_train_acc:',max(train_acc_list))
print('best_test_acc:',max(test_acc_list))
print('max_acc_step:',max_acc_step)
end_time = time.time()
print('spend_time:',end_time-start_time)