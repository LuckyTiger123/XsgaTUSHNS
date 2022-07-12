import math
from threading import Thread
import numpy as np

thread_num = 10

file_path = '/data/shangyihao/ppd/phase1_gdata.npz'
save_dir = '/data/shangyihao/ppd'

dataset = np.load(file_path)

dd_result = np.zeros((4059035, 1156))

slide_size = int(math.ceil(dataset['edge_index'].shape[0] / thread_num))

ths = []


def cal(t_n: int):
    edge_index = dataset['edge_index'][t_n * slide_size:(t_n + 1) * slide_size]
    edge_timestamp = dataset['edge_timestamp'][t_n * slide_size:(t_n + 1) * slide_size]
    for i in range(edge_index.shape[0]):
        timestamp = int(edge_timestamp[i])
        source_node = edge_index[i][0]
        target_node = edge_index[i][1]
        dd_result[source_node][timestamp * 2 - 2] += 1
        dd_result[target_node][timestamp * 2 - 1] += 1
        print('The {}th edge has been calculated.'.format(i))


for j in range(thread_num):
    th = Thread(target=cal(j))
    th.start()
    ths.append(th)

for th in ths:
    th.join()

# for i in range(dataset['edge_index'].shape[0]):
#     timestamp = int((dataset['edge_timestamp'][i] - 1422633600) / 86400)
#     source_node = dataset['edge_index'][i][0]
#     target_node = dataset['edge_index'][i][1]
#     dd_result[source_node][timestamp * 2] += 1
#     dd_result[target_node][timestamp * 2 + 1] += 1
#     print('The {}th edge has been calculated.'.format(i))

np.save(save_dir + '/origin_dd.npy', dd_result)
print('Mission completes!')
