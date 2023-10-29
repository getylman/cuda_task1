import matplotlib.pyplot as plt

with open('graph_data0.txt', 'r') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]

x_comp = list()

for i in range(5, 29):
    x_comp.append(i)
# заполнения оси Х (длянна использованного массива в задаче 2^x)

graphs = list()
# будем хранить все шрафики тут

graph_len = 24

cur_graph = list()

for i in data:
    cur_graph.append(i)
    if len(cur_graph) == graph_len:
        graphs.append(cur_graph.copy())
        cur_graph.clear()

for index, value in enumerate(graphs):
    plt.plot(x_comp, value, label=(
        'blocksize=' + str(2 ** (index + 5))))

# plt.ylim(1.68, 1.7)
# plt.xlim(26.07, 26.08)
# чтоб увидеть на сколько отличаются данные при длянне массива 2^26

# plt.ylim(5.3, 5.35)
# plt.xlim(27.7825, 27.8)
# to cmp in big len of arrs
plt.grid()
plt.xlabel('Array size')
# I used log2 of size in nature list of sizes is 32, 64,...
plt.ylabel('Time (ms)')
plt.title('Time depending on array size and block size')
plt.legend()

plt.show()
