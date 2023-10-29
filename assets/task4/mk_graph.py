import matplotlib.pyplot as plt

with open('graph_data41.txt', 'r') as f:
    lines = f.readlines()
    data1 = [float(line.strip()) for line in lines]

with open('graph_data42.txt', 'r') as f:
    lines = f.readlines()
    data2 = [float(line.strip()) for line in lines]

x_comp = list()

for i in range(5, 29):
    x_comp.append(i)
# заполнения оси Х (длянна использованного массива в задаче 2^x)

graphs1 = list()
graphs2 = list()
# будем хранить все шрафики тут

graph_len = 24

cur_graph = list()

for i in data1:
    cur_graph.append(i)
    if len(cur_graph) == graph_len:
        graphs1.append(cur_graph.copy())
        cur_graph.clear()

for i in data2:
    cur_graph.append(i)
    if len(cur_graph) == graph_len:
        graphs2.append(cur_graph.copy())
        cur_graph.clear()

# for index, value in enumerate(graphs1):
#    plt.plot(x_comp, value, label=(
#        'blocksize=' + str(2 ** (index + 5))))

for index, value in enumerate(graphs2):
    plt.plot(x_comp, value, label=(
        'blocksize=' + str(2 ** (index + 5))))

# plt.ylim(5.305, 5.34)
# plt.xlim(27.794, 27.8)
# to cmp in big len of arrs
plt.grid()
plt.xlabel('Array size: log2(real array size)')
# I used log2 of size in nature list of sizes is 32, 64,...
plt.ylabel('Time (ms)')
plt.title('Time depending on array size and block size')
plt.legend()

plt.show()
