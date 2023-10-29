import matplotlib.pyplot as plt

with open('graph_data3.txt', 'r') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]

x_comp = list()

for i in range(3, 11):
    for j in range(3, 11):
        x_comp.append(str(i) + '\n' + str(j) + '\n' + str(j))
# заполнения оси Х (длянна использованного массива в задаче 2^x)

graphs = list()
# будем хранить все шрафики тут

graph_len = 64

cur_graph = list()

for i in data:
    cur_graph.append(i)
    if len(cur_graph) == graph_len:
        graphs.append(cur_graph.copy())
        cur_graph.clear()

for index, value in enumerate(graphs):
    plt.plot(x_comp, value, label=(
        'blocksize=' + str(2 ** (index + 5))))

# plt.ylim(5.305, 5.34)
# plt.xlim(27.794, 27.8)
# to cmp in big len of arrs
plt.grid()
plt.xlabel('Matrix size log2(height) / log2(width) / log2(vector size)')
# I used log2 of size in nature list of sizes is 32, 64,...
plt.ylabel('Time (ms)')
plt.title('Time depending on array size and block size')
plt.legend()

plt.show()
