import matplotlib.pyplot as plt

with open('graph_data61.txt', 'r') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]

x_comp = list()

for i in range(3, 12):
    for j in range(3, 12):
        for k in range(3, 12):
            x_comp.append(str(i) + '\n' + str(j) + '\n' + str(k))
# заполнения оси Х (длянна использованного массива в задаче 2^x)

graphs = list()
# будем хранить все шрафики тут

graph_len = 729

cur_graph = list()

for i in data:
    cur_graph.append(i)
    if len(cur_graph) == graph_len:
        graphs.append(cur_graph.copy())
        cur_graph.clear()

for index, value in enumerate(graphs):
    plt.plot(x_comp, value, label=(
        'blocksize=' + str(2 ** (index + 5))))

# plt.xlim(710, 729)
# to cmp in big sizes

plt.grid()
plt.xlabel('Matrix size log2(heightA) / log2(widthA) / log2(widthB)')
# I used log2 of size in nature list of sizes is 32, 64,...
plt.ylabel('Time (ms)')
plt.title('Time depending on array size and block size')
plt.legend()

plt.show()
