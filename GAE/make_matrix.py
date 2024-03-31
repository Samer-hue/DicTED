import networkx as nx
import numpy as np

# 从文件中读取edgelist
def read_edgelist(file_path):
    with open(file_path, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file.readlines()]
    return edges

# 创建邻接矩阵
def create_adjacency_matrix(edges):
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edges)

    # 获取节点数
    num_nodes = len(G.nodes())

    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 填充邻接矩阵
    for edge in edges:
        adjacency_matrix[edge[0] - 1, edge[1] - 1] = 1
        adjacency_matrix[edge[1] - 1, edge[0] - 1] = 1

    return adjacency_matrix

# 将邻接矩阵输出到txt文档
def save_adjacency_matrix(matrix, output_path):
    np.savetxt(output_path, matrix, fmt='%d')

# 文件路径
for i in ['wikipedia']:
    edgelist_file = '../../data/ml_{}.edgelist'.format(i)
    output_txt_file = './matrix/{}_matrix.txt'.format(i)

    # 读取edgelist
    edges = read_edgelist(edgelist_file)

    # 创建邻接矩阵
    adj_matrix = create_adjacency_matrix(edges)

    # 保存邻接矩阵到txt文档
    save_adjacency_matrix(adj_matrix, output_txt_file)
