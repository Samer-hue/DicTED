from __future__ import print_function, division
#import argparse
#import random
#import numpy as np
from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
#from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
#from torch.utils.data import DataLoader
from torch.nn import Linear
from SDCN.utils import load_data, load_graph
from SDCN.GNN import GNNLayer
#from evaluation import eva
#from collections import Counter

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        print('x:',x)
        print(x.shape)
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        ## 由于没有model.pkl文件，这个代码被注释了，是否恰当？？
        #self.ae.load_state_dict(torch.load('./SDCN/model.pkl', map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        ## 尽管本任务与聚类无关，为了防止出现异常，cluster_layer还是被保留了
        ## n_clusters都被设置为172，n_z都被设置为10，是否合理？
        ## 在源代码中，对不同数据集，n_clusters有一特定值
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, n_input, n_z, n_clusters, device, lr, name, k):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=n_input,
                n_z=n_z,
                n_clusters=n_clusters,
                v=1.0).to(device)
    print(model)

    '''
    label_path = '../../data/%s/label.txt' % (args.name)
    y = []
    with open(label_path, 'r') as reader:
        for line in reader:
            label = int(line)
            y.append(label)
    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    '''

    optimizer = Adam(model.parameters(), lr=lr)

    # KNN Graph
    adj, n = load_graph(name)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    #y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    ## 下面这行代码，直接运行会报错，，可能是由于kmeans.cluster_centers_这个名字不对？
    ## 直接注释掉这个代码，是否合理？？
    #model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(200):
        # torch.cuda.memory_summary(device=0, abbreviated=False)
        # print(f"当前GPU显存已分配: {torch.cuda.memory_allocated(device=0) / 1024**2:.2f} MB")
        # print(f"峰值GPU显存已分配: {torch.cuda.max_memory_allocated(device=0) / 1024**2:.2f} MB")
        _, tmp_q, pred, _, emb = model(data, adj)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)

        x_bar, q, pred, _, emb = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')  # l_clu
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')  # l_gcn
        re_loss = F.mse_loss(x_bar, data)  # l_res

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    '''
        acc, nmi, ari, f1 = eva(args.n_clusters, y, emb)
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1

    print('Best performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
          (best_acc, best_nmi, best_ari, best_f1))
    '''

    save_node_embeddings(n, emb, './processed/{}/{}_SDCN.emb'.format(name, name))


def save_node_embeddings(node_num, emb, path):
    embeddings = emb.cpu().data.numpy()
    writer = open(path, 'w')
    dim = len(embeddings[0])
    writer.write('%d %d\n' % (node_num, dim))
    for n_idx in range(node_num):
        writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

    writer.close()


def SDCN_main(dataset_name, n_input, n_clusters):
    #data = 'school'
    #k_dict = {'dblp': 10, 'arxivAI': 5, 'arxivCS': 40, 'school': 9, 'brain': 10, 'patent': 6}
    '''
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default=data)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    '''
    k = 3
    lr = 1e-3
    n_z = 10
    #pretrain_path = './SDCN/model.pkl'
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    # device = torch.device("cuda" if args.cuda else "cpu")
    device = 'cuda'

    #args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(dataset_name)

    #k = None
    #n_input = 172

    #print(args)
    train_sdcn(dataset, n_input, n_z, n_clusters, device, lr, dataset_name, k)
