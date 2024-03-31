import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Pretrain_gae(model, data, adj, gamma_value):

    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):

        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)

        z_igae, z_hat, adj_hat = model(data, adj)

        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss = loss_w + gamma_value * loss_a
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_node_embeddings(z_hat, './emb/%s_GAE.emb' % (args.name))

def save_node_embeddings(emb, path):
    embeddings = emb.cpu().data.numpy()
    writer = open(path, 'w')
    node_num = len(embeddings)
    dim = len(embeddings[0])
    writer.write('%d %d\n' % (node_num, dim))
    for n_idx in range(node_num):
        writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

    writer.close()


