import sys

import os.path as osp
import time
import torch
import numpy as np
import random
import math
import time
from TREND.data_dyn_cite import DataHelper
from torch.utils.data import DataLoader
from TREND.model import Model
#from cluster_evaluation import eva

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
FType = torch.FloatTensor
LType = torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sub_main(device, file_path, node_feature_path, model_path, emb_path, feat_dim, neg_size, hist_len, directed, epoch_num, tlp_flag, batch_size, lr, hid_dim, out_dim, seed, ncoef, l2_reg, args):
    begin = time.time()
    setup_seed(seed)
    Data = DataHelper(file_path, node_feature_path, neg_size, hist_len, directed,
                      tlp_flag=tlp_flag)
    node_dim = Data.get_node_dim()
    node_emb = torch.zeros(node_dim + 1, out_dim)

    model = Model(neg_size, lr, ncoef, l2_reg, args).to(device)
    model.train()

    '''
    label_path = '../../data/%s/label.txt' % (args.data)
    labels = []
    with open(label_path, 'r') as reader:
        for line in reader:
            label = int(line)
            labels.append(label)

    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_epoch = 0
    '''

    for j in range(epoch_num):
        loader = DataLoader(Data, batch_size=batch_size, shuffle=True, num_workers=4)
        for i_batch, sample_batched in enumerate(loader):
            loss, s_emb, t_emb, dup_s_emb, neg_embs, node_pred, node_truth, s_node, t_node = model.forward(
                sample_batched['s_node'].type(LType).to(device),
                sample_batched['t_node'].type(LType).to(device),
                sample_batched['s_self_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['s_one_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['s_two_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),

                sample_batched['t_self_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['t_one_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['t_two_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),

                sample_batched['neg_self_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['neg_one_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),
                sample_batched['neg_two_hop_feat'].type(FType).reshape(-1, feat_dim).to_sparse().to(device),

                sample_batched['event_time'].type(FType).to(device),
                sample_batched['s_history_times'].type(FType).to(device),
                sample_batched['s_his_his_times_list'].type(FType).to(device),
                sample_batched['t_history_times'].type(FType).to(device),
                sample_batched['t_his_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_his_times_list'].type(FType).to(device),
                sample_batched['s_edge_rate'].type(FType).to(device),
                sample_batched['t_edge_rate'].type(FType).to(device),
            )

            node_emb[t_node] = t_emb
            node_emb[s_node] = s_emb
            if j == 0:
                if i_batch % 10 == 0:
                    print('batch_{} event_loss:'.format(i_batch), loss)
        '''
        acc, nmi, ari, f1 = eva(args.clusters, labels, node_emb)
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = j

        sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc, nmi, ari, f1))
        '''

        print('ep_{}_event_loss:'.format(j + 1), loss)
        # torch.cuda.memory_summary(device=0, abbreviated=False)
        # print(f"当前GPU显存已分配: {torch.cuda.memory_allocated(device=0) / 1024**2:.2f} MB")
        # print(f"峰值GPU显存已分配: {torch.cuda.max_memory_allocated(device=0) / 1024 ** 2:.2f} MB")

    '''
    print('Best performance in %d epoch: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
          (best_epoch, best_acc, best_nmi, best_ari, best_f1))
    '''
    #torch.save(model.state_dict(), args.model)
    end = time.time()
    print('Train Total Time: ' + str(round((end - begin)/60, 2)) + ' mins')
    save_node_embeddings(out_dim, node_emb, node_dim, emb_path)

def save_node_embeddings(out_dim, emb, node_dim, path):
    embeddings = emb.cpu().data.numpy()
    writer = open(path, 'w')
    writer.write('%d %d\n' % (node_dim, out_dim))
    for n_idx in range(node_dim):
        writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

    writer.close()

class MyConfig:
    def __init__(self, data, emb_path, file_path, node_feature_path, model_path, feat_dim, neg_size, hist_len,
                 directed, epoch_num, tlp_flag, batch_size, lr, hid_dim, out_dim, seed, ncoef, l2_reg):
        self.data = data
        self.emb_path = emb_path
        self.file_path = file_path
        self.node_feature_path = node_feature_path
        self.model_path = model_path
        self.feat_dim = feat_dim
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.epoch_num = epoch_num
        self.tlp_flag = tlp_flag
        self.batch_size = batch_size
        self.lr = lr
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.seed = seed
        self.ncoef = ncoef
        self.l2_reg = l2_reg


def TREND_main(data,dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    file_path = './processed/{}/{}_time.txt'.format(data, data)
    node_feature_path = './processed/{}/{}.txt'.format(data, data)
    model_path = './processed/{}/trend_model.pkl'.format(data)
    emb_path = './processed/{}/{}_TREND.emb'.format(data, data)
    feat_dim = dim
    neg_size = 1
    hist_len = 10
    directed = False
    epoch_num = 15
    tlp_flag = True
    batch_size = 1000
    lr = 0.001
    hid_dim = dim
    out_dim = dim
    seed = 1
    ncoef = 0.01
    l2_reg = 0.001
    
    args = MyConfig(data, emb_path, file_path, node_feature_path, model_path, feat_dim, neg_size, hist_len,
                    directed, epoch_num, tlp_flag, batch_size, lr, hid_dim, out_dim, seed, ncoef, l2_reg)

    sub_main(device, file_path, node_feature_path, model_path, emb_path, feat_dim, neg_size, hist_len, directed, epoch_num, tlp_flag, batch_size, lr, hid_dim, out_dim, seed, ncoef, l2_reg, args)
    