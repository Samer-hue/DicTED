import opt
import torch
import numpy as np
from GAE import IGAE
from utils import setup_seed
from train import Pretrain_gae
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph

setup_seed(1)

print("use cuda: {}".format(opt.args.cuda))
# device = 'cuda'
device = 'cpu'

opt.args.data_path = './matrix/{}_matrix.txt'.format(opt.args.name)
opt.args.graph_save_path = './graph/gae_{}_graph.txt'.format(opt.args.name)

print("Data: {}".format(opt.args.data_path))

x = np.loadtxt(opt.args.data_path, dtype=float)


pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)

adj = load_graph(opt.args.k, opt.args.graph_save_path, opt.args.data_path).to(device)
data = torch.Tensor(dataset.x).to(device)

model_gae = IGAE(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    gae_n_dec_1=opt.args.gae_n_dec_1,
    gae_n_dec_2=opt.args.gae_n_dec_2,
    gae_n_dec_3=opt.args.gae_n_dec_3,
    n_input=opt.args.n_components,
).to(device)

Pretrain_gae(model_gae, data, adj, opt.args.gamma_value)
