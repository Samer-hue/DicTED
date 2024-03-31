import torch
from torch import nn, optim
from TREND.dgnn import DGNN
from TREND.film import Scale_4, Shift_4
from TREND.Emlp import EMLP
from TREND.node_relu import Node_edge


class Model(nn.Module):
    def __init__(self, neg_size, lr, ncoef, l2_reg, args):
        super(Model, self).__init__()
        #self.args = args
        self.l2reg = l2_reg  # 0.001
        self.ncoef = ncoef  # 0.01
        self.EMLP = EMLP(args)  # [1,d],[1]
        # self.grow_f = E_increase(args.edge_grow_input_dim)
        self.gnn = DGNN(args)  # Dynamic Graph Neural Network
        self.scale_e = Scale_4(args)
        self.shift_e = Shift_4(args)
        self.node_edge = Node_edge(args)
        self.neg_size = neg_size
        self.lr = lr

        # self.g_optim = optim.Adam(self.grow_f.parameters(), lr=args.lr)

        self.optim = optim.Adam([{'params': self.gnn.parameters()},
                                 {'params': self.EMLP.parameters()},
                                 {'params': self.scale_e.parameters()},
                                 {'params': self.shift_e.parameters()},
                                 {'params': self.node_edge.parameters()},
                                 ], lr=self.lr)

    def forward(self, s_node, t_node, s_self_feat, s_one_hop_feat, s_two_hop_feat,
                t_self_feat, t_one_hop_feat, t_two_hop_feat,
                neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                e_time, s_his_time, s_his_his_time,
                t_his_time, t_his_his_time,
                neg_his_time, neg_his_his_time,
                s_edge_rate, t_edge_rate,
                training=True):
        s_emb = self.gnn(s_self_feat, s_one_hop_feat, s_two_hop_feat,
                         e_time, s_his_time, s_his_his_time)  # [b,d]
        t_emb = self.gnn(t_self_feat, t_one_hop_feat, t_two_hop_feat,
                         e_time, t_his_time, t_his_his_time)  # [b,d]
        neg_embs = self.gnn(neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                            e_time, neg_his_time, neg_his_his_time, neg=True)  # [b,1,d]
        # 公式4

        ij_cat = torch.cat((s_emb, t_emb), dim=1)  # [128,32]
        alpha_ij = self.scale_e(ij_cat)
        beta_ij = self.shift_e(ij_cat)
        # 公式8、9
        theta_e_new = []
        for s in range(2):
            theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (alpha_ij[s] + 1)) + beta_ij[s])
            # 公式10，应该说theta有两个位置的值是因为a，b好像都有两个位置
            # EMLP的参数即公式10中的theta，是一个事件先验，其本质上是一个线性参数？
            # 为什么要循环两次?因为EMLP中有两个参数，一个是W，一个是b，第一次对W进行优化，第二次对b进行优化
            # theta_e_new即先验信息，与相似度是直接相乘的
            # 如果学习全局先验，就是用global_emb与该参数做个加权

        p_dif = (s_emb - t_emb).pow(2)
        p_scalar = (p_dif * theta_e_new[0]).sum(dim=1, keepdim=True)
        p_scalar += theta_e_new[1]
        p_scalar_list = p_scalar
        # 公式5，注意到theta_e_new有两个位置的存放值，分别为W和b

        event_intensity = torch.sigmoid(p_scalar_list) + 1e-6  # [b,1]
        log_event_intensity = torch.mean(-torch.log(event_intensity))  # [1]
        # 公式11正样本

        dup_s_emb = s_emb.repeat(1, 1, self.neg_size)
        dup_s_emb = dup_s_emb.reshape(s_emb.size(0), self.neg_size, s_emb.size(1))
        # [b,1,d]，等于是对s_emb的扩维

        neg_ij_cat = torch.cat((dup_s_emb, neg_embs), dim=2)
        neg_alpha_ij = self.scale_e(neg_ij_cat)
        neg_beta_ij = self.shift_e(neg_ij_cat)
        neg_theta_e_new = []
        for s in range(2):
            neg_theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (neg_alpha_ij[s] + 1)) + neg_beta_ij[s])

        neg_dif = (dup_s_emb - neg_embs).pow(2)
        neg_scalar = (neg_dif * neg_theta_e_new[0]).sum(dim=2, keepdim=True)
        neg_scalar += neg_theta_e_new[1]
        big_neg_scalar_list = neg_scalar

        neg_event_intensity = torch.sigmoid(- big_neg_scalar_list) + 1e-6

        neg_mean_intensity = torch.mean(-torch.log(neg_event_intensity))
        # 公式11负样本

        pos_l2_loss = [torch.norm(s, dim=1) for s in alpha_ij]
        pos_l2_loss = [torch.mean(s) for s in pos_l2_loss]
        pos_l2_loss = torch.sum(torch.stack(pos_l2_loss))
        pos_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=1)) for s in beta_ij]))
        neg_l2_loss = torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_alpha_ij]))
        neg_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_beta_ij]))

        l_theta = pos_l2_loss + neg_l2_loss
        l_theta = l_theta * self.l2reg
        delta_e = self.node_edge(s_emb)
        # node_edge本质上是线性操作再加一层relu，得到的是预测边的数量
        smooth_loss = nn.SmoothL1Loss()
        l_node = smooth_loss(delta_e, s_edge_rate.reshape(s_edge_rate.size(0), 1))
        # s_edge_rate即当前时刻实际边的数量
        # l_node = torch.sqrt(l_node)
        l_node = self.ncoef * l_node

        node_pred = delta_e
        node_truth = s_edge_rate.reshape(s_edge_rate.size(0), 1)

        L = log_event_intensity + neg_mean_intensity + l_node + l_theta
        # 公式14，log_event_intensity + neg_mean_intensity对应l_e，l_node对应l_n，l2_loss对应第三部分

        if training == True:
            self.optim.zero_grad()
            L.backward()
            self.optim.step()

        return round((L.detach().clone()).cpu().item(), 4),\
               s_emb.detach().clone().cpu(),\
               t_emb.detach().clone().cpu(),\
               dup_s_emb.detach().clone().cpu(),\
               neg_embs.detach().clone().cpu(),\
               node_pred.detach().clone().cpu(),\
               node_truth.detach().clone().cpu(),\
               s_node.detach().clone().cpu(),\
               t_node.detach().clone().cpu()

