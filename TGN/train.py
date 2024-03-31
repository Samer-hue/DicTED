import math
import torch
import numpy as np
import time

from TGN.tgn import TGN
from TGN.data_processing import get_data, compute_time_statistics
from TGN.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
#from TGN.cluster_evaluation import eva

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FType = torch.FloatTensor
LType = torch.LongTensor


class Train:
    def __init__(self, dataset, neg_size, hist_len, directed, epoch, batch_size, lr, node_dim, time_dim,
               seed, ncoef, l2_reg, gpu, save_step, prefix, n_head, n_layer, patience, n_runs, drop_out,
               backprop_every, use_memory, embedding_module, message_function, memory_updater, aggregator,
               memory_update_at_end, message_dim, memory_dim, different_new_nodes, uniform,
               randomize_features, use_destination_embedding_in_message, use_source_embedding_in_message,
               dyrep):
        #self.args = args
        #self.file_path = './processed/{}/{}_time.txt'.format(dataset, dataset)
        self.emb_path = './processed/{}/{}_TGN.emb'.format(dataset, dataset)
        self.data, self.node_num, self.edge_num = get_data(dataset)
        self.memory_dim = memory_dim
        #self.label_path = '../../data/%s/label.txt' % dataset
        #self.labels = self.read_label()
        ## 数据的索引从1开始，而不是0
        self.node_feature = np.ones((self.node_num + 1, memory_dim))
        self.edge_feature = np.ones((self.edge_num + 1, memory_dim))
        self.final_emb = torch.ones((self.node_num + 1, memory_dim))
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0
        self.best_epoch = 0
        self.gpu = gpu
        self.uniform = uniform
        self.n_runs = n_runs
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_out = drop_out
        self.use_memory = use_memory
        self.message_dim = message_dim
        self.embedding_module = embedding_module
        self.message_function = message_function
        #self.clusters = clusters
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.seed = seed
        self.ncoef = ncoef
        self.l2_reg = l2_reg
        self.save_step = save_step
        self.prefix = prefix
        self.patience = patience
        self.backprop_every = backprop_every
        self.randomize_features = randomize_features
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep
        self.different_new_nodes = different_new_nodes
        self.memory_updater = memory_updater
        self.aggregator = aggregator
        self.memory_update_at_end = memory_update_at_end

        print("The dataset has {} nodes and {} interactions.".format(self.node_num+1, self.edge_num+1))
    '''
    def read_label(self):
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                label = int(line)
                labels.append(label)
        return labels
    '''

    def process(self):
        #args = self.args
        GPU = self.gpu
        data = self.data

        # 下述几行定义的是类函数，之后可以调用
        # Initialize validation and test neighbor finder to retrieve temporal graph
        print('开始')
        full_ngh_finder = get_neighbor_finder(data, self.uniform)

        # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        rand_sampler = RandEdgeSampler(data.sources, data.destinations, seed=0)

        # Set device
        # device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
        device_string = 'cpu'
        device = torch.device(device_string)

        # Compute time statistics
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
            compute_time_statistics(data.sources, data.destinations, data.timestamps)

        final_emb = self.final_emb
        print('马上进入循环')
        for i in range(self.n_runs):
            # Initialize Model
            tgn = TGN(neighbor_finder=full_ngh_finder, node_features=self.node_feature,
                      edge_features=self.edge_feature, device=device,
                      n_layers=self.n_layer,
                      n_heads=self.n_head, dropout=self.drop_out, use_memory=self.use_memory,
                      message_dimension=self.message_dim, memory_dimension=self.memory_dim,
                      memory_update_at_start=not self.memory_update_at_end,
                      embedding_module_type=self.embedding_module,
                      message_function=self.message_function,
                      aggregator_type=self.aggregator,
                      memory_updater_type=self.memory_updater,
                      n_neighbors=self.hist_len,
                      mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                      mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                      use_destination_embedding_in_message=self.use_destination_embedding_in_message,
                      use_source_embedding_in_message=self.use_source_embedding_in_message,
                      dyrep=self.dyrep)
            criterion = torch.nn.BCELoss()  # 这个损失函数好像只出0和1的值
            optimizer = torch.optim.Adam(tgn.parameters(), lr=self.lr)
            tgn = tgn.to(device)

            num_instance = len(data.sources)
            num_batch = math.ceil(num_instance / self.batch_size)

            idx_list = np.arange(num_instance)

            val_aps = []
            epoch_times = []
            total_epoch_times = []
            train_losses = []

            early_stopper = EarlyStopMonitor(max_round=self.patience)
            for epoch in range(self.epoch):
                print('第{}次epoch开始'.format(epoch))
                start_epoch = time.time()
                # Training

                # Reinitialize memory of the model at the start of each epoch
                if self.use_memory:
                    tgn.memory.__init_memory__()

                # Train using only training graph
                tgn.set_neighbor_finder(full_ngh_finder)
                m_loss = []

                for k in range(0, num_batch, self.backprop_every):
                    loss = 0
                    optimizer.zero_grad()

                    # Custom loop to allow to perform backpropagation only every a certain number of batches
                    for j in range(self.backprop_every):
                        batch_idx = k + j

                        if batch_idx >= num_batch:
                            continue

                        start_idx = batch_idx * self.batch_size
                        end_idx = min(num_instance, start_idx + self.batch_size)
                        sources_batch, destinations_batch = data.sources[start_idx:end_idx], \
                                                            data.destinations[start_idx:end_idx]
                        edge_idxs_batch = data.edge_idxs[start_idx: end_idx]
                        timestamps_batch = data.timestamps[start_idx:end_idx]

                        size = len(sources_batch)
                        _, negatives_batch = rand_sampler.sample(size)

                        with torch.no_grad():
                            pos_label = torch.ones(size, dtype=torch.float, device=device)
                            neg_label = torch.zeros(size, dtype=torch.float, device=device)

                        tgn = tgn.train()
                        pos_prob, neg_prob, s_emb, d_emb = tgn.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                            negatives_batch,
                                                                            timestamps_batch, edge_idxs_batch,
                                                                            self.hist_len)

                        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                    loss /= self.backprop_every

                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())

                    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                    # the start of time
                    if self.use_memory:
                        tgn.memory.detach_memory()

                    final_emb[sources_batch] = s_emb.cpu()
                    final_emb[destinations_batch] = d_emb.cpu()

                epoch_time = time.time() - start_epoch
                epoch_times.append(epoch_time)

                ### Validation
                # Validation uses the full graph
                tgn.set_neighbor_finder(full_ngh_finder)

                if self.use_memory:
                    # Backup memory at the end of training, so later we can restore it and use it for the
                    # validation on unseen nodes
                    train_memory_backup = tgn.memory.backup_memory()

                if self.use_memory:
                    val_memory_backup = tgn.memory.backup_memory()
                    # Restore memory we had at the end of training to be used when validating on new nodes.
                    # Also backup memory after validation so it can be used for testing (since test edges are
                    # strictly later in time than validation edges)
                    tgn.memory.restore_memory(train_memory_backup)

                if self.use_memory:
                    # Restore memory we had at the end of validation
                    tgn.memory.restore_memory(val_memory_backup)

                train_losses.append(np.mean(m_loss))

                total_epoch_time = time.time() - start_epoch
                total_epoch_times.append(total_epoch_time)
                '''
                acc, nmi, ari, f1 = eva(self.clusters, self.labels, final_emb)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_nmi = nmi
                    self.best_ari = ari
                    self.best_f1 = f1
                    self.best_epoch = epoch

                print('epoch %d: loss=%.4f  ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' % (
                epoch, np.mean(m_loss), acc, nmi, ari, f1))

            print('Best performance in %d epoch: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
                  (self.best_epoch, self.best_acc, self.best_nmi, self.best_ari, self.best_f1))
            '''
            self.save_node_embeddings(final_emb, self.emb_path)

        self.save_node_embeddings(final_emb, self.emb_path)

    def save_node_embeddings(self, emb, path):
        embeddings = emb.cpu().data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_num, self.memory_dim))
        for n_idx in range(self.node_num):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
