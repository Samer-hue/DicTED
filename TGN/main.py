import torch
import datetime
import argparse

from TGN import train

FType = torch.FloatTensor
LType = torch.LongTensor


class GenerEmb:
    def __init__(self):
        pass


def main_TGN(dataset, memory_dim):
    #k_dict = {'dblp': 10, 'arxivAI': 5, 'arxivCS': 40, 'school': 9, 'brain': 10, 'patent': 6, 'bitotc': 21}

    # 直接设置参数变量
    #dataset = data
    #clusters = k_dict[data]
    neg_size = 10
    hist_len = 10
    directed = False
    epoch = 10
    batch_size = 10000
    lr = 0.001
    node_dim = 100
    time_dim = 100
    seed = 1
    ncoef = 0.01
    l2_reg = 0.0001
    gpu = 0
    save_step = 10
    prefix = 'tgn-attn'
    n_head = 2
    n_layer = 1
    patience = 5
    n_runs = 1
    drop_out = 0.1
    backprop_every = 1
    use_memory = True
    embedding_module = "graph_attention"
    message_function = "identity"
    memory_updater = "gru"
    aggregator = "last"
    memory_update_at_end = True
    message_dim = 100
    #memory_dim = 128
    different_new_nodes = True
    uniform = True
    randomize_features = True
    use_destination_embedding_in_message = True
    use_source_embedding_in_message = True
    dyrep = True

    start = datetime.datetime.now()
    pret = train.Train(dataset, neg_size, hist_len, directed, epoch, batch_size, lr, node_dim, time_dim,
               seed, ncoef, l2_reg, gpu, save_step, prefix, n_head, n_layer, patience, n_runs, drop_out,
               backprop_every, use_memory, embedding_module, message_function, memory_updater, aggregator,
               memory_update_at_end, message_dim, memory_dim, different_new_nodes, uniform,
               randomize_features, use_destination_embedding_in_message, use_source_embedding_in_message,
               dyrep)
    pret.process()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))

'''
if __name__ == '__main__':
    data = 'arxivCS'
    k_dict = {'dblp': 10, 'arxivAI': 5, 'arxivCS': 40, 'school': 9, 'brain': 10, 'patent': 6, 'bitotc': 21}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--clusters', type=int, default=k_dict[data])
    parser.add_argument('--neg_size', type=int, default=10)
    parser.add_argument('--hist_len', type=int, default=10)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ncoef', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--save_step', type=int, default=10)

    parser.add_argument('--prefix', type=str, default='tgn-attn', help='Prefix to name the checkpoints')
    # 可选项：tgn-attn  JODIE-jodie_rnn，DyRep-dyrep_rnn
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', default=True,
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    args = parser.parse_args()

    main_train(args)

'''