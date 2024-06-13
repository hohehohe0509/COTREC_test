import time
from util import Data
from model import *
import os
import argparse
import pickle
import dgl
import logging

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Retailrocket', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=7, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--kg_batch_size', type=int, default=100, help='KG batch size.')
parser.add_argument('--embSize', type=int, default=112, help='embedding size')
parser.add_argument('--kg_embSize', type=int, default=64, help='embedding size')
parser.add_argument('--relation_embSize', type=int, default=112, help='Relation Embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')#用不到
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.001, help='ssl task maginitude')
parser.add_argument('--eps', type=float, default=0.2, help='eps') #這個用不到
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')#多加的
parser.add_argument('--layer_size', nargs='?', default='[64, 32, 16]', help='Output sizes of every layer')
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--drop_rate', type=float, default=0.7, help='CL dropout rate.')
parser.add_argument('--alpha', type=float, default=0.) #這個用不到
parser.add_argument('--adj_type', nargs='?', default='si',help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
parser.add_argument('--batch_size_cl', type=int, default=8192, help='CL batch size.')
parser.add_argument('--cl_alpha', type=float, default=1.)#也用不到
parser.add_argument('--temperature', type=float, default=0.7, help='Softmax temperature.') #也用不到
parser.add_argument('--K', type=int, default=10, help='')
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# parser.add_argument('--lr_dc_step', type=int, default=2, help='the number of steps after which the learning rate decay 3')


opt = parser.parse_args()
print(opt)

logging.basicConfig(
    filename='./clear/%s_test1.log' % opt.dataset, 
    level=logging.INFO, 
    format='%(asctime)s,%(msecs)03d [%(levelname)s] %(name)s: %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info(opt)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)

def main():
    init_seed(2023)
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

    if opt.dataset == 'Tmall':
        n_item = 41512
        n_node = 50840
    elif opt.dataset == 'KKBOX':
        n_item = 23838
        n_node = 60826
    elif opt.dataset == 'Retailrocket':
        n_item = 25390
        n_node = 73946
        
    train_data = Data(opt.dataset, train_data,all_train,opt, shuffle=True, n_item=n_item, n_node=n_node, KG=True)
    test_data = Data(opt.dataset, test_data,all_train,opt, shuffle=True, n_item=n_item, n_node=n_node, KG=False)
    ret_num = train_data.n_session + train_data.n_items
    
    ##新加的
    weight_size = eval(opt.layer_size)
    num_layers = len(weight_size) - 2
    heads = [opt.heads] * num_layers + [1]
    adjM = train_data.lap_list

    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to('cuda')

    edge2type = {}
    for i,mat in enumerate(train_data.kg_lap_list):
        for u,v in zip(*mat.nonzero()):
            edge2type[(u,v)] = i
    for i in range(train_data.n_entities):
        edge2type[(i,i)] = len(train_data.kg_lap_list)
    
    kg_adjM = sum(train_data.kg_lap_list)
    kg = dgl.DGLGraph(kg_adjM)
    kg = dgl.remove_self_loop(kg)
    kg = dgl.add_self_loop(kg)
    e_feat = []
    for u, v in zip(*kg.edges()):
        u = u.item()
        v = v.item()
        if u == v:
            break
        e_feat.append(edge2type[(u,v)])
    for i in range(train_data.n_entities):
        e_feat.append(edge2type[(i,i)])
    #e_feat裡面存每個triple的relation是誰，是一個list（按照kg.edges()的順序存）
    e_feat = torch.tensor(e_feat, dtype=torch.long).to('cuda')
    kg = kg.to('cuda')
    
    #有KG
    model = trans_to_cuda(COTREC(adjacency=train_data.adjacency,n_node=n_node,n_item=train_data.n_items,n_relations=train_data.n_relations,opt=opt,num_layers=num_layers, num_hidden=weight_size[-2], num_classes=weight_size[-1],  heads=heads, activation=F.elu, feat_drop=0.1, attn_drop=0., negative_slope=0.01, residual=False,emb_size=opt.embSize, relation_embSize=opt.relation_embSize, ret_num=ret_num))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        logging.info('-------------------------------------------------------')
        logging.info('epoch: %d'% epoch)
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, kg, g, epoch, opt.drop_rate)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        logging.info(metrics)
        print(metrics)
        for K in top_K:
            logging.info('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
    
    logging.info('-------------------------------------------------------')


if __name__ == '__main__':
    main()