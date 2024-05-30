from torch_geometric.datasets import *
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Dataset
import os
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import Evaluator
from datasets import load_dataset, load_metric
from data.pyg_datasets.pyg_dataset import GraphormerPYGDataset, Graphtrans_Sampling_Dataset,Graphtrans_Sampling_Dataset_v2
from torch_geometric.data import DataLoader


def get_loss_and_metric(data_name):

    if data_name in ['ZINC','pcqm4mv2','QM7','QM9','ZINC-full']:
        loss = nn.L1Loss(reduction='mean')
        metric =  nn.L1Loss(reduction='mean')
        task_type='regression'
        metric_name = 'MAE'

    elif data_name in ['UPFD',"COX2_MD", "BZR_MD", "PTC_FM", "DHFR_MD", "PROTEINS", "DBLP_v1"]:
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        metric = load_metric("accuracy.py")
        task_type='binary_classification'
        metric_name='accuracy'

    elif data_name in ["ogbg-molhiv","ogbg-molbace","ogbg-molbbbp"]:
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        metric = Evaluator(name=data_name)
        task_type='binary_classification'
        metric_name='ROC-AUC'

    elif data_name in ["Cora","Computers","PubMed","Wisconsin","Actor","Flickr",'ogbn-products','ogbn-arxiv']:
        loss = nn.CrossEntropyLoss(reduction='mean')
        metric = load_metric('accuracy.py')
        task_type='multi_classification'
        metric_name='accuracy'
    
    elif data_name in ["ogbg-molpcba"]:
        loss = nn.BCEWithLogitsLoss(reduction='mean')

        metric = Evaluator(name=data_name)
        task_type='multi_binary_classification'
        metric_name='AP'

    else:
        raise ValueError('no such dataset')

    return loss, metric, task_type,metric_name



def normalization(data_list,mean,std):
    for i in tqdm(range(len(data_list))):
        data_list[i] = (data_list[i].x-mean)/std
    return data_list


def get_graph_level_dataset(name,param=None,seed=1024,set_default_params=False,args=None):

    path = 'dataset/'+name
    print(path)
    train_set = None
    val_set = None
    test_set = None
    inner_dataset = None
    train_idx=None
    val_idx=None
    test_idx=None

    #graph regression
    if name=='ZINC':#250,000 molecular graphs with up to 38 heavy atoms
        # inner_dataset  = Dataset()
        train_set = ZINC(path,subset=True,split='train')
        val_set = ZINC(path,subset=True,split='val')
        test_set = ZINC(path,subset=True,split='test')
        args.node_feature_type='cate'
        args.num_class =1
        args.eval_steps=1000
        args.save_steps=1000
        args.greater_is_better = False

    elif name == 'ZINC-full':  # 250,000 molecular graphs with up to 38 heavy atoms
        train_set = ZINC(path, subset=False, split='train')
        val_set = ZINC(path, subset=False, split='val')
        test_set = ZINC(path, subset=False, split='test')
        args.node_feature_type = 'cate'
        args.num_class = 1
        args.eval_steps = 1000
        args.save_steps = 1000
        args.greater_is_better = False
    
    elif name in ["MNIST", "CIFAR10"]:  # benchGNNs
        
        train_set = GNNBenchmarkDataset(root=path, name=name, split='train')
        val_set = GNNBenchmarkDataset(root=path, name=name, split='val')
        test_set = GNNBenchmarkDataset(root=path, name=name, split='test')
        args.node_feature_dim = train_set.num_node_features
        args.node_feature_type = 'dense'
        args.num_class = 10
        args.eval_steps = 1000
        args.save_steps = 1000
        args.greater_is_better = True

    elif name == "ogbg-molpcba":
        inner_dataset = PygGraphPropPredDataset(name)
        idx_split = inner_dataset.get_idx_split()
        train_idx = idx_split["train"]
        val_idx = idx_split["valid"]
        test_idx = idx_split["test"]
        args.node_feature_type = 'cate'
        args.num_class = 128
        args.eval_steps = 2000
        args.save_steps = 2000
        args.greater_is_better = True


    elif name in ["ogbg-molhiv","ogbg-molbace","ogbg-molbbbp"]:
        inner_dataset = PygGraphPropPredDataset(name)
        idx_split = inner_dataset.get_idx_split()
        train_idx = idx_split["train"]
        val_idx = idx_split["valid"]
        test_idx = idx_split["test"]
        args.node_feature_type = 'cate'
        args.num_class = 1
        args.eval_steps = 1000
        args.save_steps = 1000
        args.greater_is_better = True
        # args.warmup_steps = 4000
        # args.max_steps = 10000 # 1200000

    elif name in ["COX2_MD", "BZR_MD", "PTC_FM", "DHFR_MD", "PROTEINS", "DBLP_v1"]:
        inner_dataset = TUDataset(root=path, name=name)
        inner_dataset.data.edge_attr = None
        args.node_feature_type = 'dense'
        args.node_feature_dim=inner_dataset.num_node_features
        args.num_class = 1
        args.eval_steps = 1000
        args.save_steps = 1000
        args.greater_is_better = True

    elif name=='UPFD' and param in ('politifact', 'gossipcop'):
        train_set = UPFD(path,param,'bert',split='train')
        val_set = UPFD(path,param,'bert',split='val')
        test_set = UPFD(path,param,'bert',split='test')
        args.learning_rate=1e-5
        args.node_feature_type='dense'
        args.node_feature_dim=768
        args.greater_is_better = True



    else:
        raise ValueError('no such dataset')


    dataset = GraphormerPYGDataset(
        dataset=inner_dataset,
        train_idx=train_idx,
        valid_idx=val_idx,
        test_idx=test_idx,
        train_set=train_set,
        valid_set=val_set,
        test_set=test_set,
        seed=seed,
        args=args
                )
    return dataset.train_data,dataset.valid_data,dataset.test_data, inner_dataset

from sklearn.model_selection import StratifiedKFold
import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
import numpy as np

def gen_uniform_60_20_20_split(data):
    skf = StratifiedKFold(5, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return torch.cat(idx[:3], 0), torch.cat(idx[3:4], 0), torch.cat(idx[4:], 0)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_622(data):
    split = gen_uniform_60_20_20_split(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data

def get_node_level_dataset(name,param=None,args=None):
    path = 'dataset/' + name
    print(path)

    # if args.sampling_algo=='shadowkhop':
    #     args.num_neighbors=20
    # elif args.sampling_algo=='sage':
    #     args.num_neighbors=20

    # "Cora","Computers","PubMed","Wisconsin","Actor","Flickr" false
    if name in ['cora','citeseer','dblp','pubmed']:
        dataset = CitationFull(f'dataset/{name}',name)

    elif name in ["Cora","PubMed"]:
        # dataset = CitationFull(f'dataset/{name}','cora')
        dataset = Planetoid('./dataset/', name)
        args.node_feature_dim=dataset.num_features
        args.node_feature_type='dense'
        args.num_class = dataset.num_classes
        dataset = dataset[0]
        dataset = split_622(dataset)

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=4000
        # args.max_steps=10000

        x_norm_func = lambda x:x

        train_idx = dataset.train_mask.nonzero().squeeze()
        valid_idx = dataset.val_mask.nonzero().squeeze()
        test_idx = dataset.test_mask.nonzero().squeeze()
    
    elif name in ["Computers"]:
        dataset = Amazon('./dataset/', 'Computers')
        args.node_feature_dim=dataset.num_features
        args.num_class = dataset.num_classes
        dataset = dataset[0]
        dataset = split_622(dataset)
        args.node_feature_type='dense'

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=4000
        # args.max_steps=10000

        x_norm_func = lambda x:x

        train_idx = dataset.train_mask.nonzero().squeeze()
        valid_idx = dataset.val_mask.nonzero().squeeze()
        test_idx = dataset.test_mask.nonzero().squeeze()
    
    elif name in ["Wisconsin"]:
        
        edge_file = path + '/out1_graph_edges.txt'
        feature_file = path + '/out1_node_feature_label.txt'
        mask_file = path + '/wisconsin_split_0.6_0.2_0.npz'
        data = open(feature_file).readlines()[1:]
        x = []
        y = []
        for i in data:
            tmp = i.rstrip().split('\t')
            y.append(int(tmp[-1]))
            tmp_x = tmp[1].split(',')
            tmp_x = [int(fi) for fi in tmp_x]
            x.append(tmp_x)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y)

        edges = open(edge_file)
        edges = edges.readlines()
        edge_index = []
        for i in edges[1:]:
            tmp = i.rstrip()
            tmp = tmp.split('\t')
            edge_index.append([int(tmp[0]), int(tmp[1])])
            edge_index.append([int(tmp[1]), int(tmp[0])])
        # edge_index = np.array(edge_index).transpose(1, 0)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        print('edge_index:', edge_index.size())


        # mask
        mask = np.load(mask_file)
        train_mask = torch.from_numpy(mask['train_mask.npy']).to(torch.bool)
        val_mask = torch.from_numpy(mask['val_mask.npy']).to(torch.bool)
        test_mask = torch.from_numpy(mask['test_mask.npy']).to(torch.bool)

        dataset = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        args.node_feature_dim=dataset.num_features
        args.num_class = 5
        dataset = split_622(dataset)
        args.node_feature_type='dense'

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=4000
        # args.max_steps=10000

        x_norm_func = lambda x:x

        train_idx = dataset.train_mask.nonzero().squeeze()
        valid_idx = dataset.val_mask.nonzero().squeeze()
        test_idx = dataset.test_mask.nonzero().squeeze()

    elif name == "Actor":
        dataset = Actor("./dataset/Actor/")

        args.node_feature_dim = dataset.num_features
        args.node_feature_type='dense'
        args.num_class = dataset.num_classes
        
        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True
        
        # args.warmup_steps=4000
        # args.max_steps=100000

        x_norm_func = lambda x:x

        train_idx = dataset.data.train_mask.nonzero().squeeze()
        valid_idx = dataset.data.val_mask.nonzero().squeeze()
        test_idx = dataset.data.test_mask.nonzero().squeeze()


    elif name =="Flickr":
        dataset = Flickr(path)
        x_norm_func = lambda x:x #

        args.node_feature_dim=500
        args.node_feature_type='dense'
        args.num_class =7

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=4000
        # args.max_steps=10000

        train_idx = dataset.data.train_mask.nonzero().squeeze()
        valid_idx = dataset.data.val_mask.nonzero().squeeze()
        test_idx = dataset.data.test_mask.nonzero().squeeze()


    elif name=='ogbn-products':
        dataset = PygNodePropPredDataset(name='ogbn-products')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        x_norm_func = lambda x:x

        args.node_feature_dim=100
        args.node_feature_type='dense'
        args.num_class =47

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=10000
        # args.max_steps=400000


    elif name =='ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        x_norm_func = lambda x:x

        args.node_feature_dim=128
        args.node_feature_type='dense'
        args.num_class =40

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        # args.warmup_steps=4000
        # args.max_steps=10000


    else:
        raise ValueError('no such dataset')


    if args.sampling_algo=='shadowkhop':
        Sampling_Dataset = Graphtrans_Sampling_Dataset
    elif args.sampling_algo=='sage':
        Sampling_Dataset = Graphtrans_Sampling_Dataset_v2
        # args.num_neighbors=50

    if name in ["Cora","Computers","PubMed","Wisconsin","Actor"]:
        train_set = Sampling_Dataset(dataset,
                                            node_idx=train_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                                args=args)
        valid_set = Sampling_Dataset(dataset,
                                            node_idx=valid_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                                args=args)
        test_set = Sampling_Dataset(dataset,
                                            node_idx=test_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                            args=args)
    else:
        train_set = Sampling_Dataset(dataset.data,
                                            node_idx=train_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                                args=args)
        valid_set = Sampling_Dataset(dataset.data,
                                            node_idx=valid_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                                args=args)
        test_set = Sampling_Dataset(dataset.data,
                                            node_idx=test_idx,
                                            depth=args.depth,
                                            num_neighbors=args.num_neighbors,
                                            replace=False,
                                            x_norm_func=x_norm_func,
                                            args=args)
    return train_set,valid_set,test_set, dataset, args







#just test
if __name__=='__main__':
    pass



