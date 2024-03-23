
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os.path as osp
import random
## dataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from mnistsp_dataset import MNIST75sp
## training
from model_mnistsp import Graph_C2R
from utils import init_weights, get_args, eval_mnist,train_student,eval_mnist_noise,train_www_epoch0,train_www,get_kmeans
import copy
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


args_first = get_args()
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
logger = logging.getLogger(__name__)

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    datadir = './data/'

    # n_train_data, n_val_data = 20000, 5000
    n_train_data, n_val_data = 5000, 1000
    train_val = MNIST75sp(osp.join(datadir, 'mnistsp/'), mode='train')
    perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
    train_val = train_val[perm_idx]
    train_dataset, val_dataset = train_val[:n_train_data], train_val[-n_val_data:]
    test_dataset = MNIST75sp(osp.join(datadir, 'mnistsp/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset[0:1000], batch_size=args.batch_size, shuffle=False)
    n_test_data = float(len(test_loader.dataset))

    color_noises = torch.load(osp.join(datadir, 'mnistsp/raw/mnist_75sp_color_noise.pt')).view(-1,3)

    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), float(len(test_loader.dataset))
    logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

    model = Graph_WWW_Gumbel(args, gnn_type = args.gnn, num_tasks = 10, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)    


    init_weights(model, args.initw_name, init_gain=0.02)


    
    opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    
    opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()) +list(model.infonce.parameters()) +list(model.infonce2.parameters()) + list(model.env_mlp.parameters()), lr=args.lr, weight_decay=args.l2reg)


    optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    loss_logger = []
    valid_logger = []
    test_logger = []
    task_type = ["classification"]
    for epoch in range(args.epochs):
        # if epoch>2:break
        # print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'

        if epoch==0:
            model.train()
            train_www_epoch0(args, model, device, train_loader, optimizers, task_type, optimizer_name,loss_logger)
        else:
            cluster_ids_x, cluster_centers,num_clusters = get_kmeans(args, model, device, train_loader)
            
            def get_random_env(num_clusters,idx):
                random_number = random.randrange(0, num_clusters, 1)
                while random_number == idx:
                    random_number = random.randrange(0, num_clusters, 1)
                return random_number

            cluster_ids = []
            xx = [cluster_ids_x[i] for i in cluster_ids_x.cpu().numpy().tolist()]
            for i in range(0, len(xx), args.batch_size):
                cluster_ids.append(xx[i:i + args.batch_size])

            second_cluster_ids = []
            for i in cluster_ids_x:
                second_cluster_ids.append(get_random_env(num_clusters,i))

            sec_cluster_ids = []
            xx = [second_cluster_ids[i] for i in cluster_ids_x.cpu().numpy().tolist()]
            for i in range(0, len(xx), args.batch_size):
                sec_cluster_ids.append(xx[i:i + args.batch_size])

            model.train()
            train_www(args, model, device, train_loader, optimizers, task_type, optimizer_name,loss_logger,cluster_ids,sec_cluster_ids,cluster_centers)



        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval_mnist(args, model, device, train_loader)[0]
        valid_perf = eval_mnist(args, model, device, valid_loader)[0]
        test_logger_perfs = eval_mnist_noise(args, model, device, test_loader,color_noises)[0]
        valid_logger.append(valid_perf)
        test_logger.append(test_logger_perfs)
        update_test = False
        if epoch != 0:
            if 'classification' in task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in task_type and valid_perf <  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval_mnist_noise(args, model, device, test_loader,color_noises)
            test_perfs2 = eval_mnist(args, model, device, test_loader)[0]
            
            test_auc  = test_perfs[0]
            logger.info("=====Epoch {}, Metric: {}, Validation: {}, Test_ood: {}, Test: {}".format(epoch, 'AUC', valid_perf, test_auc,test_perfs2))
            print("=====Epoch {}, Metric: {}, Validation: {}, Test_ood: {}, Test: {}".format(epoch, 'AUC', valid_perf, test_auc,test_perfs2))
            
        else:
            # print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    logger.info('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))



    print('Test auc: {}'.format(test_auc))
    return [best_valid_perf, test_auc]

    

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'mnistsp':
            args.epochs = 40
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin-virtual'
                args.l2reg = 1e-3
                args.lr = 1e-3
                args.gamma = 0.55
                args.num_layer = 2  
                args.batch_size = 256
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 20
                args.drop_ratio = 0.3
                args.initw_name = 'orthogonal' 
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
                args.lr = 1e-3
                args.patience = 20
                args.initw_name = 'orthogonal' 
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 256

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    args.plym_prop = 'none' 
    results = {'valid_auc': [], 'test_auc': []}
    for seed in range(args.trails):
        if args.dataset.startswith('plym'):
            valid_rmse, test_rmse, test_r2 = main(args)
            results['test_r2'].append(test_r2)
            results['test_rmse'].append(test_rmse)
            results['valid_rmse'].append(valid_rmse)
        else:
            set_seed(seed)
            valid_auc, test_auc = main(args)
            results['valid_auc'].append(valid_auc)
            results['test_auc'].append(test_auc)
    for mode, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))

if __name__ == "__main__":
    args = get_args()
    config_and_run(args)
    




