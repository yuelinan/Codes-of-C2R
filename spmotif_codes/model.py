import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import torch.nn as nn
from conv import GNN_node, GNN_node_Virtualnode
import numpy as np
import copy
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
nn_act = torch.nn.ReLU()
F_act = F.relu


class SPMotifNet_GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(SPMotifNet_GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        emb_dim_rat = emb_dim
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
        else:
            rationale_gnn_node = GNN_node(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
            
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))
        self.pool = global_mean_pool
        self.node_enoder = nn.Linear(4,self.emb_dim)
    def forward(self, batched_data):
        # print(batched_data)
        # print(batched_data.size())
        
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)
        
        
        h_node = self.graph_encoder(batched_data)
        

        batch = batched_data.batch
        size = batch[-1].item() + 1 

        h_out = global_mean_pool(  h_node, batched_data.batch)
        # h_out = scatter_add( h_node, batch, dim=0, dim_size=size)
        

        # h_graph = self.pool(h_node, batched_data.batch)

        pred_rem = self.predictor(h_out)
        output = { 'pred_rem': pred_rem}
        return output
        # return output

    def eval_forward(self, batched_data):
        
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)
        h_node = self.graph_encoder(batched_data)
        batch = batched_data.batch
        size = batch[-1].item() + 1 
        h_r = global_mean_pool(  h_node, batch)
        # h_r = scatter_add( h_node, batch, dim=0, dim_size=size)
        pred_rem = self.predictor(h_r)
        return pred_rem 

    def eval_forward2(self, batched_data):
        
        
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)

        h_node = self.graph_encoder(batched_data)
        batch = batched_data.batch
        size = batch[-1].item() + 1 
        h_r = global_mean_pool(  h_node, batch)
        # h_r = scatter_add( h_node, batch, dim=0, dim_size=size)
        return h_r 



class Graph_C2R(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(Graph_WWW_Gumbel, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma
        self.infonce = InfoNCE(emb_dim)
        self.infonce2 = InfoNCE(emb_dim)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        emb_dim_rat = emb_dim
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
        else:
            rationale_gnn_node = GNN_node(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name,atom_encode=False)
        self.separator = separator_gum(
            rationale_gnn_node=rationale_gnn_node, 
            gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim_rat, 2*emb_dim_rat), torch.nn.BatchNorm1d(2*emb_dim_rat), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim_rat, 2)),
            nn=None
            )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))

        self.node_enoder = nn.Linear(4,self.emb_dim)
        self.env_mlp = torch.nn.Linear(rep_dim*2, rep_dim)
            # loss_infonce = self.infonce(h_r,combine_rationale)
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, batched_data,ori_cluster_one_hot=None,cluster_one_hot=None,cluster_centers=None):
        # print(batched_data)
        # print(batched_data.size())
        if cluster_one_hot  == None:
            batched_data.edge_attr = batched_data.edge_attr.long()
            batched_data.x = self.node_enoder(batched_data.x)
            h_node = self.graph_encoder(batched_data)

            ##  rationale
            h_r, h_env, r_node_num, env_node_num,gate = self.separator(batched_data, h_node)
            pred_rem = self.predictor(h_r)
            loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()

            ##  GNN
            batch = batched_data.batch
            size = batch[-1].item() + 1 
            h_out = global_mean_pool(  h_node, batch)
            # h_out = scatter_add( h_node, batch, dim=0, dim_size=size)
            pred_gnn = self.predictor(h_out)

            output = {'pred_gnn': pred_gnn, 'pred_rem': pred_rem, 'loss_reg':loss_reg}
            return output
        else:
            ##  GNN
            batched_data.edge_attr = batched_data.edge_attr.long()
            batched_data.x = self.node_enoder(batched_data.x)
            h_node = self.graph_encoder(batched_data)
            batch = batched_data.batch
            size = batch[-1].item() + 1 
            h_out = global_mean_pool(  h_node, batch)
            # h_out = scatter_add( h_node, batch, dim=0, dim_size=size)
            ##  cycle loss
            ##  sample env id 
            env = torch.matmul(cluster_one_hot,cluster_centers)
            ori_env = torch.matmul(ori_cluster_one_hot,cluster_centers)
            augument_g = self.env_mlp(torch.cat([h_out,env],-1))
            cycle_g = self.env_mlp(torch.cat([augument_g,ori_env],-1))
            ## pred
            pred_gnn = self.predictor(h_out)
            pred_augument_gnn = self.predictor(augument_g)
            cycle_loss = self.infonce(h_out,cycle_g)
            # cycle_loss = self.mse(h_out,cycle_g)


            ##  rationale
            h_r, h_env, r_node_num, env_node_num,gate = self.separator(batched_data, h_node)
            ## pred
            pred_rem = self.predictor(h_r)
            loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
            teacher_loss =  self.infonce2(h_out,h_r)
            # teacher_loss =  self.mse(h_out,h_r)

            output = {'pred_gnn': pred_gnn, 'pred_rem': pred_rem, 'loss_reg':loss_reg,'pred_augument_gnn':pred_augument_gnn, 'cycle_loss':cycle_loss, 'teacher_loss':teacher_loss}
            return output

    def get_kmeans_forward(self, batched_data):
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)

        h_node = self.graph_encoder(batched_data)
        h_r, h_env, _, _,gate = self.separator(batched_data, h_node)
        return h_env
       

    
    def eval_forward(self, batched_data):
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)
        
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _,gate = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem 

    def eval_explain_forward(self, batched_data):
        batched_data.edge_attr = batched_data.edge_attr.long()
        
        batched_data.x = self.node_enoder(batched_data.x)
        
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _,gate = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem ,gate

    def get_non_rationale(self,batched_data):
        batched_data.edge_attr = batched_data.edge_attr.long()
        batched_data.x = self.node_enoder(batched_data.x)
        
        h_node = self.graph_encoder(batched_data)
        h_r, h_n, _, _,gate = self.separator(batched_data, h_node)
        
        return h_n 



class separator_gum(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator_gum, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)
        
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        
        gate = F.gumbel_softmax(gate,hard=False,dim=-1)

        gate = gate[:,-1].unsqueeze(-1)

        h_out = global_mean_pool(gate * h_node, batch)
        # h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        c_out = global_mean_pool((1 - gate) * h_node, batch)
        # c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)
        
        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        
        
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)
        
        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 ,self.gate_nn(x)[:,-1].unsqueeze(-1)




class InfoNCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(InfoNCE, self).__init__()
        print('InfoNCE')
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
                                    
    def forward(self, x_samples, y_samples): 
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#
        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]
        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size))    
        return -lower_bound

