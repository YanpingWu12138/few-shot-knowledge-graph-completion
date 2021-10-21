import logging
from modules import *
from gnn import *
from utils import *
from torch.nn.parameter import Parameter
import math
from torch.nn import functional as F


class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_input=0.3, finetune=False,
                 dropout_neighbors=0.0,
                 device=torch.device("cpu")):
        super(EntityEncoder, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(self.device)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.NeighborAggregator = AttentionSelectContext(dim=embed_dim, dropout=dropout_neighbors)

    def neighbor_encoder_mean(self, connections, num_neighbors):
        """
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()

    def neighbor_encoder_soft_select(self, connections_left, connections_right, head_left, head_right):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        relations_left = connections_left[:, :, 0].squeeze(-1)
        entities_left = connections_left[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).squeeze(-1)  # [b, max]

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (batch, 200, embed_dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (batch, 200, embed_dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).squeeze(-1)  # [b, max]

        left = [head_left, rel_embeds_left, ent_embeds_left]
        right = [head_right, rel_embeds_right, ent_embeds_right]
        output = self.NeighborAggregator(left, right, mask_matrix_left, mask_matrix_right)
        return output

    def forward(self, entity, entity_meta=None):
        '''
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         '''
        if entity_meta is not None:
            entity = self.symbol_emb(entity)
            entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
            entity_left, entity_right = self.neighbor_encoder_soft_select(entity_left_connections,
                                                                          entity_right_connections,
                                                                          entity_left, entity_right)
        else:
            # no_meta
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
        return entity_left, entity_right


class RelationRepresentation(nn.Module):
    def __init__(self, device,emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(device=device, model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3,
                                                  with_pos=True)

    def forward(self, left, right):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.RelationEncoder(left, right)
        return relation


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(self.device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('device:', input.device, self.weight.device)
        num_query = input.size(0)
        weight = self.weight.expand(num_query,-1,-1)
        support = torch.bmm(input, weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def norm(self, adj, symmetric=True):
        # A = A+I
        new_adj = adj + torch.eye(adj.size(0)).to(self.device)
        # 所有节点的度
        degree = new_adj.sum(1)
        if symmetric:
            # degree = degree^-1/2
            degree = torch.diag(torch.pow(degree, -0.5))
            return degree.mm(new_adj).mm(degree)
        else:
            # degree=degree^-1
            degree = torch.diag(torch.pow(degree, -1))
            return degree.mm(new_adj)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class Matcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_layers=0.1, dropout_input=0.3,
                 dropout_neighbors=0.0,
                 finetune=False, num_transformer_layers=6, num_transformer_heads=4,
                 device=torch.device("cpu")
                 ):
        super(Matcher, self).__init__()
        self.hidden_layers = [640,320,320,160]
        self.n_feat = 400
        self.n_output_feat = 100
        self.num_features = 11
        self.dropout_p = 0.5
        self.device = device
        self.EntityEncoder = EntityEncoder(embed_dim, num_symbols,
                                           use_pretrain=use_pretrain,
                                           embed=embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors,
                                           finetune=finetune, device=self.device)
        self.RelationRepresentation = RelationRepresentation(device=self.device,
                                                             emb_dim=embed_dim,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.gc1 = GraphConvolution(self.n_feat, self.n_output_feat, self.device)
        self.gc2 = GraphConvolution(self.n_feat //2, self.n_output_feat, self.device)
        self.Prototype = SoftSelectPrototype(embed_dim * num_transformer_heads)

        self.fc_1 = nn.Sequential(nn.Linear(in_features=self.n_feat * 2, out_features=self.hidden_layers[0], bias=True),
                                  nn.BatchNorm1d(self.num_features**2),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=self.dropout_p),
                                  nn.Linear(in_features=self.hidden_layers[0], out_features=self.hidden_layers[1], bias=True),
                                  nn.BatchNorm1d(self.num_features**2),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=self.dropout_p),
                                  nn.Linear(in_features=self.hidden_layers[1], out_features=1, bias=True),
                                  nn.Softmax())

        self.fc_2 = nn.Sequential(nn.Linear(in_features=self.n_feat * 2, out_features=self.hidden_layers[2], bias=True),
                                  nn.BatchNorm1d(5),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=self.dropout_p),
                                  nn.Linear(in_features=self.hidden_layers[2], out_features=self.hidden_layers[3], bias=True),
                                  nn.BatchNorm1d(5),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=self.dropout_p),
                                  nn.Linear(in_features=self.hidden_layers[3], out_features=1, bias=True),
                                  nn.BatchNorm1d(5))

    def forward(self, support, support_negative, query, false=None, isEval=False, init_adj=None, support_meta=None, support_negative_meta=None, query_meta=None, false_meta=None):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """
        if not isEval:
            support_r = self.EntityEncoder(support, support_meta)
            support_negative_r = self.EntityEncoder(support_negative, support_negative_meta)
            query_r = self.EntityEncoder(query, query_meta)
            false_r = self.EntityEncoder(false, false_meta)


            # 利用Transformer建模每个样本为单向量
            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            support_negative_r = self.RelationRepresentation(support_negative_r[0], support_negative_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            false_r = self.RelationRepresentation(false_r[0], false_r[1])
            support_data = torch.cat((support_r,support_negative_r),0)
            query_data = torch.cat((query_r,false_r),0)

            num_support_positive = support_r.size(0)
            num_support_negative = support_negative.size(0)
            num_support = support_data.size(0)
            num_query_positive = query_r.size(0)
            num_query_negative = false_r.size(0)
            num_query = query_r.size(0) + false_r.size(0)
            num_samples = num_support + num_query

            support_data_tiled = support_data.unsqueeze(0).repeat(num_query, 1, 1)  # num_queries x num_support x featdim
            query_data_reshaped =query_data.contiguous().view(num_query, -1).unsqueeze(1)  # (num_queries) x 1 x featdim
            input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],1)  # (num_queries) x (num_support + 1) x featdim



            node_size = input_node_feat[0].size(0)
            x_i = input_node_feat.unsqueeze(2).repeat(1,1,node_size,1)
            x_j = torch.transpose(x_i,1,2)
            x_ij = torch.cat((x_i,x_j),-1)
            adj = self.fc_1(x_ij.contiguous().view(num_query,node_size **2, self.n_feat*2)).view(num_query,node_size, node_size)
            adj = adj + torch.eye(node_size).unsqueeze(0).repeat(num_query,1,1).to(self.device)
            ones = torch.ones(adj[:,:num_support,:num_support].size()).to(self.device)
            init_adj = init_adj[:, 0, :num_support, :num_support]

            result = torch.where(init_adj>0, adj[:,:num_support,:num_support], -adj[:, :num_support, :num_support])
            gcn_adj = adj.clone()
            gcn_adj[:,:num_support,:num_support] = result


            output_node_feat = self.gc1(input_node_feat, gcn_adj)


            output_support_positive_r = output_node_feat[:,:num_support_positive,:]
            output_query_positive_r = output_node_feat[:,num_support:,:].repeat(1,num_support_positive,1)
            center_positive_q = self.Prototype(output_support_positive_r, output_query_positive_r)
            score = torch.sum((output_node_feat[:,num_support:,:].squeeze()) * center_positive_q, dim=1)
            positive_score = score[:num_query_positive]
            negative_score = score[num_query_positive:]

            output_support_negative_r = output_node_feat[:,num_support_positive:num_support,:]
            output_query_negative_r = output_node_feat[:,num_support:,:].repeat(1,num_support_negative,1)
            center_negative_q = self.Prototype(output_support_negative_r, output_query_negative_r)
            n_score = torch.sum((output_node_feat[:,num_support:,:].squeeze()) * center_negative_q, dim=1)
            n_positive_score = n_score[:num_query_positive]
            n_negative_score = n_score[num_query_positive:]

            del support_r,support,query,query_r,support_negative,support_data_tiled,query_data_reshaped,input_node_feat,output_query_negative_r,output_query_positive_r,output_support_negative_r,output_support_positive_r,output_node_feat
            torch.cuda.empty_cache()

            # output_support_r = output_node_feat[:,:num_support_positive,:]
            # output_query_r = output_node_feat[:,num_support:,:].repeat(1,num_support_positive,1)
            # output_node_feat = torch.cat((output_support_r, output_query_r), 2)
            #
            # score = torch.sum(self.fc_2(output_node_feat).squeeze(),dim=1)
            # positive_score = score[:num_query_positive]
            # negative_score = score[num_query_positive:]

        else:
            support_r = self.EntityEncoder(support, support_meta)
            support_negative_r = self.EntityEncoder(support_negative, support_negative_meta)
            query_r = self.EntityEncoder(query, query_meta)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            support_negative_r = self.RelationRepresentation(support_negative_r[0], support_negative_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])

            query_data = query_r
            support_data = torch.cat((support_r, support_negative_r),0)

            num_support_positive = support_r.size(0)
            num_support = support_data.size(0)
            num_query = query_r.size(0)
            num_samples = num_support + num_query

            support_data_tiled = support_data.unsqueeze(0).repeat(num_query, 1, 1)  # num_queries x num_support x featdim
            query_data_reshaped = query_data.contiguous().view(num_query, -1).unsqueeze(1)  # (num_queries) x 1 x featdim
            input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],1)  # (num_queries) x (num_support + 1) x featdim

            node_size = input_node_feat[0].size(0)
            x_i = input_node_feat.unsqueeze(2).repeat(1, 1, node_size, 1)
            x_j = torch.transpose(x_i, 1, 2)

            del query, query_r, query_data, query_data_reshaped
            torch.cuda.empty_cache()

            #
            if num_query>500:
                chunk_num = math.ceil(num_query / 500)
                input_node_feat = list(torch.chunk(input_node_feat,chunk_num,0))
                # x_ij = list(torch.chunk(x_ij, chunk_num,0))

                output_node_feat_list = []
                adj_list = []
                curr_index = 0
                for j in range(len(input_node_feat)):
                    node_feat = input_node_feat[j]
                    num_query_x = node_feat.size(0)
                    x_ij = torch.cat((x_i[curr_index:num_query_x+curr_index],x_j[curr_index:num_query_x+curr_index]),-1)
                    adj = self.fc_1(x_ij.contiguous().view(num_query_x, node_size ** 2, self.n_feat * 2)).view(num_query_x, node_size, node_size)
                    adj = adj + torch.eye(node_size).unsqueeze(0).repeat(num_query_x,1,1).to(self.device)
                    ones = torch.ones(adj[:,:num_support,:num_support].size()).to(self.device)
                    init_adj_x = init_adj[curr_index:num_query_x+curr_index, 0, :num_support, :num_support]
                    adj_temp = torch.where(init_adj_x>0, adj[:,:num_support,:num_support], -adj[:,:num_support,:num_support])
                    gcn_adj = adj.clone()
                    gcn_adj[:, :num_support, :num_support] = adj_temp
                    output_node_feat = self.gc1(node_feat, gcn_adj)
                    output_node_feat_list.append(output_node_feat)
                    adj_list.append(adj)
                    curr_index += num_query_x

                output_node_feat = torch.cat(output_node_feat_list,0)
                adj = torch.cat(adj_list,0)
            else:
                x_ij = torch.cat((x_i,x_j),-1)
                adj = self.fc_1(x_ij.contiguous().view(num_query, node_size ** 2, self.n_feat * 2)).view(num_query, node_size, node_size)
                adj = adj + torch.eye(node_size).unsqueeze(0).repeat(num_query,1,1).to(self.device)
                ones = torch.ones(adj[:, :num_support, :num_support].size()).to(self.device)
                init_adj = init_adj[:, 0, :num_support, :num_support]

                adj_temp = torch.where(init_adj > 0, adj[:,:num_support,:num_support], -adj[:, :num_support, :num_support])
                gcn_adj = adj.clone()
                gcn_adj[:, :num_support, :num_support] = adj_temp

                output_node_feat = self.gc1(input_node_feat, gcn_adj)



            output_support_r = output_node_feat[:, :num_support_positive, :]
            output_query_r = output_node_feat[:, num_support:, :].repeat(1, num_support_positive, 1)

            # todo 加入GNN模块，通过样本之间的相似度，传播样本的特征，建模样本之间的关系
            center_q = self.Prototype(output_support_r, output_query_r)
            score = torch.sum((output_node_feat[:, num_support:, :].squeeze()) * center_q, dim=1)
            positive_score = score
            negative_score = None

            # output_support_r = output_node_feat[:, :num_support_positive, :]
            # output_query_r = output_node_feat[:, num_support:, :].repeat(1, num_support_positive, 1)
            # output_node_feat = torch.cat((output_support_r, output_query_r), 2)
            #
            # score = torch.sum(self.fc_2(output_node_feat).squeeze(), dim=1)
            # positive_score = score
            # negative_score = None

            n_positive_score = None
            n_negative_score = None

        return positive_score, negative_score, adj, n_positive_score, n_negative_score
