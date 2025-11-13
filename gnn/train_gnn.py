import sys
import os
import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from model_pyg import GCN, LinkPred, GAT, get_link_feat, SmallMLP
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, auc, precision_recall_curve
import torch.nn.functional as F
from collections import defaultdict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", type=str, default='/data/AResKG_species_model/')
ap.add_argument("--fold", type=int, default=0, help='fold number')
ap.add_argument("--nega_num", type=int, default=3, help='negative sample number')
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--method", type=str, default='GCN', help='GCN, GAT')
ap.add_argument("--strain", type=str, default='Klebsiella')
ap.add_argument("--in_channels", type=int, default=256, help='the dimension of word embedding')
ap.add_argument("--hidden_channels", type=int, default=256, help='the dimension of hidden layer of GNN')
ap.add_argument("--out_channels", type=int, default=256, help='the dimension of input of classifier')

args = ap.parse_args()

prefix = f'{args.prefix}/kg_constructor_llm_1021_{args.strain}/output/folds'
path = f'./kg_{args.strain}'
if not os.path.exists(path):
    os.makedirs(path)

k = args.fold
in_channels = args.in_channels
hidden_channels = args.hidden_channels
out_channels = args.out_channels
dropout = 0.0
global total_nodes

method = args.method
link_aggr = 'cat'
nega_num = args.nega_num
lr = args.lr

if method == 'GAT':
    head = 8

def convert2str(x):
    try:
        if '_' in str(x): 
            pass
        else:
            x = float(x)
    except:
        pass

    return str(x)


def format_df(dff, unique_node_types):
    dff['x_type'] = dff['Predicate'].map(x_map)
    dff['y_type'] = dff['Predicate'].map(y_map)
    dff['x_idx'] = np.nan
    dff['y_idx'] = np.nan
    dff['Subject'] = dff.Subject.apply(lambda x: convert2str(x))
    dff['Object'] = dff.Object.apply(lambda x: convert2str(x))

    print('Iterating over node types...')
    # print(f'unique_node_types: {unique_node_types}')
    for i in tqdm(unique_node_types):
        names2idx = idx_map[i]

        names_ = np.unique(np.append(dff[dff.x_type == i]['Subject'].values, dff[dff.y_type == i]['Object'].values))
        for n in names_:
            if n not in names2idx:
                names2idx[n] = len(names2idx)
        dff.loc[dff.x_type == i, 'x_idx'] = dff[dff.x_type == i]['Subject'].apply(lambda x: names2idx[x])
        dff.loc[dff.y_type == i, 'y_idx'] = dff[dff.y_type == i]['Object'].apply(lambda x: names2idx[x])
    return dff


def reverse_rel_generation(df, dff, unique_rel):

    temp_list = []
    for i in unique_rel.values:
        temp = dff[dff.Predicate == i[1]]
        temp = temp.rename(columns={"x_type": "y_type", 
                     "x_id": "y_id", 
                     "x_idx": "y_idx",
                     "y_type": "x_type", 
                     "y_id": "x_id", 
                     "y_idx": "x_idx"})

        if i[0] != i[2]:
            # bi identity
            temp["Predicate"] = 'rev_' + i[1]
        temp_list.append(temp)
    return pd.concat([dff] + temp_list, ignore_index=True)

def create_homograph(df_train: pd.DataFrame, idx_map: dict):
    # 1) Node-type to contiguous global ID ranges
    ntype2len = {nt: len(idx_map[nt]) for nt in idx_map}
    ntypes_sorted = list(ntype2len.keys())  # keep natural order
    ntype2id = {nt: i for i, nt in enumerate(ntypes_sorted)}

    # offsets: cumulative
    ntype2offset = {}
    running = 0
    for nt in ntypes_sorted:
        ntype2offset[nt] = running
        running += ntype2len[nt]
    total_nodes = running

    # 2) Build global src/dst (vectorized)
    # ensure correct dtype
    df_local = df_train[['x_type', 'y_type', 'x_idx', 'y_idx', 'Predicate']].copy()

    # x_idx / y_idx might be float from earlier NaN init; ensure int64
    df_local['x_idx'] = df_local['x_idx'].astype(np.int64)
    df_local['y_idx'] = df_local['y_idx'].astype(np.int64)

    # map type -> offset
    df_local['x_off'] = df_local['x_type'].map(ntype2offset)
    df_local['y_off'] = df_local['y_type'].map(ntype2offset)

    # global ids
    src = torch.from_numpy((df_local['x_off'].to_numpy() + df_local['x_idx'].to_numpy()).astype(np.int64))
    dst = torch.from_numpy((df_local['y_off'].to_numpy() + df_local['y_idx'].to_numpy()).astype(np.int64))

    # 3) Build homogeneous graph
    graph_edges = torch.stack([src, dst], dim=0)

    # 4) Edge type (predicate) as integer feature
    etypes = df_local['Predicate'].astype(str).unique().tolist()
    etype2id = {e: i for i, e in enumerate(etypes)}
    graph_etype = torch.from_numpy(df_local['Predicate'].map(etype2id).to_numpy().astype(np.int64))

    # 5) Node type (original ntype) as integer feature (length = total_nodes)
    ntype_id = torch.empty(total_nodes, dtype=torch.int64)
    for nt in ntypes_sorted:
        off = ntype2offset[nt]
        ln = ntype2len[nt]
        ntype_id[off:off+ln] = ntype2id[nt]
    graph_ntype = ntype_id

    return graph_edges, graph_etype, graph_ntype


def create_pos_neg_data(dff, idx_map):
    ntype2len = {nt: len(idx_map[nt]) for nt in idx_map}
    ntypes_sorted = list(ntype2len.keys())  # keep natural order
    ntype2id = {nt: i for i, nt in enumerate(ntypes_sorted)}

    # offsets: cumulative
    ntype2offset = {}
    running = 0
    for nt in ntypes_sorted:
        ntype2offset[nt] = running
        running += ntype2len[nt]
    total_nodes = running
    df_pos = dff[dff['Label'] == 1][['x_type','y_type','x_idx','y_idx','Predicate']]
    df_neg = dff[dff['Label'] == -1][['x_type','y_type','x_idx','y_idx','Predicate']]
    def transform_id(df_edges):
        x_off = df_edges['x_type'].map(ntype2offset).to_numpy()
        y_off = df_edges['y_type'].map(ntype2offset).to_numpy()
        src = torch.from_numpy((df_edges['x_idx'].to_numpy().astype(np.int64) + x_off.astype(np.int64)))
        dst = torch.from_numpy((df_edges['y_idx'].to_numpy().astype(np.int64) + y_off.astype(np.int64)))
        return torch.stack([src, dst], dim=0)
    pos_edge = transform_id(df_pos)
    neg_edge = transform_id(df_neg)
    return pos_edge, neg_edge

def create_pos_neg_data_from_kg(dff, idx_map):
    ntype2len = {nt: len(idx_map[nt]) for nt in idx_map}
    ntypes_sorted = list(ntype2len.keys())  # keep natural order
    ntype2id = {nt: i for i, nt in enumerate(ntypes_sorted)}

    # offsets: cumulative
    ntype2offset = {}
    running = 0
    for nt in ntypes_sorted:
        ntype2offset[nt] = running
        running += ntype2len[nt]
    total_nodes = running
    df_resist = dff[dff['Predicate'] == 'confers#SPACE#resistance#SPACE#to'][['x_type','y_type','x_idx','y_idx','Predicate']]
    def transform_id(df_edges):
        x_off = df_edges['x_type'].map(ntype2offset).to_numpy()
        y_off = df_edges['y_type'].map(ntype2offset).to_numpy()
        src = torch.from_numpy((df_edges['x_idx'].to_numpy().astype(np.int64) + x_off.astype(np.int64)))
        dst = torch.from_numpy((df_edges['y_idx'].to_numpy().astype(np.int64) + y_off.astype(np.int64)))
        return torch.stack([src, dst], dim=0)
    pos_edge = transform_id(df_resist)
    df_other = dff[~dff['Predicate'].isin(['rev_confers#SPACE#resistance#SPACE#to', 'confers#SPACE#resistance#SPACE#to'])][['x_type','y_type','x_idx','y_idx','Predicate']]

    pos_other_edge = transform_id(df_other)
    return pos_edge, ntype2offset['Genes'], df_resist['x_idx'], df_resist['y_idx'], pos_other_edge


if os.path.exists(os.path.join(path, f'fold_{k}_train_local.csv')):
    print('Splits detected... Loading splits....')
    df_train = pd.read_csv(os.path.join(path, f'fold_{k}_train.csv'))
    df_valid = pd.read_csv(os.path.join(path, f'fold_{k}_valid.csv'))
    df_test = pd.read_csv(os.path.join(path, f'fold_{k}_test.csv'))
    df_train_local = pd.read_csv(os.path.join(path, f'fold_{k}_train_local.csv'))
    with open(os.path.join(path, f'fold_{k}_idx_map.pkl'), 'rb') as f: 
        idx_map = pkl.load(f)
    print('Splits loaded...')

    with open(os.path.join(path, f'entities_fold_{k}.txt'), 'r') as f:
        lines = f.readlines()
        entites = [line.strip() for line in lines]

else:
    df = pd.read_csv(f'{prefix}/fold_{k}/train.txt', 
                    names=['Subject', 'Predicate', 'Object', 'Label'], sep='\t')
    assert (df['Label'] == 0).sum() == 0
    unique_relation = np.unique(df.Predicate.values)
    print(f'unique_relation: {unique_relation}')
    file_path = f'{prefix}/fold_{k}/domain_range.txt'
    predicate_to_type = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            predicate = parts[0]
            x_type = parts[1]
            y_type = parts[2]
            predicate_to_type[predicate] = (x_type, y_type)
    x_map = {k: v[0] for k, v in predicate_to_type.items()}
    y_map = {k: v[1] for k, v in predicate_to_type.items()}
    df['x_type'] = df['Predicate'].map(x_map)
    df['y_type'] = df['Predicate'].map(y_map)
    print(f'df with x_type\n: {df.head()}')
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))
    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['Subject'] = df.Subject.apply(lambda x: convert2str(x))
    df['Object'] = df.Object.apply(lambda x: convert2str(x))
    idx_map = {}
    print('Iterating over node types...')
    for i in tqdm(unique_node_types):
        names = np.unique(np.append(df[df.x_type == i]['Subject'].values, df[df.y_type == i]['Object'].values))
        names2idx = dict(zip(names, list(range(len(names)))))
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['Subject'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['Object'].apply(lambda x: names2idx[x])
        idx_map[i] = names2idx
        
            
    print('save kg_directed.csv...')
    df.to_csv(os.path.join(path, f'fold_{k}_kg_directed.csv'), index = False)
    df_train_local =  pd.read_csv(f'{prefix}/fold_{k}/train_local.txt', 
                    names=['Subject', 'Predicate', 'Object', 'Label'], sep='\t')
    df_train_local = format_df(df_train_local, unique_node_types)
    df_valid = pd.read_csv(f'{prefix}/fold_{k}/dev.txt', 
                    names=['Subject', 'Predicate', 'Object', 'Label'], sep='\t')
    df_valid = format_df(df_valid, unique_node_types)
    df_test = pd.read_csv(f'{prefix}/fold_{k}/test.txt', 
                    names=['Subject', 'Predicate', 'Object', 'Label'], sep='\t')
    df_test = format_df(df_test, unique_node_types)
    
    if not os.path.exists(os.path.join(path, f'fold_{k}_train.csv')):
        if not os.path.exists(path):
            os.mkdir(path)           
        print('Transform the directed kg to undirected kg...')
        unique_rel = df[['x_type', 'Predicate', 'y_type']].drop_duplicates()
        df_train = reverse_rel_generation(df, df, unique_rel)
        df_train.to_csv(os.path.join(path, f'fold_{k}_train.csv'), index = False)
        df_train_local.to_csv(os.path.join(path, f'fold_{k}_train_local.csv'), index = False)
        df_valid.to_csv(os.path.join(path, f'fold_{k}_valid.csv'), index = False)
        df_test.to_csv(os.path.join(path, f'fold_{k}_test.csv'), index = False)
        
    with open(os.path.join(path, f'fold_{k}_idx_map.pkl'), 'wb') as f:
        pkl.dump(idx_map, f)

    entites = []
    for i in idx_map:
        for e in idx_map[i]:
            entites.append(e)
    with open(os.path.join(path, f'entities_fold_{k}.txt'), 'w', encoding='utf-8') as f:
        for e in entites:
            f.write(e + '\n')

total_nodes = 0
for i in idx_map:
    total_nodes += len(idx_map[i])
print(f'total_nodes: {total_nodes}')
print(f'entities: {len(entites)}')
graph_edges, graph_etype, graph_ntype = create_homograph(df_train, idx_map)
train_pos, gene_start, mat_x, mat_y, train_other_edges = create_pos_neg_data_from_kg(df_train, idx_map)
val_pos, val_neg = create_pos_neg_data(df_valid, idx_map)
test_pos, test_neg = create_pos_neg_data(df_test, idx_map)
print(f'graph_edges: {graph_edges.max()}')


from word_embedding import machine_translate_using_word, _get_padded_indices_and_weights, lookup_embeddings

indexed_entities, num_entity_words, entity_dic = machine_translate_using_word(
    os.path.join(path, f'entities_fold_{k}.txt'))
print(f'indexed_entities: {len(indexed_entities)}')
weights, padded_entity_indices = _get_padded_indices_and_weights(indexed_entities)

weights, padded_entity_indices = torch.tensor(weights, dtype=torch.float32), torch.tensor(padded_entity_indices, dtype=torch.int64)

word_weights = nn.Embedding(num_entity_words, in_channels)

classifier_in = out_channels * 2
linkpred = LinkPred(in_channels=classifier_in, hidden_channels=hidden_channels).to(device='cuda')
smlp = SmallMLP(classifier_in, hidden_channels).to(device='cuda')
if method == 'GCN':
    gnn = GCN(in_channels, hidden_channels, out_channels).to(device='cuda')
if method == 'GAT':
    gnn = GAT(in_channels, hidden_channels, out_channels, head, dropout).to(device='cuda')

optimizer = torch.optim.Adam(list(word_weights.parameters()) + 
                            list(gnn.parameters()) + 
                            list(linkpred.parameters()) + 
                            list(smlp.parameters()), 
                            lr=lr)

bce_loss = torch.nn.BCELoss()

best_val_auprc = 0
best_test_acc = 0
best_test_auc = 0
best_test_auprc = 0
best_test_recall = 0
best_test_precision = 0
best_test_f1 = 0
best_test_ap = 0
best_epoch = 0

graph_edges = graph_edges.to(device='cuda')
train_pos = train_pos.to(device='cuda')
val_pos = val_pos.to(device='cuda')
val_neg = val_neg.to(device='cuda')
test_pos = test_pos.to(device='cuda')
test_neg = test_neg.to(device='cuda')

train_losses = []
train_has_kg_losses = []
train_no_kg_losses = []
val_metric = defaultdict(list)
val_has_kg_metric = defaultdict(list)
val_no_kg_metric = defaultdict(list)
test_metric = defaultdict(list)
test_has_kg_metric = defaultdict(list)
test_no_kg_metric = defaultdict(list)

gene_num = len(idx_map['Genes'])
known_intert_mat = torch.ones((gene_num, len(idx_map['Antibiotic'])))
known_intert_mat[mat_x, mat_y] = 0.0

for e in tqdm(range(1000)):
    gnn.train()
    linkpred.train()
    optimizer.zero_grad()
    wfeat = lookup_embeddings(padded_entity_indices, word_weights.weight, weights).to(device='cuda')
    x = wfeat
    x = gnn(x, graph_edges)

    link_feat = get_link_feat(link_aggr, x[train_pos[0]], x[train_pos[1]])
    pos_out = linkpred(link_feat).squeeze()
    pos_num = len(train_pos[0])

    neg_x = torch.randint(0, gene_num, (pos_num, nega_num)).reshape(-1)
    neg_y = torch.cat([train_pos[1]] * nega_num, dim=0).to(neg_x.device)
    neg_idx = known_intert_mat[neg_x, neg_y].nonzero().reshape(-1)
    neg_x, neg_y = neg_x[neg_idx], neg_y[neg_idx]
    neg_x = neg_x + gene_start

    link_feat = get_link_feat(link_aggr, x[neg_x], x[neg_y])
    neg_out = linkpred(link_feat).squeeze()
    out = torch.cat([pos_out, neg_out])
    label = torch.cat([torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])]).to(device='cuda')
    loss = bce_loss(out, label)

    link_feat = get_link_feat(link_aggr, x[train_other_edges[0]], x[train_other_edges[1]])
    pos_out = smlp(link_feat).squeeze()
    pos_loss = bce_loss(pos_out, torch.ones(pos_out.shape[0]).to(device='cuda'))

    loss += pos_loss
    
    loss.backward()
    optimizer.step()

    if e % 10 == 0:
        gnn.eval()
        linkpred.eval()
        with torch.no_grad():
            wfeat = lookup_embeddings(padded_entity_indices, word_weights.weight, weights).to(device='cuda')

            x = wfeat
            x = gnn(x, graph_edges)
            link_feat = get_link_feat(link_aggr, x[val_pos[0]], x[val_pos[1]])
            pos_out = linkpred(link_feat).squeeze()
            link_feat = get_link_feat(link_aggr, x[val_neg[0]], x[val_neg[1]])
            neg_out = linkpred(link_feat).squeeze()
            out = torch.cat([pos_out, neg_out])
            label = torch.cat([torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])]).to(device='cuda')
            val_loss = bce_loss(out, label)
            label = label.cpu().numpy()
            out = out.cpu().numpy()
            auroc = roc_auc_score(label, out)
            ap = average_precision_score(label, out)
            f1 = f1_score(label, out.round())
            recall = recall_score(label, out.round())
            precision = precision_score(label, out.round())
            prc, rec, _ = precision_recall_curve(label, out)
            pr_auc = auc(rec, prc)

            if pr_auc.item() > best_val_auprc:
                best_val_auprc = pr_auc.item()
                link_feat = get_link_feat(link_aggr, x[test_pos[0]], x[test_pos[1]])
                pos_out = linkpred(link_feat).squeeze()
                link_feat = get_link_feat(link_aggr, x[test_neg[0]], x[test_neg[1]])
                neg_out = linkpred(link_feat).squeeze()
                out = torch.cat([pos_out, neg_out])
                label = torch.cat([torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])]).to(device='cuda')
                test_loss = bce_loss(out, label)
                label = label.cpu().numpy()
                out = out.cpu().numpy()
                auroc = roc_auc_score(label, out)
                ap = average_precision_score(label, out)
                f1 = f1_score(label, out.round())
                recall = recall_score(label, out.round())
                precision = precision_score(label, out.round())
                prc, rec, _ = precision_recall_curve(label, out)
                pr_auc = auc(rec, prc)
                
                best_test_auprc = pr_auc.item()
                best_test_auc = auroc.item()
                best_test_recall = recall.item()
                best_test_precision = precision.item()
                best_test_f1 = f1.item()
                best_test_ap = ap.item()
                best_epoch = e


respath = f'./res_{args.strain}'
if not os.path.exists(respath):
    os.makedirs(respath)
with open(f'{respath}/{method}.txt', 'a') as f:
    f.write(f'auc: {best_test_auc}\n')
    f.write(f'ap: {best_test_ap}\n')
    f.write(f'f1: {best_test_f1}\n')
    f.write(f'recall: {best_test_recall}\n')
    f.write(f'precision: {best_test_precision}\n')
    f.write(f'pr_auc: {best_test_auprc}\n')
    f.write(f'\n')
