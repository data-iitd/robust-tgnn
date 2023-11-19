
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn_ance import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
import random
# torch.manual_seed(0)
# np.random.seed(0)

#data_path = '/home/shubh/TGG/TG/data/'
import os
data_path = os.path.dirname(os.path.abspath(__file__))
data_path = data_path+"/data/"
print(data_path)
import configs.config1 as args

parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--er', type=int, default=1, help='embedding refresh interval, batch_id% for embedding refresh')
parser.add_argument('--n_epoch', type=int, default=50, help='Num of epochs to train the model')
parser.add_argument('--warmup_epoch', type=int, default=20, help='Num of warm up epochs')
parser.add_argument('--num_random_samples', type=int, default=1, help='No. of uniform random negatives')
parser.add_argument('--num_hard_samples', type=int, default=1, help='Num of hard negatives')
parser.add_argument('--save_result_file', type=str,default='', help='File name of result to be saved')
parser.add_argument('--use_memory', type=int,default=1, help='If memory to be used, if memory is not used then its a TGAT')
parser.add_argument('--topk', type=int,default=5, help='Topk nearest nodes to be sampled')
parser.add_argument('--data', type=str,default='wikipedia', help='Dataset name')



new_args = parser.parse_args()
NUM_EPOCH = new_args.n_epoch
WARMUP_EPOCH = new_args.warmup_epoch
topk_sample = new_args.topk
NUM_RANDOM_SAMPLES = new_args.num_random_samples
NUM_HARD_SAMPLES = new_args.num_hard_samples
total_negative_samples= NUM_RANDOM_SAMPLES+ NUM_HARD_SAMPLES
save_result_file = new_args.save_result_file
USE_MEMORY = bool(new_args.use_memory)
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
DATA = new_args.data


NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = new_args.gpu
DYNAMIC_EMBEDDING_REFRESH = new_args.er
#GPU = 3
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
if DATA in ["mooc","uci"]:
  #args.randomize_features = True
  MEMORY_DIM = 100
if DATA in ["ucip"]:
  MEMORY_DIM = 1900
if DATA  in  ["twitter","twitter_bp","twitter_1000_5","twitter_2000_10"]:
  MEMORY_DIM = 768

args.prefix = 'ANCE_ER_{}'.format(str(DYNAMIC_EMBEDDING_REFRESH))
print("BATCH_SIZE ", BATCH_SIZE)
print("NUM_NEIGHBORS ", NUM_NEIGHBORS)
print("NUM_EPOCH ", NUM_EPOCH)
print("NUM_HEADS ", NUM_HEADS)
print("DROP_OUT ", DROP_OUT)
print("GPU ", GPU)
print("DATA ", DATA)
print("NUM_LAYER ", NUM_LAYER)
print("LEARNING_RATE ", LEARNING_RATE)
print("NODE_DIM ", NODE_DIM)
print("TIME_DIM ", TIME_DIM)
print("USE_MEMORY ", USE_MEMORY)
print("MESSAGE_DIM ", MESSAGE_DIM)
print("MEMORY_DIM ", MEMORY_DIM)
print("Randomize features", args.randomize_features)
print("Different new nodes between val and test", args.different_new_nodes)
print("ER Refresh,", DYNAMIC_EMBEDDING_REFRESH)
print("log file,", args.prefix)
print("Num of random negative samples, ",NUM_RANDOM_SAMPLES)
print("Num of hard negative samples, ",NUM_HARD_SAMPLES)
print("Save result file,", save_result_file)
print("Top K nodes,", topk_sample)
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}_{}.log'.format(args.prefix,str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


### Extract data for training, validation and testing

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(data_path,DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)




# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
print(device)


# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
print(mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)
print(num_instance,num_batch)

i = 0
results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
Path("results/").mkdir(parents=True, exist_ok=True)

# Initialize Model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
        edge_features=edge_features, device=device,
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=NUM_NEIGHBORS,
        mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep,bipartite=True)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)


logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)

new_nodes_val_aps = []
val_aps = []
epoch_times = []
total_epoch_times = []
train_losses = []

early_stopper = EarlyStopMonitor(max_round=args.patience)

print("Memory use, ", USE_MEMORY, tgn.use_memory)



def calculate_rank(lst,item):
    for rk, row in enumerate(lst):
        if row[0] == item:
            return rk+1
    return 0 ### in case item is not in the list
def eval_mrr_link_prediction(data,model,dst_nodes):
    ranks = []
    total = len(data.sources)
    if model.use_memory:
        if model.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
            memory, last_update = model.get_updated_memory(list(range(model.n_nodes)),
                                                  model.memory.messages)
        else:
            memory = model.memory.get_memory(list(range(model.n_nodes)))
            last_update = model.memory.last_update
    else:
      memory= None
    with torch.no_grad():
        model = model.eval()
        ct = 0
        for src,dst,ts in zip(data.sources, data.destinations, data.timestamps):
            #print(src,dst,ts)
            #print("\r%d/%d"%(ct,total),end="")
            ct += 1
            nodes = np.concatenate((np.array([src]),dst_nodes))
            times = np.zeros(len(nodes))
            times.fill(ts)
            time_diffs = None
            if model.use_memory:
              time_diffs = torch.LongTensor(times).to(tgn.device) - last_update[nodes].long()
              time_diffs = (time_diffs - tgn.mean_time_shift_src) / tgn.std_time_shift_src
            node_embedding = model.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=times,
                                                                 n_layers=model.n_layers,
                                                                 n_neighbors=NUM_NEIGHBORS,
                                                                 time_diffs=time_diffs,
                                                                     
                                                                     )
            src_embed = node_embedding[0]
            target_embed = node_embedding[1:]
            src_embed = src_embed.repeat(target_embed.shape[0],1)
            #print(src_embed.shape,target_embed.shape)
            scores = model.affinity_score(src_embed,target_embed).squeeze(dim=0).squeeze(dim=1).sigmoid().cpu().numpy()
            scores = list(zip(dst_nodes,scores))
            scores.sort(key=lambda val: val[1],reverse=True)
            #print(len(scores))
            ranks.append(calculate_rank(scores,dst))

            
    mrr = np.mean([1.0 / r for r in ranks])
    rec1 = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
    rec5 = sum(np.array(ranks) <= 5)*1.0 / len(ranks)
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    return (mrr,rec1,rec5,rec10)
  
  
static_time_val = max(train_data.timestamps)
static_time_test= max(val_data.timestamps)
print(static_time_val,static_time_test)
import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction_ance(model, negative_edge_sampler, data, n_neighbors, batch_size=200,static_time=None):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    #print("Num of test batch",num_test_batch)
    #print("static embedding time",static_time)
    for k in range(num_test_batch):
        
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        sources_batch = data.sources[s_idx:e_idx]
        destinations_batch = data.destinations[s_idx:e_idx]
        timestamps_batch = data.timestamps[s_idx:e_idx]
        edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

        size = len(sources_batch)
        _, negative_samples = negative_edge_sampler.sample(size)
        # static_timestamps = np.zeros(timestamps_batch.shape[0])
        # static_timestamps.fill(static_time)
        #print(k,s_idx,e_idx,len(sources_batch), len(destinations_batch),len(negative_samples),len( timestamps_batch), len(edge_idxs_batch),n_neighbors, static_timestamps.shape)
        #print(negative_samples)
        pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors,static_timestamps=None)

        pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
        true_label = np.concatenate([np.ones(size), np.zeros(size)])

        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)
  

def compute_item_embeddings(nodes,time,model):
    with torch.no_grad():
        model = model.eval()
        if USE_MEMORY:
          memory = model.memory.get_memory(list(range(tgn.n_nodes)))
        else:
          memory = None
        times = np.zeros(nodes.shape[0])
        times.fill(time)
        node_embeddings = model.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=times,
                                                                 n_layers=model.n_layers,
                                                                 n_neighbors=NUM_NEIGHBORS,
                                                                 time_diffs=None)
    return node_embeddings

def compute_nearest_nbrs(node_embedding,dst_nodes,dst_embeddings,model,topk=1):
    src_embed = node_embedding
    target_embed = dst_embeddings
    src_embed = src_embed.repeat(target_embed.shape[0],1)
    #print(src_embed.shape,target_embed.shape)
    with torch.no_grad():
        model = model.eval()
        scores = model.affinity_score(src_embed,target_embed).squeeze(dim=0).squeeze(dim=1).sigmoid().cpu().numpy()
        scores = list(zip(dst_nodes,scores))
        scores.sort(key=lambda val: val[1],reverse=True)
    return np.array([item[0] for item in scores[:topk]])

#compute_nearest_nbrs(tpp[10],train_dsts,tpp,tgn,topk=4)

def ance_negative_sample(src_idx,dst_idx,times,dsts,dst_embeddings,model,samples=1,topk=5,choose="uniform"):
    with torch.no_grad():
        model = model.eval()
        if USE_MEMORY:
          memory = model.memory.get_memory(list(range(tgn.n_nodes)))
          last_update = model.memory.get_last_update(list(range(tgn.n_nodes)))
          #times = np.zeros(src_idx.shape[0])
          #times.fill(time)
          time_diffs = torch.LongTensor(times).to(tgn.device) - last_update[src_idx].long()
          time_diffs = (time_diffs - tgn.mean_time_shift_src) / tgn.std_time_shift_src
        else:
          memory = None
          time_diffs = None
        node_embeddings = model.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=src_idx,
                                                                 timestamps=times,
                                                                 n_layers=model.n_layers,
                                                                 n_neighbors=NUM_NEIGHBORS,
                                                                 time_diffs=time_diffs)
        
    
    negatives=np.zeros((samples,src_idx.shape[0]),dtype=int)
    i = 0
    for src,dst in zip(src_idx,dst_idx):
        nearest_nbrs = list(compute_nearest_nbrs(node_embeddings[i],dsts,dst_embeddings,model,topk))
        if dst in nearest_nbrs:
            nearest_nbrs.remove(dst)
        # if src in nearest_nbrs:
        #     nearest_nbrs.remove(src)
        if choose == 'uniform': ### pick from top-K neighbours
            negatives[:,i] = random.sample(nearest_nbrs,k=samples)
        else:   ### pick top neighbours
            negatives[:,i] = nearest_nbrs[0:samples]
        i+= 1
    return negatives


### Correct this for  times which should be actual instead of first time stamp


  
train_dsts = np.unique(train_data.destinations)
print(len(train_dsts))
train_srcs = np.unique(train_data.sources)
print(len(train_srcs))
dst_nodes = np.unique(full_data.destinations)
print(len(dst_nodes))


static_tp = []

for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training
    
    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    #tgn.static_embedding_module.neighbor_finder = train_ngh_finder
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      # 1 batch starting
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue
 
            
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        # time_to_consider = timestamps_batch[0]
        # static_time = find_the_static_index_time_cut(time_to_consider,timestamps_milestones)
        # static_tp.append(static_time)
        # static_timestamps = np.zeros(timestamps_batch.shape[0])
        # static_timestamps.fill(static_time)
        size = len(sources_batch)
        list_negatives_batch = []
        if epoch < WARMUP_EPOCH:
            for tppp in range(total_negative_samples):
                _, negatives_batch = train_rand_sampler.sample(size)
                list_negatives_batch.append(negatives_batch)
        else:
            if epoch ==WARMUP_EPOCH and batch_idx == 0:
                logger.info('switching to hard negatives')
            if batch_idx%DYNAMIC_EMBEDDING_REFRESH == 0:   ### for each epoch, batch_idx will start from 0
                time_ance = timestamps_batch[0]
                train_dst_embeddings = compute_item_embeddings(train_dsts,time_ance,tgn)
            random_negatives = []
            for tppp in range(NUM_RANDOM_SAMPLES):
                _, negatives_batch = train_rand_sampler.sample(size)
                random_negatives.append(negatives_batch)
            hard_negatives_batch = ance_negative_sample(sources_batch,destinations_batch,timestamps_batch,train_dsts,train_dst_embeddings,tgn,samples=NUM_HARD_SAMPLES,topk=topk_sample,choose='uniform')
            list_negatives_batch = random_negatives+list(hard_negatives_batch)
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
        tgn = tgn.train()

        pos_prob, neg_prob = tgn.compute_edge_probabilities_final(sources_batch, destinations_batch, list_negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS,static_timestamps=None)
        loss += criterion(pos_prob.squeeze(), pos_label) + total_negative_samples*criterion(neg_prob.squeeze(), neg_label.tile((total_negative_samples,)))# +criterion(neg_prob_2.squeeze(), neg_label_2)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      #detach memeory after every epoch
      if USE_MEMORY:
        tgn.memory.detach_memory()
    ### 1 epoch finished
    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)
    #print()
    
    
    tgn.set_neighbor_finder(full_ngh_finder)
    #tgn.static_embedding_module.neighbor_finder = full_ngh_finder
    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
        train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction_ance(model=tgn,negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS,
                                                           )
    logger.info('{} {} {}'.format(np.mean(m_loss),val_ap,val_auc))
    
    mod_val_window = 1000
    if epoch%mod_val_window==0:
        mrr,rec1,rec5,rec10 = eval_mrr_link_prediction(val_data,tgn,dst_nodes)
        logger.info('MRR {} rec1 {} rec5 {} rec10 {}'.format(mrr,rec1,rec5,rec10))
        # if early_stopper.early_stop_check(mrr):
        #     logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        #     logger.info(f'Loading the best model at epoch {early_stopper.best_epoch*mod_val_window}')
        #     best_model_path = get_checkpoint_path(early_stopper.best_epoch*mod_val_window)
        #     tgn.load_state_dict(torch.load(best_model_path))
        #     logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        #     tgn.eval()
        #     break
        # else:
        #     torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
        
if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()
  # Restore memory we had at the end of training to be used when validating on new nodes.
  # Also backup memory after validation so it can be used for testing (since test edges are
  # strictly later in time than validation edges)
    tgn.memory.restore_memory(train_memory_backup)

    
nn_val_ap, nn_val_auc = eval_edge_prediction_ance(model=tgn,
                                    negative_edge_sampler=val_rand_sampler,
                                    data=new_node_val_data,
                                    n_neighbors=NUM_NEIGHBORS)

if USE_MEMORY:
  # Restore memory we had at the end of validation
  tgn.memory.restore_memory(val_memory_backup)
print(nn_val_ap, nn_val_auc )

# nodes
if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()
    
tgn.embedding_module.neighbor_finder = full_ngh_finder
test_ap, test_auc = eval_edge_prediction_ance(model=tgn,
                     negative_edge_sampler=test_rand_sampler,
                    data=test_data,
                    n_neighbors=NUM_NEIGHBORS)

if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)
    
print(test_ap, test_auc )

nn_test_ap, nn_test_auc = eval_edge_prediction_ance(model=tgn,
                                    negative_edge_sampler=nn_test_rand_sampler,
                                    data=new_node_test_data,
                                    n_neighbors=NUM_NEIGHBORS)

    
    
if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)
print(nn_test_ap, nn_test_auc )


dst_nodes = np.unique(full_data.destinations)
print(len(dst_nodes))


if tgn.use_memory:
    if tgn.memory_update_at_start:
    # Update memory for all nodes with messages stored in previous batches
        memory, last_update = tgn.get_updated_memory(list(range(tgn.n_nodes)),
                                              tgn.memory.messages)
    else:
        memory = tgn.memory.get_memory(list(range(tgn.n_nodes)))
        last_update = tgn.memory.last_update

        
ranks = []
total = len(test_data.sources)
if not tgn.use_memory:
  memory = None
with torch.no_grad():
    tgn = tgn.eval()
    ct = 0
    for src,dst,ts in zip(test_data.sources, test_data.destinations, test_data.timestamps):
        #print(src,dst,ts)
        print("\r%d/%d"%(ct,total),end="")
        ct += 1
        nodes = np.concatenate((np.array([src]),dst_nodes))
        times = np.zeros(len(nodes))
        times.fill(ts)
        time_diffs = None
        if tgn.use_memory:
          time_diffs = torch.LongTensor(times).to(tgn.device) - last_update[nodes].long()
          time_diffs = (time_diffs - tgn.mean_time_shift_src) / tgn.std_time_shift_src
        node_embedding = tgn.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=times,
                                                             n_layers=tgn.n_layers,
                                                             n_neighbors=NUM_NEIGHBORS,
                                                             time_diffs=time_diffs)
        src_embed = node_embedding[0]
        target_embed = node_embedding[1:]
        src_embed = src_embed.repeat(target_embed.shape[0],1)
        #print(src_embed.shape,target_embed.shape)
        scores = tgn.affinity_score(src_embed,target_embed).squeeze(dim=0).squeeze(dim=1).sigmoid().cpu().numpy()
        scores = list(zip(dst_nodes,scores))
        scores.sort(key=lambda val: val[1],reverse=True)
        #print(len(scores))
        ranks.append(calculate_rank(scores,dst))
        
mrr = np.mean([1.0 / r for r in ranks])
rec1 = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
rec5 = sum(np.array(ranks) <= 5)*1.0 / len(ranks)
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
print(mrr,rec1,rec5,rec10)
logger.info("For refreshing the dynamic embeddings for hard negatives at {}, MRR {}, rec1 {}, rec5 {} and rec10 {} for test data".format(DYNAMIC_EMBEDDING_REFRESH,mrr,rec1,rec5,rec10))


if save_result_file != '':
  import csv 
  import os
  # list of column names
  field_names = ['data', 'er', 'warmup_epochs','total_epochs', 'random_negatives', 'hard_negatives','MRR','RECALL@1','RECALL@5','RECALL@10']
  result = [DATA,DYNAMIC_EMBEDDING_REFRESH,WARMUP_EPOCH,NUM_EPOCH,NUM_RANDOM_SAMPLES,NUM_HARD_SAMPLES,mrr,rec1,rec5,rec10]
  if not os.path.exists(save_result_file):
      with open(save_result_file, 'w') as fp:
          writer = csv.writer(fp)
          writer.writerow(field_names)

  with open(save_result_file, 'a') as fp:
      writer = csv.writer(fp)
      writer.writerow(result)