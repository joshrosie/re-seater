import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np

from utils import data_utils


class SequenceDataset(Dataset):
    '''
    For dual-encoder (two-towel) models
    '''
    def __init__(self, data_args, mode=None) -> None:
        '''
        Args:
            data_args(object): data setting arguments
            mode(string): 'training', 'validation' or 'test'
        '''
        super().__init__()
        self.dataset_name = data_args.dataset_name
        assert mode in ['training', 'validation', 'test'], 'invalid mode!'
        self.mode = mode
        self.dm = data_args

        self._build()

    def _build(self):
        self.reco_his_max_len = self.dm.reco_his_max_length #padding or truncation length
        
        if self.mode == 'training':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.training_file, sep='\t', engine="pyarrow")
        elif self.mode == 'validation':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.validation_file, sep='\t', engine="pyarrow")
            reco_his_seq['predicting_items'] = reco_his_seq['predicting_items'].apply(
                lambda x:eval(x) # string --> list
            )

        elif self.mode == 'test':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.test_file, sep='\t', engine="pyarrow")
            reco_his_seq['predicting_items'] = reco_his_seq['predicting_items'].apply(
                lambda x:eval(x) # string --> list
            )

        reco_his_seq.columns = ['uid','his_seq','next_item']

        self.record = reco_his_seq
        
        if self.mode == 'training':
            self.item_table_with_popularity = self.record['next_item'].tolist()
        print('build {} dataset successfully'.format(self.mode))

    def __len__(self):
        return self.record.shape[0]

    def _parse_line(self, index):
        line = self.record.iloc[index]

        uid, user_his_seq, next_item = line['uid'], line['his_seq'], line['next_item']
        
        user_his_seq = eval(user_his_seq)[-self.reco_his_max_len:] # string --> list

        if self.reco_his_max_len > len(user_his_seq):
            user_his_seq = user_his_seq + [0] * (self.reco_his_max_len - len(user_his_seq))

        assert self.reco_his_max_len == len(user_his_seq)

        return uid, user_his_seq, next_item


    def __getitem__(self, index):

        uid, user_his_seq, next_item = self._parse_line(index)

        if self.mode == 'training':

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
                torch.tensor(next_item, dtype=torch.long),
            )

        else:

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long)
            )
        
        return cur_tensor

        
class SASREC_Dataset(SequenceDataset):
    def __init__(self, data_args, mode=None) -> None:
        super().__init__(data_args, mode)

class SEATER_Dataset(SequenceDataset):
    def __init__(self, data_args, mode=None) -> None:
        super().__init__(data_args, mode)

        self.mapping_matrix = data_utils.load_npy_file(
            load_path=self.dm.tree_data_par_path, filename=self.dm.tree_based_itemID_2_indexID
        ) # containing start_token, end_token and pad_token

        self.prefix_allowed_token = data_utils.load_npy_file(
            self.dm.tree_data_par_path, self.dm.tree_based_prefix_tree
        )

        self.tree_nodes_num = self.prefix_allowed_token.shape[0]
        self.item_num = self.mapping_matrix.shape[0]
        self.item_IDs = np.arange(1, self.item_num)
        self.max_len = self.mapping_matrix.shape[1]

        self.num_tree_branches = self.prefix_allowed_token.shape[1]

        self.num_neg = self.dm.num_neg

        self._construct_prefixe_aware_item_index_groups()

        # self.validate_tree(
        #     mapping_matrix=self.mapping_matrix,
        #     item_index_groups=self.item_index_groups,
        #     num_items=self.item_num,
            
        # )
    
    def validate_tree(self,mapping_matrix, item_index_groups, num_items):
        used_nodes = set(mapping_matrix[:, :-1].flatten())
        unused_nodes = []

        print(f"\n[INFO] Total items: {num_items}")
        print(f"[INFO] Mapping matrix shape: {mapping_matrix.shape}")
        print(f"[INFO] Total nodes in item_index_groups: {len(item_index_groups)}")

        for i in range(len(item_index_groups)):
            if len(item_index_groups[i]) == 0 and i not in used_nodes:
                unused_nodes.append(i)

        print(f"[CHECK] Unused nodes: {len(unused_nodes)}")
        if unused_nodes:
            print(f"First 10 unused nodes: {unused_nodes[:10]}")
        
        

        # Print some sample item paths
        print("\n[INFO] Sample item paths:")
        for i in range(min(100, num_items)):
            print(f"Item {i}: {mapping_matrix[i]}")


    def _construct_prefixe_aware_item_index_groups(self):
        '''
        create mapping from each token to identifiers containing this token
        '''

        self.item_index_groups = []
        for i in range(self.tree_nodes_num):
            self.item_index_groups.append([])
        
        for item_index in self.mapping_matrix[1:, :]:
            tmp = item_index[:-1]
            for node in tmp:
                self.item_index_groups[node].append(item_index)
        
        self.item_index_groups = [np.array(group) for group in self.item_index_groups]

        for i in range(self.item_num+3, self.tree_nodes_num):
            assert len(self.item_index_groups[i]) > 1, f'node number: {i}'


    # def neg_sample(self, next_item):
    #     '''used for imbalanced tree'''
    #     neg_items = []
    #     while len(neg_items) < self.num_neg: # 10 neg samples  
    #         neg_item = random.choice(self.item_IDs)

    #         while neg_item == next_item:
    #             neg_item = random.choice(self.item_IDs)
    #         neg_items.append(self.mapping_matrix[neg_item])
        
    #     neg_items = np.array(neg_items)

    #     return neg_items


    def neg_sample(self, next_item):  
        '''
        negative sampling for ranking loss (triplet loss)
        only used for balanced trees. if imbalanced do not use
        '''
        
        sample_nodes = next_item[:-2]
        sample_nodes = sample_nodes[-self.num_neg:]
        neg_item_index_ls = []
        for node in sample_nodes:
            
            all_possible_items = self.item_index_groups[node]
            
            neg_item = random.choice(all_possible_items)
            while np.all( neg_item == next_item ):
                neg_item = random.choice(all_possible_items)
            neg_item_index_ls.append(neg_item)

        neg_item_index_ls = np.array(neg_item_index_ls)

        return neg_item_index_ls

    def __getitem__(self, index):

        uid, user_his_seq, next_item = self._parse_line(index)

        if self.mode == 'training':
            
            next_item_index_id = self.mapping_matrix[next_item]
            neg_item_index_id = self.neg_sample(next_item_index_id)
            # neg_item_index_id = self.neg_sample(next_item)

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
                torch.tensor(next_item_index_id, dtype=torch.long),
                torch.tensor(neg_item_index_id, dtype=torch.long)
            )

        else:

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
            )
        
        return cur_tensor


class BERT4REC_Dataset(Dataset):
    def __init__(self, data_args, mode=None) -> None:
        super().__init__()
        assert mode in ['training', 'validation', 'test'], 'Invalid mode!'
        self.mode = mode
        self.dm = data_args

        self.reco_his_max_len = self.dm.reco_his_max_length
        self.mask_token = self.dm.item_ID_num + 1
        self.mask_prob = self.dm.mask_prob

        print('mask token: {}, mask prob: {}'.format(self.mask_token, self.mask_prob))
        print('reco_his_max_len: {}'.format(self.reco_his_max_len))

        self._build()

    def _build(self):
        if self.mode == 'training':
            df = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.training_file, sep='\t', engine="pyarrow")
            df.columns = ['uid', 'his_seq']
        elif self.mode == 'validation':
            df = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.validation_file, sep='\t', engine="pyarrow")
            df.columns = ['uid', 'his_seq', 'next_item']
            df['next_item'] = df['next_item'].apply(lambda x: eval(x))
        elif self.mode == 'test':
            df = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.test_file, sep='\t', engine="pyarrow")
            df.columns = ['uid', 'his_seq', 'next_item']
            df['next_item'] = df['next_item'].apply(lambda x: eval(x))

        self.record = df
        print('build {} dataset successfully'.format(self.mode))

    def __len__(self):
        return len(self.record)

    def _pad_seq(self, seq, is_training=True):
        seq = seq[-self.reco_his_max_len:]
        if is_training:
            return [0] * (self.reco_his_max_len+1 - len(seq)) + seq
        else:
            return [0] * (self.reco_his_max_len - len(seq)) + seq + [0]

    def __getitem__(self, index):
        row = self.record.iloc[index]
        uid = row['uid']
        his_seq = eval(row['his_seq']) if isinstance(row['his_seq'], str) else row['his_seq']
        if self.mode == 'training':
            his_seq = self._pad_seq(his_seq)
            tokens, labels = self._random_mask(his_seq)
            return (
                torch.tensor(uid, dtype=torch.long),
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )
        else:
            # Find the position of the last real item
            his_seq = self._pad_seq(his_seq, is_training=False)
            # real_len = sum([1 for x in his_seq if x != 0])
            # if real_len == 0:
            #     raise ValueError("Sequence is all padding; cannot insert mask.")
            # his_seq[real_len] = self.mask_token
            
            his_seq[-1] = self.mask_token
            return (
                torch.tensor(uid, dtype=torch.long),
                torch.tensor(his_seq, dtype=torch.long)
            )


    def _random_mask(self, seq):
        tokens, labels = [], []
        for item in seq:
            if item == 0:
                tokens.append(0)
                labels.append(0)
                continue
            if random.random() < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)
        return tokens, labels
    

GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, **kwargs):
    return DataLoader(  data_set, batch_size = bs,
                        pin_memory = True, 
                        worker_init_fn=worker_init_fn, **kwargs
                    )