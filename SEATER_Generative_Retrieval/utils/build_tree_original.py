import numpy as np
import pickle
from queue import Queue
import pandas as pd
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from typing import Optional
import os
import warnings
import time


def build_hieraichical_clustering_tree_original(
        item_emb_path : str, 
        output_file_path : str, 
        vocab_size : int = 10, 
        is_balanced: Optional[bool] = True
        ) -> None:
    '''
    Args:
        vocab_size: number of the trie's branches

    '''
    start_all = time.time()
    warnings.filterwarnings('ignore')

    all_item_emb = np.load(item_emb_path)
    num_items = all_item_emb.shape[0]

    max_len = None
    
    start_token = num_items
    end_token = num_items+1
    pad_token = num_items+2

    all_items = np.arange(start=1, stop=num_items, step=1, dtype=np.int64) #remove padding item

    print('start hierarchical clustering')

    '''hierarchical clustering items'''
    start_clustering = time.time()
    indexIDs, itemIDs = hierarchical_clustering(all_items, all_item_emb, vocab_size, is_balanced)
    end_clustering = time.time()
    print(f'Hierarchical clustering finished in {end_clustering - start_clustering:.2f}s')

    print('start build the prefix tree')
    start_tree = time.time()
    '''build prefix tree'''
    builder = TreeBuilder(start_token, num_items)
    tree_node_index_ls = []
    for index, item in zip(indexIDs, itemIDs):
        # We re-encode each node on the tree, assigning unique IDs to them.
        this_tree_index = builder.add(index, item)
        tree_node_index_ls.append(this_tree_index)

    max_len = max([len(i) for i in tree_node_index_ls])
    itemID_2_tree_indexID = np.ones((num_items, max_len), dtype=np.int64) * pad_token

    for itemID, tree_node_index in zip(itemIDs, tree_node_index_ls):
        itemID_2_tree_indexID[itemID][:len(tree_node_index)] = tree_node_index

    # the last token of the longest identifier is the end token. 
    # So the last token can be removed
    itemID_2_tree_indexID = itemID_2_tree_indexID[:, :-1] 

    print('build tree successfully')
    end_tree = time.time()

    root = builder.build()

    tree_node, allowed_next_tokens_ls = level_order_traversal(root)

    print('BFS successfully')

    tree_node_allowed_next_tokens = np.ones((max(tree_node)+1, vocab_size), dtype=np.int64) * pad_token

    for tree_node_id, next_tokens in zip(tree_node, allowed_next_tokens_ls):
        assert len(next_tokens) <= vocab_size
        if len(next_tokens) > 0:
            tree_node_allowed_next_tokens[tree_node_id][:len(next_tokens)] = next_tokens
        else:
            tree_node_allowed_next_tokens[tree_node_id][0] = end_token

    tree_node_allowed_next_tokens[num_items+1][0] = end_token
    tree_node_allowed_next_tokens[num_items+1][1:] = [pad_token]*(vocab_size-1)
    tree_node_allowed_next_tokens[num_items+2][0] = end_token
    tree_node_allowed_next_tokens[num_items+2][1:] = [pad_token]*(vocab_size-1)

    end_all = time.time()
    print(f'\n✅ Total execution time: {end_all - start_all:.2f}s')
    print(f' - Clustering time: {end_clustering - start_clustering:.2f}s')
    print(f' - Tree building time: {end_tree - start_tree:.2f}s')


    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    np.save(f'{output_file_path}/itemID_2_tree_indexID.npy', itemID_2_tree_indexID,  allow_pickle=True)
    np.save(f'{output_file_path}/tree_node_allowed_next_tokens.npy', tree_node_allowed_next_tokens,  allow_pickle=True)

    print('finish tree building and save files successfully!')

    print(f'\nFinish tree building successfully! Files saved in "{output_file_path}"')


    num_nodes = len(tree_node)
    num_leaves = sum(1 for children in allowed_next_tokens_ls if len(children) == 0)
    avg_branching = np.mean([len(children) for children in allowed_next_tokens_ls if children]) if allowed_next_tokens_ls else 0
    max_depth = max(len(p.split()) for p in indexIDs) if indexIDs else 0

    print(f'\n🌲 Tree statistics:')
    print(f' - Total nodes: {num_nodes}')
    print(f' - Leaf nodes: {num_leaves}')
    print(f' - Average branching factor: {avg_branching:.2f}')
    print(f' - Max tree depth (estimated): {max_depth}')

    print(f'\n📐 Output arrays:')
    print(f' - itemID_2_tree_indexID shape: {itemID_2_tree_indexID.shape}')
    print(f' - tree_node_allowed_next_tokens shape: {tree_node_allowed_next_tokens.shape}')
    non_pad = np.sum(tree_node_allowed_next_tokens != pad_token)
    total = tree_node_allowed_next_tokens.size
    print(f' - tree_node_allowed_next_tokens sparsity: {non_pad / total:.2%} non-pad entries')

    path_lengths = [len(path.split()) for path in indexIDs]
    print(f"\n📏 Path length statistics:")
    print(f" - Min: {np.min(path_lengths)}")
    print(f" - Max: {np.max(path_lengths)}")
    print(f" - Mean: {np.mean(path_lengths):.2f}")
    print(f" - Median: {np.median(path_lengths)}")


def random_cluster(input_items, item_emb_matrix, k:int, is_balanced: bool):
    results_items = []
    chunk_size = len(input_items) // k 
    np.random.shuffle(input_items) # random
    # cluster data into k groups
    for i in range(k):
        start = i * chunk_size
        end = start + chunk_size
        if i == k - 1:
            end = len(input_items)  
        chunk = input_items[start:end]

        results_items.append(chunk)

    return results_items

def cluster(input_items, item_emb_matrix, k:int, is_balanced: bool):
    '''
    Args:
        input_items: item id list, [10, 20, 33, 42], only containing items to be clustered
        item_emb_matrix: embeddings of all items, (num_all_items, dim)
        k: number of clusters
    '''

    X = item_emb_matrix[input_items]

    if is_balanced:
        min_size_per_cluster = X.shape[0] // k
        kmeans = KMeansConstrained(
            n_clusters=k, 
            random_state=0, 
            n_jobs = -2,
            size_min = min_size_per_cluster,
            size_max = min_size_per_cluster + 1,
            ).fit(X)
        cluster_labels = kmeans.labels_ # output is like : array([1, 3, 1, 2, 4, 0], dtype=int32)   

        cluster_nums = len(np.unique(cluster_labels))
        assert cluster_nums == k

        # group data into k chunks
        results_items = []
        for i in range(k):
            chunk = input_items[cluster_labels==i]
            results_items.append(chunk)
    else:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        cluster_labels = kmeans.labels_ # output is like : array([1, 3, 1, 2, 4, 0], dtype=int32)

        cluster_nums = len(np.unique(cluster_labels))

        results_items = []
        if cluster_nums < (k // 2):
            '''
            Number of distinct clusters found smaller than n_clusters(k)
            any cluster with no items --> these items cannot be grouped into k clusters
            '''
            chunk_size = len(input_items) // k 
            # group data into k chunks
            for i in range(k):
                start = i * chunk_size
                end = start + chunk_size
                if i == k - 1:
                    end = len(input_items)  
                chunk = input_items[start:end]
                # print(len(chunk))
                results_items.append(chunk)
        else:
            
            for i in range(k):
                chunk = input_items[cluster_labels==i]
                results_items.append(chunk)

    return results_items

def hierarchical_clustering(items, item_emb_matrix, k:int, is_balanced:bool, level=0):
    start_time = time.time()
    J = []
    new_items = []

    cluster_items = cluster(items, item_emb_matrix, k, is_balanced)
    # cluster_items = cluster(items, item_emb_matrix, k, False) # imbalanced tree
    # cluster_items = random_cluster(items, item_emb_matrix, k, is_balanced) # random construct trees

    level_time = time.time() - start_time
    print(f"[LEVEL {level}] Clustering {len(items)} items into {k} clusters took {level_time:.2f}s")

    for i in range(k):
        J_current = [str(i)] * len(cluster_items[i])

        if len(cluster_items[i]) > k:
            rest_indexIDs, rest_new_items = hierarchical_clustering(cluster_items[i], item_emb_matrix, k, is_balanced, level+1)
        else:
            if len(cluster_items[i]) == 1:
                rest_indexIDs = [None]
            else:
                rest_indexIDs = [str(i) for i in range(len(cluster_items[i]))]
            rest_new_items = cluster_items[i]
        J_cluster = [ ' '.join([J_current[i], rest_indexIDs[i]]) if rest_indexIDs[i]!= None else J_current[i]
                      for i in range(len(J_current))
                    ]
        new_items_cluster = rest_new_items

        J.extend(J_cluster)
        new_items.extend(new_items_cluster)

        
    return J, new_items



class Node(object):
    def __init__(self, token_id, node_id) -> None:
        self.token_id = int(token_id)
        self.node_id = int(node_id)
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self, start_token, num_item) -> None:
        self.root = Node(start_token, num_item) # s means start token
        self.tree_node_step = num_item + 3 # num_item -> start, num_item+1 -> end, num_item+2:pad
        self.end_token = num_item + 1

    def build(self) -> Node:
        return self.root

    def add(self, seq, item_id) -> None:
        '''
        seq is string representing id, without pad_token, start_token and end_token
        e.g. '1 2 3 4 5 6 7 8 9 10'
        '''
        index_id_ls = []

        cur = self.root
        index_id_ls.append(cur.node_id)
        seq = [int(s) for s in seq.split(' ')]
        for tok in seq[:-1]:
            if tok not in cur.children:
                cur.children[tok] = Node(tok, self.tree_node_step)
                self.tree_node_step += 1
            index_id_ls.append(cur.children[tok].node_id)
            cur = cur.children[tok]
        assert seq[-1] not in cur.children
        cur.children[seq[-1]] = Node(seq[-1], item_id)
        index_id_ls.append(item_id)
        index_id_ls.append(self.end_token)

        return index_id_ls

def level_order_traversal(root):
    '''BFS traverse prefix tree'''
    if not root:
        return []
    q = Queue()
    q.put(root)

    res = []
    res_children = []
    while not q.empty():
        node = q.get()
        res.append(node.node_id)
        cur_children = []
        for child_token, child_node in node.children.items():
            q.put(child_node)
            cur_children.append(child_node.node_id)
        res_children.append(cur_children)
            
    return res, res_children