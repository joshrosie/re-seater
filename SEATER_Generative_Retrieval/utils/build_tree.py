# import numpy as np
# import pickle
# from queue import Queue
# import pandas as pd
# from sklearn.cluster import KMeans
# from k_means_constrained import KMeansConstrained
# from typing import Optional
# import os
# import warnings


# def build_hieraichical_clustering_tree(
#         item_emb_path : str, 
#         output_file_path : str, 
#         vocab_size : int = 10, 
#         is_balanced: Optional[bool] = True
#         ) -> None:
#     '''
#     Args:
#         vocab_size: number of the trie's branches

#     '''
#     warnings.filterwarnings('ignore')

#     all_item_emb = np.load(item_emb_path)
#     num_items = all_item_emb.shape[0]

#     max_len = None
    
#     start_token = num_items
#     end_token = num_items+1
#     pad_token = num_items+2

#     all_items = np.arange(start=1, stop=num_items, step=1, dtype=np.int64) #remove padding item

#     print('start hierarchical clustering')

#     '''hierarchical clustering items'''
#     indexIDs, itemIDs = hierarchical_clustering(all_items, all_item_emb, vocab_size, is_balanced)

#     print('start build the prefix tree')
#     '''build prefix tree'''
#     builder = TreeBuilder(start_token, num_items)
#     tree_node_index_ls = []
#     for index, item in zip(indexIDs, itemIDs):
#         # We re-encode each node on the tree, assigning unique IDs to them.
#         this_tree_index = builder.add(index, item)
#         tree_node_index_ls.append(this_tree_index)

#     max_len = max([len(i) for i in tree_node_index_ls])
#     itemID_2_tree_indexID = np.ones((num_items, max_len), dtype=np.int64) * pad_token

#     for itemID, tree_node_index in zip(itemIDs, tree_node_index_ls):
#         itemID_2_tree_indexID[itemID][:len(tree_node_index)] = tree_node_index

#     # the last token of the longest identifier is the end token. 
#     # So the last token can be removed
#     itemID_2_tree_indexID = itemID_2_tree_indexID[:, :-1] 

#     print('build tree successfully')

#     root = builder.build()

#     tree_node, allowed_next_tokens_ls = level_order_traversal(root)

#     print('BFS successfully')

#     tree_node_allowed_next_tokens = np.ones((max(tree_node)+1, vocab_size), dtype=np.int64) * pad_token

#     for tree_node_id, next_tokens in zip(tree_node, allowed_next_tokens_ls):
#         assert len(next_tokens) <= vocab_size
#         if len(next_tokens) > 0:
#             tree_node_allowed_next_tokens[tree_node_id][:len(next_tokens)] = next_tokens
#         else:
#             tree_node_allowed_next_tokens[tree_node_id][0] = end_token

#     tree_node_allowed_next_tokens[num_items+1][0] = end_token
#     tree_node_allowed_next_tokens[num_items+1][1:] = [pad_token]*(vocab_size-1)
#     tree_node_allowed_next_tokens[num_items+2][0] = end_token
#     tree_node_allowed_next_tokens[num_items+2][1:] = [pad_token]*(vocab_size-1)

#     if not os.path.exists(output_file_path):
#         os.mkdir(output_file_path)

#     np.save(f'{output_file_path}/itemID_2_tree_indexID.npy', itemID_2_tree_indexID,  allow_pickle=True)
#     np.save(f'{output_file_path}/tree_node_allowed_next_tokens.npy', tree_node_allowed_next_tokens,  allow_pickle=True)

#     print('finish tree building and save files successfully!')

# def random_cluster(input_items, item_emb_matrix, k:int, is_balanced: bool):
#     results_items = []
#     chunk_size = len(input_items) // k 
#     np.random.shuffle(input_items) # random
#     # cluster data into k groups
#     for i in range(k):
#         start = i * chunk_size
#         end = start + chunk_size
#         if i == k - 1:
#             end = len(input_items)  
#         chunk = input_items[start:end]

#         results_items.append(chunk)

#     return results_items

# def cluster(input_items, item_emb_matrix, k:int, is_balanced: bool):
#     '''
#     Args:
#         input_items: item id list, [10, 20, 33, 42], only containing items to be clustered
#         item_emb_matrix: embeddings of all items, (num_all_items, dim)
#         k: number of clusters
#     '''

#     X = item_emb_matrix[input_items]

#     if is_balanced:
#         min_size_per_cluster = X.shape[0] // k
#         kmeans = KMeansConstrained(
#             n_clusters=k, 
#             random_state=0, 
#             n_jobs = -2,
#             size_min = min_size_per_cluster,
#             size_max = min_size_per_cluster + 1,
#             ).fit(X)
#         cluster_labels = kmeans.labels_ # output is like : array([1, 3, 1, 2, 4, 0], dtype=int32)   

#         cluster_nums = len(np.unique(cluster_labels))
#         assert cluster_nums == k

#         # group data into k chunks
#         results_items = []
#         for i in range(k):
#             chunk = input_items[cluster_labels==i]
#             results_items.append(chunk)
#     else:
#         kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
#         cluster_labels = kmeans.labels_ # output is like : array([1, 3, 1, 2, 4, 0], dtype=int32)

#         cluster_nums = len(np.unique(cluster_labels))

#         results_items = []
#         if cluster_nums < (k // 2):
#             '''
#             Number of distinct clusters found smaller than n_clusters(k)
#             any cluster with no items --> these items cannot be grouped into k clusters
#             '''
#             chunk_size = len(input_items) // k 
#             # group data into k chunks
#             for i in range(k):
#                 start = i * chunk_size
#                 end = start + chunk_size
#                 if i == k - 1:
#                     end = len(input_items)  
#                 chunk = input_items[start:end]
#                 # print(len(chunk))
#                 results_items.append(chunk)
#         else:
            
#             for i in range(k):
#                 chunk = input_items[cluster_labels==i]
#                 results_items.append(chunk)

#     return results_items

# def hierarchical_clustering(items, item_emb_matrix, k:int, is_balanced:bool):
#     J = []
#     new_items = []

#     cluster_items = cluster(items, item_emb_matrix, k, is_balanced)
#     # cluster_items = cluster(items, item_emb_matrix, k, False) # imbalanced tree
#     # cluster_items = random_cluster(items, item_emb_matrix, k, is_balanced) # random construct trees

#     for i in range(k):
#         J_current = [str(i)] * len(cluster_items[i])

#         if len(cluster_items[i]) > k:
#             rest_indexIDs, rest_new_items = hierarchical_clustering(cluster_items[i], item_emb_matrix, k, is_balanced)
#         else:
#             if len(cluster_items[i]) == 1:
#                 rest_indexIDs = [None]
#             else:
#                 rest_indexIDs = [str(i) for i in range(len(cluster_items[i]))]
#             rest_new_items = cluster_items[i]
#         J_cluster = [ ' '.join([J_current[i], rest_indexIDs[i]]) if rest_indexIDs[i]!= None else J_current[i]
#                       for i in range(len(J_current))
#                     ]
#         new_items_cluster = rest_new_items

#         J.extend(J_cluster)
#         new_items.extend(new_items_cluster)

        
#     return J, new_items



# class Node(object):
#     def __init__(self, token_id, node_id) -> None:
#         self.token_id = int(token_id)
#         self.node_id = int(node_id)
#         self.children = {}

#     def __str__(self, level=0):
#         ret = "\t" * level + repr(self.token_id) + "\n"
#         for child in self.children.values():
#             ret += child.__str__(level + 1)
#         return ret

#     def __repr__(self):
#         return '<tree node representation>'


# class TreeBuilder(object):
#     def __init__(self, start_token, num_item) -> None:
#         self.root = Node(start_token, num_item) # s means start token
#         self.tree_node_step = num_item + 3 # num_item -> start, num_item+1 -> end, num_item+2:pad
#         self.end_token = num_item + 1

#     def build(self) -> Node:
#         return self.root

#     def add(self, seq, item_id) -> None:
#         '''
#         seq is string representing id, without pad_token, start_token and end_token
#         e.g. '1 2 3 4 5 6 7 8 9 10'
#         '''
#         index_id_ls = []

#         cur = self.root
#         index_id_ls.append(cur.node_id)
#         seq = [int(s) for s in seq.split(' ')]
#         for tok in seq[:-1]:
#             if tok not in cur.children:
#                 cur.children[tok] = Node(tok, self.tree_node_step)
#                 self.tree_node_step += 1
#             index_id_ls.append(cur.children[tok].node_id)
#             cur = cur.children[tok]
#         assert seq[-1] not in cur.children
#         cur.children[seq[-1]] = Node(seq[-1], item_id)
#         index_id_ls.append(item_id)
#         index_id_ls.append(self.end_token)

#         return index_id_ls

# def level_order_traversal(root):
#     '''BFS traverse prefix tree'''
#     if not root:
#         return []
#     q = Queue()
#     q.put(root)

#     res = []
#     res_children = []
#     while not q.empty():
#         node = q.get()
#         res.append(node.node_id)
#         cur_children = []
#         for child_token, child_node in node.children.items():
#             q.put(child_node)
#             cur_children.append(child_node.node_id)
#         res_children.append(cur_children)
            
#     return res, res_children




import numpy as np
from queue import Queue
from typing import Optional
import os
import warnings
import time
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans

# --- DEPENDENCIES ---
# This script now requires both faiss and scikit-learn
# pip install faiss-cpu scikit-learn (or faiss-gpu)

try:
    import faiss
except ImportError:
    print("FATAL: faiss library not found. Please install it: pip install faiss-cpu")
    exit()

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("FATAL: scikit-learn library not found. Please install it: pip install scikit-learn")
    exit()

# ==============================================================================
# 1. CLUSTERING FUNCTIONS
# ==============================================================================

# --- A) Fast FAISS implementation for the BALANCED case ---
def faiss_cluster_balanced(input_items, item_emb_matrix, k: int, gpu: bool = True):
    """
    Performs balanced clustering using a two-step approach with modern Faiss.
    1. Use faiss.Kmeans to find optimal centroids quickly.
    2. Use a fast, greedy assignment algorithm to assign points to centroids, ensuring balance.
    """
    X = item_emb_matrix[input_items].astype('float32')
    n_samples, d = X.shape
    faiss_threshold = 100 # Threshold for using Faiss Kmeans

    if k <= 0: return []
    if n_samples < k: return [np.array([item]) for item in input_items]

    if len(input_items)>faiss_threshold:
        # Step 1: Find optimal centroids with faiss.Kmeans
        num_gpus = faiss.get_num_gpus() if gpu else 0
        kmeans = faiss.Kmeans(d=d, k=k, niter=20, nredo=5, verbose=False, gpu=(num_gpus > 0))
        kmeans.train(X)
        centroids = kmeans.centroids
    else:
        kmeans = KMeans(n_clusters=k, n_init=5, max_iter=20, verbose=0, random_state=42)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_

    # Step 2: Perform a fast, greedy balanced assignment
    D = np.zeros((n_samples, k), dtype='float32')
    for i in range(k):
        D[:, i] = np.sum((X - centroids[i])**2, axis=1)

    assignments = []
    for i in range(n_samples):
        for j in range(k):
            assignments.append((D[i, j], i, j))
    assignments.sort()

    cluster_labels = np.full(n_samples, -1, dtype=np.int64)
    cluster_sizes = np.zeros(k, dtype=np.int32)
    
    min_size = n_samples // k
    target_sizes = np.full(k, min_size)
    n_big_clusters = n_samples % k
    if n_big_clusters > 0:
      target_sizes[:n_big_clusters] = min_size + 1

    assigned_count = 0
    for dist, point_idx, cluster_idx in assignments:
        if assigned_count == n_samples: break
        if cluster_labels[point_idx] == -1 and cluster_sizes[cluster_idx] < target_sizes[cluster_idx]:
            cluster_labels[point_idx] = cluster_idx
            cluster_sizes[cluster_idx] += 1
            assigned_count += 1
            
    results_items = [input_items[cluster_labels == i] for i in range(k)]
    return results_items


# --- B) The ORIGINAL logic from your script for the UNBALANCED case ---
def original_cluster_unbalanced(input_items, item_emb_matrix, k: int):
    """
    This is the exact logic from your original script for the unbalanced case,
    using sklearn.KMeans and the fallback to random chunking.
    """
    X = item_emb_matrix[input_items]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    cluster_labels = kmeans.labels_
    cluster_nums = len(np.unique(cluster_labels))

    results_items = []
    if cluster_nums < (k // 2):
        # Fallback logic: If KMeans fails, randomly chunk the data
        print(f"Warning: KMeans found only {cluster_nums}/{k} clusters. Falling back to random chunking.")
        chunk_size = len(input_items) // k 
        shuffled_items = np.copy(input_items)
        np.random.shuffle(shuffled_items)
        for i in range(k):
            start = i * chunk_size
            end = start + chunk_size
            if i == k - 1:
                end = len(shuffled_items)  
            chunk = shuffled_items[start:end]
            results_items.append(chunk)
    else:
        # Standard logic: Group items by the labels found
        for i in range(k):
            chunk = input_items[cluster_labels==i]
            results_items.append(chunk)

    return results_items

def cluster_old(input_items, item_emb_matrix, k:int, is_balanced: bool):
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


# ==============================================================================
# 2. HIERARCHICAL CLUSTERING DRIVER (DISPATCHER)
# ==============================================================================

def hierarchical_clustering(items, item_emb_matrix, k: int, is_balanced: bool, use_gpu: bool = True, level=0, num_items_for_greedy=2000):
    start_time = time.time()
    J = []
    new_items = []

    # --- Dispatch to the correct clustering function based on the flag ---
    if is_balanced and len(items) > num_items_for_greedy:
        # Use the new, fast Faiss implementation
        cluster_items = faiss_cluster_balanced(items, item_emb_matrix, k, gpu=use_gpu)
    elif is_balanced and len(items) <= num_items_for_greedy:
        cluster_items = cluster_old(items, item_emb_matrix, k, is_balanced)
    else:
        # Use the original sklearn implementation
        cluster_items = original_cluster_unbalanced(items, item_emb_matrix, k)
    
    level_time = time.time() - start_time
    print(f"[LEVEL {level}] Clustering {len(items)} items into {k} clusters took {level_time:.2f}s")


    # The rest of the logic remains the same
    for i in range(k):
        current_cluster = cluster_items[i]
        if len(current_cluster) == 0:
            continue
            
        J_current = [str(i)] * len(current_cluster)

        if len(current_cluster) > k:
            rest_indexIDs, rest_new_items = hierarchical_clustering(current_cluster, item_emb_matrix, k, is_balanced, use_gpu, level = level + 1, num_items_for_greedy=num_items_for_greedy)
        else:
            if len(current_cluster) == 1:
                rest_indexIDs = [None]
            else:
                rest_indexIDs = [str(j) for j in range(len(current_cluster))]
            rest_new_items = current_cluster
            
        J_cluster = [ ' '.join([J_current[idx], rest_indexIDs[idx]]) if rest_indexIDs[idx] is not None else J_current[idx]
                      for idx in range(len(J_current))
                    ]
        
        J.extend(J_cluster)
        new_items.extend(rest_new_items)
        
    return J, new_items


# ==============================================================================
# 3. TREE BUILDING & TRAVERSAL LOGIC (UNCHANGED FROM ORIGINAL)
# ==============================================================================

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

class TreeBuilder(object):
    def __init__(self, start_token, num_item) -> None:
        self.root = Node(start_token, num_item)
        self.tree_node_step = num_item + 3
        self.end_token = num_item + 1

    def build(self) -> Node:
        return self.root

    def add(self, seq, item_id) -> list:
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
        
        cur.children[seq[-1]] = Node(seq[-1], item_id)
        index_id_ls.append(item_id)
        index_id_ls.append(self.end_token)
        return index_id_ls

def level_order_traversal(root):
    if not root: return [], []
    q = Queue()
    q.put(root)
    res, res_children = [], []
    while not q.empty():
        node = q.get()
        res.append(node.node_id)
        cur_children = []
        for child_token, child_node in sorted(node.children.items()):
            q.put(child_node)
            cur_children.append(child_node.node_id)
        res_children.append(cur_children)
    return res, res_children


# ==============================================================================
# 4. MAIN WORKFLOW FUNCTION
# ==============================================================================

def build_hieraichical_clustering_tree(
        item_emb_path : str, 
        output_file_path : str, 
        vocab_size : int = 10, 
        is_balanced: Optional[bool] = True,
        use_gpu: Optional[bool] = True,
        num_items_for_greedy: int = 0
        ) -> None:
    '''
    Args:
        vocab_size: number of the trie's branches (k in k-means)
        is_balanced: whether to create balanced clusters at each step
        use_gpu: whether to use GPU for faiss clustering (requires faiss-gpu)
    '''
    warnings.filterwarnings('ignore', category=FutureWarning) # Ignore sklearn n_init warning
    warnings.filterwarnings('ignore', category=UserWarning)

    print(f'\n Building hierarchical clustering tree with faiss above {num_items_for_greedy} items, vocab_size={vocab_size}, is_balanced={is_balanced}, use_gpu={use_gpu}...')

    start_all = time.time()

    if use_gpu and (not faiss.get_num_gpus()):
        print("Warning: use_gpu=True but faiss-gpu is not installed or no GPU is found. Falling back to CPU.")
        use_gpu = False

    all_item_emb = np.load(item_emb_path)
    num_items = all_item_emb.shape[0]
    
    start_token = num_items
    end_token = num_items + 1
    pad_token = num_items + 2

    all_items = np.arange(start=1, stop=num_items, step=1, dtype=np.int64)

    print('Start hierarchical clustering...')
    start_clustering = time.time()
    indexIDs, itemIDs = hierarchical_clustering(all_items, all_item_emb, vocab_size, is_balanced, use_gpu, num_items_for_greedy=num_items_for_greedy)
    end_clustering = time.time()
    print(f'Hierarchical clustering finished in {end_clustering - start_clustering:.2f}s')

    print('Start building the prefix tree...')
    start_tree = time.time()
    builder = TreeBuilder(start_token, num_items)
    
    sorted_pairs = sorted(zip(itemIDs, indexIDs))
    itemID_2_path = {item: path for item, path in sorted_pairs}
    tree_node_index_ls = []
    sorted_itemIDs = sorted(itemID_2_path.keys())
    
    for item in sorted_itemIDs:
        index = itemID_2_path[item]
        this_tree_index = builder.add(index, item)
        tree_node_index_ls.append(this_tree_index)
        
    print('Tree building finished.')
    end_tree = time.time()


    max_len = max([len(i) for i in tree_node_index_ls]) if tree_node_index_ls else 0
    itemID_2_tree_indexID = np.ones((num_items, max_len), dtype=np.int64) * pad_token

    for itemID, tree_node_index in zip(sorted_itemIDs, tree_node_index_ls):
        itemID_2_tree_indexID[itemID][:len(tree_node_index)] = tree_node_index

    if max_len > 0:
        itemID_2_tree_indexID = itemID_2_tree_indexID[:, :-1] 

    root = builder.build()
    # tree_node, allowed_next_tokens_ls = level_order_traversal(root)
    # print('BFS traversal of the tree finished.')
    
    # max_node_id = builder.tree_node_step 
    # if tree_node:
    #     max_node_id = max(max_node_id, max(tree_node))

    tree_node, allowed_next_tokens_ls = level_order_traversal(root)

    print('BFS successfully')

    tree_node_allowed_next_tokens = np.ones((max(tree_node)+1, vocab_size), dtype=np.int64) * pad_token

   # tree_node_allowed_next_tokens = np.ones((max_node_id + 1, vocab_size), dtype=np.int64) * pad_token

    for tree_node_id, next_tokens in zip(tree_node, allowed_next_tokens_ls):
        if len(next_tokens) > 0:
            tree_node_allowed_next_tokens[tree_node_id][:len(next_tokens)] = next_tokens
        else:
            tree_node_allowed_next_tokens[tree_node_id][0] = end_token

    tree_node_allowed_next_tokens[end_token][0] = end_token
    tree_node_allowed_next_tokens[pad_token][0] = pad_token

    end_all = time.time()
    print(f'\n✅ Total execution time: {end_all - start_all:.2f}s')
    print(f' - Clustering time: {end_clustering - start_clustering:.2f}s')
    print(f' - Tree building time: {end_tree - start_tree:.2f}s')

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    np.save(f'{output_file_path}/itemID_2_tree_indexID.npy', itemID_2_tree_indexID)
    np.save(f'{output_file_path}/tree_node_allowed_next_tokens.npy', tree_node_allowed_next_tokens)

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
