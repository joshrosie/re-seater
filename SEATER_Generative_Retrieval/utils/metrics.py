import numpy as np
from utils.intra_list_diversity import ILD
from utils.gini_coefficient import GiniCoefficient

def eva(pre, ground_truth, num_items=0, id2cat=None, comi_ndcg=False):
    calcGini = True
    
    hit20, recall20, NDCG20, hit50, recall50, NDCG50, gini20, gini50 = (0, 0, 0, 0, 0, 0, 0, 0)
    Gini = GiniCoefficient()
    

    epsilon = 0.1 ** 10

    # Initialize sets to collect all unique recommended items
    all_recommended_items_20 = set()
    all_recommended_items_50 = set()
    
    for i in range(len(ground_truth)):
        one_DCG20, one_recall20, IDCG20, one_hit20, one_DCG50, one_recall50, IDCG50, one_hit50 = (
        0, 0, 0, 0, 0, 0, 0, 0)
        top_20_item = pre[i][0:20].tolist()
        top_50_item = pre[i][0:50].tolist()
        positive_item = ground_truth[i]


           # <-- ADDED: Update the sets with the recommended items for this user
        all_recommended_items_20.update(top_20_item)
        all_recommended_items_50.update(top_50_item)

        for pos, iid in enumerate(top_20_item): 
            if iid in positive_item:
                one_recall20 += 1
                one_DCG20 += 1 / np.log2(pos + 2)
        for pos, iid in enumerate(top_50_item): 
            if iid in positive_item:
                one_recall50 += 1
                one_DCG50 += 1 / np.log2(pos + 2)

        if comi_ndcg:
            '''calculate NDCG as Comirec, which is incorrect'''
            for pos in range(one_recall20):
                IDCG20 += 1 / np.log2(pos + 2)
            for pos in range(one_recall50):
                IDCG50 += 1 / np.log2(pos + 2)
        else:
            '''caculate according to the formal defination'''
            for pos in range(len(positive_item[:20])):
                IDCG20 += 1 / np.log2(pos + 2)
            for pos in range(len(positive_item[:50])):
                IDCG50 += 1 / np.log2(pos + 2)

        NDCG20 += one_DCG20 / max(IDCG20, epsilon) # avoid dividing zero
        NDCG50 += one_DCG50 / max(IDCG50, epsilon)
        top_20_item = set(top_20_item)
        top_50_item = set(top_50_item)
        positive_item = set(positive_item)
        if len(top_20_item & positive_item) > 0:
            hit20 += 1
        if len(top_50_item & positive_item) > 0:
            hit50 += 1
        recall20 += len(top_20_item & positive_item) / max(len(positive_item), epsilon)
        recall50 += len(top_50_item & positive_item) / max(len(positive_item), epsilon)

        if calcGini and id2cat is not None:
            gini20 += Gini.calculate_list_gini(top_20_item, id2cat)
            gini50 += Gini.calculate_list_gini(top_50_item, id2cat)

        

    hit20, recall20, NDCG20, hit50, recall50, NDCG50, gini20, gini50 = \
        hit20 / len(ground_truth), recall20 / len(ground_truth), NDCG20 / len(ground_truth),\
        hit50 / len(ground_truth), recall50 / len(ground_truth), NDCG50 / len(ground_truth),\
        gini20 / len(ground_truth), gini50 / len(ground_truth)
    
 
    # Calculate coverage after the loop is finished
    # Coverage is the proportion of all items in the catalog that were recommended at least once.
    coverage20 = len(all_recommended_items_20) / num_items if num_items > 0 else 0
    coverage50 = len(all_recommended_items_50) / num_items if num_items > 0 else 0
    
    result = {
        'ndcg@20': round(NDCG20, 4), 'ndcg@50': round(NDCG50, 4),
        'hit@20': round(hit20, 4), 'hit@50': round(hit50, 4),
        'recall@20': round(recall20, 4), 'recall@50': round(recall50, 4),
        'gini@20': round(gini20, 4), 'gini@50': round(gini50, 4),
        'coverage@20': round(coverage20, 4), 'coverage@50': round(coverage50, 4)
    }

    return result
