
class Books_Config(object):
    def __init__(self) -> None:
        
        # data file path
        # self.datafile_par_path = './data_bert4rec/Books'
        self.datafile_par_path = './data/Books'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'
        # self.itemID_2_attr = 'vocab/item_2_attr_mapping.npy'

        # for tree-based index
        #self.tree_data_par_path = './data/Books/tree_data_SASREC'
        #self.tree_data_par_path = './data/Books/tree_data_SASREC_hybrid'
        self.tree_data_par_path = './data/Books/tree_data_SASREC_speedup_sklearn'

        # self.tree_data_par_path = './data/Books/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Books_SASREC_item_emb.npy'

        # dataset config
        #self.users_ID_num = 603671 Original
        # self.users_ID_num = 603668 Ours
        self.item_ID_num = 367982 + 1 # zero for padding Ours+Original same
        #self.item_cate_num = 1600 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20

class Yelp_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = './data/Yelp'
        # self.datafile_par_path = './data_bert4rec/Yelp'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        #self.tree_data_par_path = './data/Yelp/tree_data_SASREC'
        # self.tree_data_par_path = './data/Yelp/tree_data_SASREC_hybrid'
        #self.tree_data_par_path = './data/Yelp/tree_data_SASREC_speedup'
        #self.tree_data_par_path = './data/Yelp/tree_data_SASREC_normal'
        self.tree_data_par_path = './data/Yelp/tree_data_SASREC_speedup_sklearn'

        # self.tree_data_par_path = './data/Yelp/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Yelp_SASREC_item_emb.npy'

        #dataset config
        #self.users_ID_num = 31668
        #self.item_ID_num = 38048 + 1 # zero for padding -> DEFAULT ONE
        self.item_ID_num = 38048 + 1 # OURS: same as original

        # experiment const config
        self.reco_his_max_length = 20 

class MIND_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = './data/MIND'
        # self.datafile_par_path = './data_bert4rec/MIND'
        # self.training_file = 'dataset/training.tsv'
        # self.validation_file = 'dataset/validation.tsv'
        # self.test_file = 'dataset/test.tsv'

        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        #self.tree_data_par_path = '../../MIND/MIND/tree_data_SASREC'
        #self.tree_data_par_path = './data/MIND/tree_data_SASREC'
        #self.tree_data_par_path = './data/MIND/tree_data_SASREC_hybrid'
        #self.tree_data_par_path = './data/MIND/tree_data_SASREC_speedup'
        #self.tree_data_par_path = './data/MIND/tree_data_SASREC_normal'
        self.tree_data_par_path = './data/MIND/tree_data_SASREC_speedup_sklearn'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'      
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'
    
        self.two_tower_item_emb = 'vocab/MIND_SASREC_item_emb.npy'

        #dataset config
        #self.users_ID_num = 50000
        #self.item_ID_num = 39815 + 1 # Ours: Using history and predicting only positive impressions
        #self.item_ID_num = 33195 + 1 # Ours: Using only the history without the impressions
        self.item_ID_num = 39865 + 1  # Ours: Using history concatenated with positive impressions
        # experiment const config
        self.reco_his_max_length = 20 

class Yambda_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = './data/Yambda'
        # self.datafile_par_path = './data_bert4rec/Yambda'

        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        #self.tree_data_par_path = '../../MIND/MIND/tree_data_SASREC'
        #self.tree_data_par_path = './data/Yambda/tree_data_SASREC'
        #self.tree_data_par_path = './data/Yambda/tree_data_SASREC_hybrid'
        #self.tree_data_par_path = './data/Yambda/tree_data_SASREC_speedup'
        #self.tree_data_par_path = './data/Yambda/tree_data_SASREC_normal'
        self.tree_data_par_path = './data/Yambda/tree_data_SASREC_speedup_sklearn'

        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'      
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'
    
        self.two_tower_item_emb = 'vocab/Yambda_SASREC_item_emb.npy'

        #dataset config
        #self.users_ID_num = 50000
        self.item_ID_num = 475910 + 1  # TODO: Maybe try branching factor 16 for the seater tree as in books
        # experiment const config
        self.reco_his_max_length = 20 

