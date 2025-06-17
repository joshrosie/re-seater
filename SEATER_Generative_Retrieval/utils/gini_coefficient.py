import numpy as np

from collections import defaultdict

class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n+1)
        gini = (np.sum((2 * index - n - 1) * arr) ) / (n * n * mu)
        # print(f"Gini coefficient calculated: {gini}")
        return gini

    def calculate_list_gini(self, articles, id2cat,key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        # id2cat = data_utils.load_tsv_file("../../Books/dataset/", "id2cat", sep='\t', engine="pyarrow")
       
        category_scores = defaultdict(float)
        for article_id in articles:
            categories = id2cat.get(str(article_id), [])
            len_categories = len(categories)
            if len_categories == 0:
                # print(f"Article ID {article_id} has no categories, assigning to 'UNK'")
                category_scores["UNK"] += 1
            else:
                # print(f"Article ID {article_id} categories: {categories}")
                for category in categories:
                    category_scores[category] += 1
        return self.gini_coefficient(list(category_scores.values()))