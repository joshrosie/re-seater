import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

class ILD:
    """
    Intra-list diversity (ILD) measures the diversity of a recommendation list by calculating the average dissimilarity between all pairs of items in the list.
    """
    def calculate_ild(self, recommendation_articles, representation='tfidf'): 
        """
        Calculate the intra-list diversity (ILD) between candidate articles and recommendation articles.

        Parameters:
        - recommendation_articles: List of recommended articles.

        Returns:
        - ILD value.
        """
        # Handle sparse matrices properly
        vecs = []
        for article in recommendation_articles:
            if representation == 'tfidf':
                vecs.append(article['tfidf_vector'])
            elif representation == 'st':
                vecs.append(article['st_vector'])
        
        if len(vecs) < 2:
            return 0.0  # no diversity if only 0 or 1 item
        
        # For sparse matrices, use scipy's vstack
        if representation == 'tfidf':
            X = vstack(vecs)
        else:
            X = np.array(vecs)

        # Compute cosine-similarity matrix (works with sparse matrices)
        sim_matrix = cosine_similarity(X)  # shape (n, n)
        
        # Extract the upper triangle (excluding diagonal)
        n = sim_matrix.shape[0]
        # Indices i<j
        iu = np.triu_indices(n, k=1)
        sims = sim_matrix[iu]
        
        # Compute dissimilarities and average
        dissimilarities = 1.0 - sims
        avg_diversity = dissimilarities.mean()
        
        return avg_diversity
    
