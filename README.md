# RE-SEATER: Optimizing Hierarchical Identifiers for Generative Sequential Recommendation


## 🧑‍💻 Team Members
- Odysseas Boufalis – odyssefs.boufalis@student.uva.nl
- Polytimi Gkotsi – polytimi.gkotsi@student.uva.nl
- Joshua Rosenthal – joshua.rosenthal@student.uva.nl
- Federica Valeau - federica.valeau@student.uva.nl

## 👥 Supervising TAs
- David Vos (Main Supervisor)
- Owen de Jong (Co-supervisor)


---

## 🧾 Project Abstract
Generative retrieval models like SEATER, proposed by Si et al., enhance sequential recommendation by utilizing balanced tree-structured item identifiers and contrastive objectives, leading to improved inference latency and accuracy. This work first presents a reproducibility study of SEATER, validating its reported performance gains over baseline models across multiple datasets, including a novel one. Our replication confirms SEATER's consistent outperformance and explores its recommendation fairness and diversity, noting generally higher catalog coverage. A critical limitation of the original SEATER, its tree construction time which scales poorly with large item catalogs, is subsequently addressed. We introduce an optimized tree-building algorithm that drastically reduces construction time (e.g., to 5-8\% of the original for large datasets) while maintaining, and in some cases even improving, recommendation accuracy. This study thus not only verifies SEATER's efficacy but also significantly enhances its practical scalability for large-scale recommendation systems.

---

## 📊 Summary of Results


### Reproducability 
- **Absolute Performance**
    - Reproduced SEATER results closely match original values on Books and Yelp (e.g., Books NDCG@50: 0.0714 vs. 0.0713).
    - Slight improvements were observed on News (e.g., SEATER NDCG@50: 0.1251 vs. original 0.1225).
- **Relative Trends**
    - Model ranking and SEATER’s superiority are preserved on Books and Yelp.
    - On News, SEATER still leads but with a narrower margin compared to the original, likely due to differences in baseline implementations or preprocessing.
    - GRU4Rec's performance on News is much closer to SEATER than originally reported (Recall@50: 0.2541 vs. 0.2581).
- **Generalization to New Dataset (Yambda)**
    - SEATER achieves top performance across all metrics on Yambda.
    - SASRec is consistently second-best, reaffirming SEATER’s robustness and generalization capabilities.
- **Conclusion**
    - The core claims of the original SEATER paper are reproduced.
    - However, the performance gap varies by dataset, suggesting sensitivity to preprocessing and implementation choices.

### Extensions

📊 **Fairness & Diversity Evaluation**
- Motivation: To evaluate whether SEATER’s recommendations mitigate popularity bias and promote catalog diversity beyond just accuracy.
- Metrics Introduced:
    - Coverage@k ↑: Measures how many unique items are recommended across users.
    - Gini Index@k ↓: Measures inequality in category-level exposure (lower = fairer).
- Key Findings:
    - SEATER consistently achieved the highest Coverage@k across datasets (Books, Yelp, Yambda), indicating better exploration of the item space.
    - Gini scores were similar across models, with no clear winner, likely due to overlapping item categories in datasets.
    - SEATER leads in diversity, while fairness (via Gini) remains dataset-dependent and less conclusive.
 
🌲 **Efficient Tree Building**
- Problem: SEATER’s original tree-building method (constrained k-means at each level) scales poorly with the number of items (complexity: $O((N^3k + N^2k^2 + Nk^3)\log(N + k))$)
- Proposed Solutions:
    - Greedy ("Speedup") Method: Faster assignment of items using standard k-means + capacity-constrained nearest assignment.
    - Hybrid Method: Greedy method for top levels, constrained k-means for lower levels (cutoff at 2000 items).
- Results:
    - Hybrid method reduced tree construction time to 5–8% of original with <0.6% drop in accuracy.
    - On Yambda, Hybrid and Speedup even outperformed the original SEATER, suggesting exact trees are not always optimal.
    - SEATER is robust to tree noise, likely due to its alignment and ranking losses compensating for imprecise tree structure.
 
🎯 **Popularity Normalization**
- Goal: Reduce bias toward popular items during tree construction.
- Strategies:
    - Diversity-focused: Inverse popularity weights.
    - Accuracy-focused: Normalized frequency weights.
- Findings:
    - No consistent accuracy gains, and in some cases, slight losses in ranking performance.
    - Fairness improvements observed, especially in Gini Index and Coverage for certain datasets (e.g., Yelp).
    - Effectiveness is dataset-specific; gains in fairness may come at a small cost to relevance.

---

## 🛠️ Task Definition
We address the generative sequential recommendation task, where the goal is to predict the next item a user will interact with based on their historical sequence of interactions. Given a user’s interaction history $\mathbf X=[x_1,x_2,\dots,x_n]$, the model generates a structured item identifier $\mathbf Y=[y_1,y_2,\dots,y_m]$ that corresponds to the next recommended item.

Unlike traditional retrieve-then-rank systems, we follow a generative retrieval paradigm where item identifiers are decoded token-by-token using an autoregressive Transformer. These identifiers are structured as paths in a balanced kk-ary tree, encoding semantic item relationships. The model jointly optimizes for generation accuracy, semantic alignment within the tree, and item ranking, thereby producing recommendations that are both relevant and diverse.

The input to the model is a sequence of user interactions (e.g., items viewed or clicked), and the output is a structured token sequence representing the next item recommendation, constrained to valid paths in the identifier tree.

---

## 📂 Datasets

Following the original implementation, datasets are partitioned by user into 8 : 1 : 1 train : validation : test splits. User interaction histories are truncated to a maximum of 20 items. For validation and test, we use a sequential protocol: the first 80% of a user's history serves as input context, and the remaining 20% as prediction targets.

- [ ] [Amazon Books](https://nijianmo.github.io/amazon/)
  - [ ] Pre-processing: Used [ComiRec]() logic.
  - [ ] Dataset size: 482,934 users, 361,245 items
  - [ ] Attributes for item fairness: Book Categories for Gini Index
  

- [ ] [Yelp](https://business.yelp.com/data/resources/open-dataset/)
  - [ ] Pre-processing: Used [LightGCN]() repository splits.
  - [ ] Dataset size: 31,668 users, 38,048 items
  - [ ] Attributes for item fairness: Business Categories for Gini Index

- [ ] [MIND](https://msnews.github.io/)
  - [ ] Pre-processing: MIND small; Used only the training portion of the dataset.
  - [ ] Dataset size: 50,000 users, 39,815 items
  - [ ] Attributes for item fairness: News categories for Gini Index
     
- [ ] [Yambda](https://huggingface.co/datasets/yandex/yambda)
  - [ ] Pre-processing: Kept only organic (non-recommended) interactions; Removed tracks played for <50% duration; Global temporal split: 299 days train / 1 day val / 30 min gap / 1 day test.
  - [ ] Dataset size: 8,798 users, 475,910 items
  - [ ] Genre-based or content-based categories for Gini Index
---

## 📏 Metrics

### Accuracy Metrics

- [ ] **Recall@k**  
  - [ ] Measures the proportion of relevant items successfully retrieved in the top-k recommendations. Indicates how well the model captures relevant content for downstream ranking.

- [ ] **Hit Rate (HR@k)**  
  - [ ] A binary indicator of success: 1 if at least one relevant item is present in the top-k list, 0 otherwise. Reflects the likelihood of recommending at least one correct item.

- [ ] **NDCG@k (Normalized Discounted Cumulative Gain)**  
  - [ ] A rank-aware metric that rewards placing relevant items higher in the list. Normalized against the ideal ranking to ensure comparability across users.

### Diversity & Fairness Metrics

- [ ] **Coverage@k**  
  - [ ] Measures the proportion of unique items in the catalog that appear in at least one user’s top-k recommendations. Higher values indicate greater aggregate diversity.

- [ ] **Gini Index@k**  
  - [ ] Measures the inequality of exposure across item categories in the top-k recommendations. A value of 0 implies perfect fairness (equal exposure), while 1 indicates maximum disparity.

---

## 🔬 Baselines & Methods

_Describe each baseline, primary methods, and how they are implemented. Mention tools/frameworks used (e.g., Surprise, LightFM, RecBole, PyTorch)._
Describe each baseline
- [ ] [GRU4Rec](https://arxiv.org/abs/1511.06939)
  - [ ] A foundational session-based model using Gated Recurrent Units (GRUs) to encode sequential user interactions.
  - [ ] Implemented based on the [ComiRec repository](https://github.com/THUDM/ComiRec).
  - [ ] User representation is the final hidden state of the GRU.
- [ ] [BERT4Rec](https://dl.acm.org/doi/10.1145/3357384.3357895)
  - [ ] A bidirectional Transformer-based model for sequential recommendation using a Cloze-style task (random token masking).
  - [ ] Inference uses a _MASK_ token to predict the next item.
  - [ ] Implementation adapted from [this PyTorch repo](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)
- [ ] [SASRec](https://ieeexplore.ieee.org/document/8594844)
  - [ ] A unidirectional self-attention model for autoregressive sequence modeling in recommendation.
  - [ ] User representation is derived from the final attended item embedding.
  - [ ] Also used to pretrain item embeddings for SEATER's tree construction.
  - [ ] Based on the original [SASRec implementation](https://github.com/kang205/SASRec).
- [ ] [SEATER](https://dl.acm.org/doi/10.1145/3673791.3698408)
  - [ ] A generative retrieval model using semantic tree-structured identifiers and a contrastive multi-task training objective.
  - [ ] Uses a Transformer decoder to autoregressively generate hierarchical item identifiers.
  - [ ] Implemented in [PyTorch](https://dl.acm.org/doi/10.5555/3454287.3455008) using [authors’ original codebase](https://github.com/ethan00si/seater_generative_retrieval) and extended in this repo.
     
### 🧠 High-Level Description of Method

Our approach builds on SEATER, a generative model for sequential recommendation that predicts the next item a user will interact with — but instead of selecting items from a flat list, it generates structured item identifiers that reflect semantic relationships.

**🔄 Model Pipeline**

1. Data Input
  - Users' interaction histories (e.g., items clicked or listened to) are represented as sequences. These sequences are capped at a maximum length (e.g., 20).

2. Embedding / Representation
  - Each item is represented by an embedding. Item embeddings are learned using SASRec, a Transformer-based sequential model. These embeddings are then used to build a balanced k-ary tree, where each item is assigned a path of tokens (i.e., a hierarchical identifier).

3. Sequence Encoding
  - A Transformer encoder processes the input sequence (user history) to produce a summary representation of user intent.

4. Prediction (Identifier Generation)
  - A Transformer decoder autoregressively generates the next item’s identifier token-by-token. At each decoding step, the model is constrained to valid paths in the identifier tree using constrained beam search, which ensures efficient and semantically valid decoding.

5. Ranking
  - The generated identifier corresponds to an item in the catalog, which is returned as the top recommendation. Multiple items can be decoded and ranked by generation probability.

🧩 **Design Highlights**

- Hierarchical Identifiers: Items are encoded as paths in a tree built using constrained k-means on semantic embeddings. This structure makes generation more efficient and aligns with item similarity.

- Multi-task Training: The model is trained with three loss functions:
  - Generation loss (predict correct token sequence)
  - Alignment loss (ensure tree semantics are respected)
  - Ranking loss (push similar items closer in the hierarchy)


---

## 🌱 Proposed Extensions

We extend the original SEATER framework in both methodology and evaluation. Our contributions are as follows:

- [ ] Efficient Tree Construction
  - [ ] Introduced two alternative tree-building methods to reduce the computational cost of SEATER’s constrained k-means clustering:
  - [ ] Greedy (Speedup): Uses standard k-means followed by sequential assignment to nearest available cluster.
  - [ ] Hybrid: Combines greedy assignment for upper tree levels with constrained k-means for deeper levels.
  - [ ] These methods reduce tree-building time by up to 95% with minimal (or even improved) accuracy changes.

- [ ] Fairness & Diversity Evaluation
  - [ ] Extended the evaluation by incorporating Coverage@k and Gini Index@k to assess:
  - [ ] How many unique items are recommended across users.
  - [ ] How equitably item categories are exposed in the recommendation lists.
  - [ ] This helps reveal whether the model mitigates popularity bias.

- [ ] New Dataset Evaluation (Yambda)
  - [ ] Evaluated SEATER on the Yambda music dataset from Yandex, featuring large item catalogs and a different domain (music streaming).
  - [ ] Results confirm SEATER’s strong generalization and robustness across domains.

- [ ] Popularity-Based Normalization
  - [ ] Investigated the use of popularity-aware sampling during tree construction to mitigate over-representation of popular items:
  - [ ] Tested both diversity-oriented (inverse popularity) and accuracy-oriented (normalized popularity) weighting.
  - [ ] Found mixed results—diversity often improved, but accuracy gains were inconsistent across datasets.



