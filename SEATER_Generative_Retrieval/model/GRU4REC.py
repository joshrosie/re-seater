"""gru4rec_pytorch.py – GRU4Rec rewritten so it can be plugged into the **same
SASRec data‑loading pipeline** (item_feat wrapper, three‑tuple dataloader, and
`sampled_softmax` loss helper).

The public API mirrors SASREC’s:

* **parse_input_train / parse_input_test** – read the tuple yielded by the
  SASRec dataloader and produce ready‑to‑consume tensors.
* **log2feats** – turn a *sequence of item embeddings* into a *user (session)
  embedding*, here via a single‑layer GRU instead of self‑attention.
* **forward(input_data)** – returns a `sampled_softmax` loss identical to the
  one used by SASRec, so training loops don’t have to change.
* **predict(input_data, topK)** – full‑item ranking for evaluation.

Any code that already trains SASRec can now swap the model construction line
for `model = GRU4Rec(config)` and the rest will work unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .inputs import item_feat_no_add_feat as item_feat  # same helper SASRec uses
from .module_utils import sampled_softmax               # identical loss helper


class GRU4REC(nn.Module):
    """GRU4Rec rebuilt around the **SASRec I/O contract**.

    Args
    ----
    config : dict
        Same dict you pass to SASRec. Must contain at least:
        ``num_neg``, ``max_reco_his`` (max sequence length), and
        ``dropout_rate``. Extra keys are silently ignored.
    """

    def __init__(self, config: dict):
        super().__init__()

        # ------------------------------------------------------------------
        # Hyper‑params & helpers from the SASRec eco‑system
        # ------------------------------------------------------------------
        self.num_neg: int = config["num_neg"]
        self.item_feat = item_feat(config)                  # shared embedder
        hidden_dim: int = self.item_feat.size              # embedding size
        self.item_num: int = self.item_feat.item_num       # catalogue size

        # ------------------------------------------------------------------
        # Sequence encoder: one‑layer GRU → final hidden state is user emb
        # ------------------------------------------------------------------
        self.gru = nn.GRU(input_size=hidden_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)
        self.dropout = nn.Dropout(p=config.get("dropout_rate", 0.0))

        # ------------------------------------------------------------------
        # Sigmoid is used only for *predict* – same as SASRec implementation.
        # ------------------------------------------------------------------
        self.sigmoid = nn.Sigmoid()

        # Init – follow SASRec practice for fair comparison
        self._init_weights()

    # ----------------------------------------------------------------------
    # Initialisation identical to SASRec’s for apples‑to‑apples results.
    # ----------------------------------------------------------------------
    def _init_weights(self): # TODO: might remove
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    # ----------------------------------------------------------------------
    # Utilities to align with SASRec’s dataset/dataloader contract
    # ----------------------------------------------------------------------
    def _neg_sample_per_batch(self, device: torch.device) -> torch.LongTensor:
        """Uniform *unique* negatives, exactly like SASRec’s helper."""
        choices = np.arange(1, self.item_num)  # item IDs start from 1
        ids = np.random.choice(choices, size=self.num_neg, replace=False)
        return torch.tensor(ids, dtype=torch.long, device=device)

    # ----------------- I/O helpers (same names as SASRec) ------------------
    def parse_input_test(self, input_data):
        """Input: ``(user, rec_his)`` as yielded by SASRec’s test loader."""
        _, rec_his = input_data                                      # ignore user id
        rec_his_emb = self.item_feat(rec_his)                        # [B, T, D]
        # rec_his_mask = (rec_his == 0).bool()   
        rec_his_mask = torch.where( # FIXME: this might break
                            rec_his==0,
                            1, 0).bool()# 1 on PAD
        return rec_his_emb, rec_his_mask

    def parse_input_train(self, input_data):
        """Input: ``(user, rec_his, pos_item)`` from SASRec train loader."""
        _, rec_his, pos_item = input_data                            # unpack

        device = pos_item.device
        neg_item = self._neg_sample_per_batch(device)                # [K]
        neg_item = neg_item.unsqueeze(0).expand(pos_item.size(0), -1)  # [B, K]

        pos_item_emb = self.item_feat(pos_item)                      # [B, D]
        neg_item_emb = self.item_feat(neg_item)                      # [B, K, D]

        rec_his_emb = self.item_feat(rec_his)                        # [B, T, D]
        rec_his_mask = torch.where( # FIXME: this might break
                            rec_his==0,
                            1, 0).bool()                      # [B, T]

        return pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask

    # ----------------------------------------------------------------------
    # Core: sequence → final hidden state (user / session embedding)
    # ----------------------------------------------------------------------
    def log2feats(self, seqs: torch.Tensor, log_seqs_mask: torch.Tensor) -> torch.Tensor:
        """Turn the item‑embedding sequence into a fixed‑length user vector.

        * **seqs** – FloatTensor `[B, T, D]` – embedded history.
        * **log_seqs_mask** – BoolTensor `[B, T]` – *True* on **PAD** tokens.
        """
        seqs = self.dropout(seqs)

        # Compute real lengths so the GRU ignores paddings
        lengths = (~log_seqs_mask).sum(dim=1).clamp(min=1).cpu()  # avoid 0
        packed = pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                                 # [1, B, D]
        log_feats = h_n.squeeze(0)                                # [B, D]
        return log_feats

    # ----------------------------------------------------------------------
    # Public API ------------------------------------------------------------
    # ----------------------------------------------------------------------
    def forward(self, input_data):
        """Compute sampled‑softmax loss – **identical call‑site as SASRec**."""
        pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask = self.parse_input_train(input_data)
        log_feats = self.log2feats(rec_his_emb, rec_his_mask)      # [B, D]
        loss = sampled_softmax(log_feats, pos_item_emb, neg_item_emb)
        return loss

    @torch.no_grad()
    def predict(self, input_data, topK: int = 50):
        """Full‑catalogue ranking (same signature as SASRec.predict)."""
        rec_his_emb, rec_his_mask = self.parse_input_test(input_data)
        user_vec = self.log2feats(rec_his_emb, rec_his_mask)       # [B, D]

        # Pre‑compute all item embeddings once per predict() call
        all_item_ids = torch.arange(1, self.item_num, device=user_vec.device)  # exclude PAD index 0
        all_item_emb = self.item_feat(all_item_ids)                 # [N, D]

        logits = self.sigmoid(torch.matmul(user_vec, all_item_emb.t()))  # [B, N]
        logit_values, logit_indices = torch.topk(logits, k=topK, dim=1)
        logit_indices += 1  # shift back to original item IDs (start at 1)
        return logit_values, logit_indices

    # Convenience – same alias SASRec exposes --------------------------------
    # def log2feats_and_items(self, input_data):
    #     """Utility sometimes used in downstream scripts."""
    #     rec_his_emb, rec_his_mask = self.parse_input_test(input_data)
    #     return self.log2feats(rec_his_emb, rec_his_mask)
