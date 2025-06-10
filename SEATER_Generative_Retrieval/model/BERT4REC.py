import torch
import torch.nn as nn
import numpy as np

from .bert_modules.bert import BERT
from .module_utils import sampled_softmax

from types import SimpleNamespace


class BERT4REC(nn.Module):
    def __init__(self, config, random_seed=None):
        super().__init__()

        self.item_num = config['item_num'] + 2  # +2 for [PAD]=0 and [MASK]=num_items+1
        self.num_neg = config['num_neg']

        args = SimpleNamespace(
            model_init_seed=random_seed,
            bert_max_len=config['max_reco_his']+1,  # +1 for [MASK] token
            num_items=config['item_num'],
            bert_num_blocks=config['num_blocks'],
            bert_num_heads=config['num_heads'],
            bert_hidden_units=config['embedding_dim'],
            bert_dropout=config['dropout'],
        )

        print(f'BERT4REC arguments: {args}')

        self.bert = BERT(args)
        self.token_embedding = self.bert.embedding.token  # [vocab_size, H]

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, input_data):
        """
        Training forward pass with sampled softmax.
        input_data: (tokens, labels)
            - tokens: [B, L]  — input sequence (with [MASK] tokens)
            - labels: [B, L] — ground truth item ids at masked positions (else 0)
        """
        _, tokens, labels = input_data
        hidden_states = self.bert(tokens)  # [B, L, H]

        mask_pos = labels > 0              # [B, L] → bool mask
        masked_hidden = hidden_states[mask_pos]  # [N, H]
        masked_labels = labels[mask_pos]         # [N]

        if masked_hidden.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=hidden_states.device)

        item_embeddings = self.token_embedding.weight  # [item_num, H]
        pos_emb = item_embeddings[masked_labels]       # [N, H]
        neg_ids = self._neg_sample(masked_hidden.size(0), device=tokens.device)
        neg_emb = item_embeddings[neg_ids]             # [N, num_neg, H]

        loss = sampled_softmax(masked_hidden, pos_emb, neg_emb)
        return loss

    def predict(self, input_data, topK=50):
        """
        Predict top-K items at MASK positions during inference.
        input_seq: [B, L] with one or more [MASK] tokens.
        """
        user, input_seq = input_data  # input_seq: [B, L]
        with torch.no_grad():
            hidden_states = self.bert(input_seq)         # [B, L, H]
            mask_pos = input_seq == (self.item_num - 1)  # [MASK] token = last ID

            masked_hidden = hidden_states[mask_pos]      # [N, H]
            item_emb = self.token_embedding.weight       # [item_num, H]

            logits = torch.matmul(masked_hidden, item_emb.T)  # [N, item_num]
            scores, indices = torch.topk(logits, topK, dim=-1)
            return scores, indices

    def _neg_sample(self, N, device):
        """
        Uniformly sample negative item ids (excluding [PAD]=0 and [MASK])
        Returns: [N, num_neg]
        """
        candidates = np.arange(1, self.item_num - 1)
        sampled = np.random.choice(candidates, size=(N, self.num_neg), replace=True)
        return torch.tensor(sampled, dtype=torch.long, device=device)
