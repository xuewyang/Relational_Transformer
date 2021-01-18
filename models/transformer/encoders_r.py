from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
import pdb


class EncoderLayerR(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayerR, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoderR(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoderR, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_rel_cls = 51
        self.layers = nn.ModuleList([EncoderLayerR(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering,
                                                   attention_module=attention_module,
                                                   attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.hidden_dim = d_model
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.post_union = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.rel_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

    def forward(self, input, bboxes=None, rel_pairs=None, rel_labels=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))
        outs = torch.cat(outs, 1)
        if bboxes is not None:
            edge_rep = self.post_emb(out)
            edge_rep = edge_rep.view(edge_rep.size(0), 2, -1, self.hidden_dim)
            head_rep = edge_rep[:, 0]
            tail_rep = edge_rep[:, 1]

            inputs, head_reps, tail_reps, rel_pair_idxs, rel_labels_, num_objects, num_rels = \
                self.prepare_relation_features(input, head_rep, tail_rep, bboxes, rel_pairs, rel_labels)
            prod_reps = []
            union_features = []
            for pair_idx, head_rep_, tail_rep_, input_ in zip(rel_pair_idxs, head_reps, tail_reps, inputs):
                prod_reps.append(torch.cat((head_rep_[pair_idx[:, 0]], tail_rep_[pair_idx[:, 1]]), dim=-1))
                union_features.append(torch.cat((input_[pair_idx[:, 0]], input_[pair_idx[:, 1]]), dim=-1))
            prod_reps = torch.cat(prod_reps, dim=0)
            union_features = torch.cat(union_features, dim=0)
            ctx_gate = self.post_cat(prod_reps)
            union_features = self.post_union(union_features)
            visual_rep = ctx_gate * union_features
            rel_dists = self.rel_compress(visual_rep)
            rel_dists = rel_dists.split(num_rels, dim=0)
            rel_dists = torch.cat(rel_dists, dim=0)
            rel_labels_ = torch.cat(rel_labels_, dim=0)

            return outs, attention_mask, rel_dists, rel_labels_     # [bs, 3, seq_len, 512], [bs, 1, 1, seq_len]
        else:
            return outs, attention_mask, None, None

    def prepare_relation_features(self, input, head_rep, tail_rep, bboxes, rel_pairs, rel_labels):
        # get number of objects in each sample of the batch
        num_objects = []
        num_rels = []
        rel_pair_idxs = []
        rel_labels_ = []
        head_reps = []
        tail_reps = []
        input_ = []
        for i in range(len(bboxes)):
            y2 = bboxes[i][:, -1]
            num = len(y2[y2 > -1])
            # print(num)
            num_objects.append(num)
            num_rels.append(num * (num-1))
            rel_pair_idxs.append(rel_pairs[i][:num * (num-1)])
            rel_labels_.append(rel_labels[i][:num * (num-1)])
            head_reps.append(head_rep[i, :num])
            tail_reps.append(tail_rep[i, :num])
            input_.append(input[i, :num])

        return input_, head_reps, tail_reps, rel_pair_idxs, rel_labels_, num_objects, num_rels


class MemoryAugmentedEncoderR(MultiLevelEncoderR):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoderR, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, bboxes, rel_pairs, rel_labels, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(MemoryAugmentedEncoderR, self).forward(out, bboxes, rel_pairs, rel_labels, attention_weights=attention_weights)
