import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model_r import CaptioningModelR
import pdb


class TransformerR(CaptioningModelR):
    def __init__(self, bos_idx, encoder, decoder):
        super(TransformerR, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, bboxes, rel_pairs, rel_labels, *args):
        enc_output, mask_enc, rel_dists, rel_labels_ = self.encoder(images, bboxes, rel_pairs, rel_labels)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output, rel_dists, rel_labels_

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, bboxes=None, rel_pairs=None, rel_labels=None, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc, rel_dists, rel_labels_ = self.encoder(visual, bboxes, rel_pairs, rel_labels)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsembleR(CaptioningModelR):
    def __init__(self, model: TransformerR, weight_files):
        super(TransformerEnsembleR, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, bboxes=None, rel_pairs=None, rel_labels=None, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, bboxes, rel_pairs, rel_labels, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
