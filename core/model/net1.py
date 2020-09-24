# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED, MHAtt, FFN

import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import grad_mul_const

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,       # 512
            mid_size=__C.FLAT_MLP_SIZE,    # 512
            out_size=__C.FLAT_GLIMPSES,    # 1
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,    # 512 * 1
            __C.FLAT_OUT_SIZE                       # 1024
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE    # 300
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,    # Faster-rcnn 2048D features
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm_img = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm_lang = LayerNorm(__C.FLAT_OUT_SIZE)
        
        self.proj_img = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.proj_lang = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        q_embed, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            q_embed,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Linear multimodal fusion function
        img_feat = self.proj_norm_img(img_feat)    # Layer Normalization
        lang_feat = self.proj_norm_lang(lang_feat)
        
        proj_feat_img = self.proj_img(img_feat)
        proj_feat_lang = self.proj_lang(lang_feat)
        
        proj_feat = torch.sigmoid(proj_feat_img + proj_feat_lang)

        return proj_feat, proj_feat_img, proj_feat_lang


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


def Diff_loss(lang_out, img_out, pred, target, diff = 0.1):
    q_pred = torch.argmax(lang_out.long(),dim=1)
    #img_pred = torch.argmax(img_out.long(),dim=1)
    loss = 0
    for i in range(64):
        if (pred[i] == target[i]):
            p = img_out[i, pred[i]]/(img_out[i, pred[i]] + lang_out[i, pred[i]])
            if ((2*p - 1) >= diff):
                loss = 0
            else:
                loss = (2*p - 1 - diff)**2
        else:
            #q_pred = torch.argmax(lang_out.long(),dim=1)
            if (q_pred[i] != target[i]):
                #print(lang_out[i].size())
                loss = (2*F.softmax(lang_out[i])[q_pred[i]])**2
    return loss

'''
class Diff_loss(torch.autograd.Function):
    def forward(ctx, lang_out, img_out, pred, target):
        ctx.save_for_backward(ctx, lang_out, img_out, pred, target)
        diff = 0.1
        q_pred = torch.argmax(lang_out.long(),dim=1)
        #img_pred = torch.argmax(img_out.long(),dim=1)
        loss = 0
        for i in range(64):
            if (pred[i] == target[i]):
                p = img_out[i, pred[i]]/(img_out[i, pred[i]] + lang_out[i, pred[i]])
                if ((2*p - 1) >= diff):
                    loss = 0
                else:
                    loss = (2*p - 1 - diff)**2
            else:
                #q_pred = torch.argmax(lang_out.long(),dim=1)
                if (q_pred[i] != target[i]):
                    #print(lang_out[i].size())
                    loss = F.softmax(lang_out[i], dim=0)[q_pred[i]]**2
        return loss
    
    def backward(ctx, grad_output):
        #lang_b, img_b, pred_b, target_b, diff_b = ctx.saved_tensors
        lang_out, img_out, pred, target = ctx.saved_tensors
        diff = 0.1
        grad_input = grad_output.clone()
        
        q_pred = torch.argmax(lang_out.long(),dim=1)
        #img_pred = torch.argmax(img_out.long(),dim=1)
        for i in range(64):
            if (pred[i] == target[i]):
                p = img_out[i, pred[i]]/(img_out[i, pred[i]] + lang_out[i, pred[i]])
                if ((2*p - 1) >= diff):
                    grad_input = 0
                else:
                    grad_input = 4*(1 + diff - 2*p)
            else:
                #q_pred = torch.argmax(lang_out.long(),dim=1)
                if (q_pred[i] != target[i]):
                    #print(lang_out[i].size())
                    grad_input = 2*F.softmax(lang_out[i], dim=0)[q_pred[i]]
                else:
                    grad_input = 0
        return grad_input
        '''