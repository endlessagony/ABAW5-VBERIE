import torch
import torch.nn as nn

from   models.trans import *
from   models.TCN import *


class MModel(nn.Module):
    def __init__(self, modalities_features: dict = {'visual': 1290, 'acoustic': 768, 'AUs': 34}, d_model: int = 128):
        super(MModel, self).__init__()
        self.modal_num = 2
        
        self.input = nn.ModuleDict()
        for modality_type, num_features in modalities_features.items():
            if modality_type == 'visual':
                self.input[modality_type] = nn.Sequential(
                        nn.LayerNorm(num_features + modalities_features['AUs']),
                        PositionalEncoding(d_model=num_features + modalities_features['AUs'], dropout=0.5),
                        ModalEncoder(dim=num_features + modalities_features['AUs'], dropout=0.5),
                        nn.Linear(in_features=(num_features + modalities_features['AUs']) * 3, out_features=d_model)
                    )
                
            elif modality_type == 'acoustic':
                self.input[modality_type] = nn.Sequential(
                        nn.LayerNorm(num_features),
                        TemporalConvNet(num_inputs=num_features, num_channels=[d_model]),
                        ModalEncoder(dim=d_model, dropout=0.4),
                        nn.Linear(in_features=d_model * 3, out_features=d_model * 2),
                        nn.Dropout(p=0.1),
                        nn.ReLU(),
                        nn.Linear(in_features=d_model * 2, out_features=d_model)
                    )
                
        self.dropout_embed = nn.Dropout(p=0.4)
        multimodal_encoder_layer = nn.ModuleList()
        for i in range(8):
            mm_attention = MultiModalAttention(
                h=4, d_model=d_model, modal_num=self.modal_num, dropout=0.5)
            
            mt_attention, feed_forward = nn.ModuleList(), nn.ModuleList()
            for j in range(self.modal_num):
                mt_attention.append(MultiHeadedAttention(
                    h=4, d_model=d_model, dropout=0.5))
                feed_forward.append(PositionwiseFeedForward(
                    d_model=d_model, d_ff=512, dropout=0.4))
            
            multimodal_encoder_layer.append(MultiModalEncoderLayer(
                size=d_model, modal_num=self.modal_num, mm_atten=mm_attention, mt_atten=mt_attention,
                feed_forward=feed_forward, dropout=0.5
            ))
        self.encoder = MultiModalEncoder(layer=multimodal_encoder_layer, N=8, modal_num=self.modal_num)
        
        self.regress = nn.Sequential(
            nn.Linear(in_features=d_model * 4, out_features=d_model * 4 // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model * 4 // 2),
            nn.Linear(in_features=d_model * 4 // 2, out_features=7)
        )
        
        self.final_activation = nn.Sigmoid()
        
    def forward(self, v_input: torch.Tensor=None, a_input: torch.Tensor=None, au_input: torch.Tensor=None):
        batch_size, _, _ = v_input.shape
        input_dict = {'visual': torch.concat((v_input, au_input[:, :, :-1]), dim=-1), 'acoustic': a_input}
        
        _x_list = []
        for modality_type, input_tensor in input_dict.items():
            _x_list.append(self.input[modality_type](input_tensor))
            
        x = torch.cat(_x_list, dim=-1)
        x = self.dropout_embed(x)
        
        out = self.encoder(torch.cat(_x_list, dim=-1), mask=None)
        outs = self.final_activation(self.regress(torch.cat((out, x), dim=-1)))
        return outs.mean(dim=1)