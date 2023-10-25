from typing import List

import torch
from torch import nn as nn
import torchvision
from einops import repeat
import math


class FC_mlp(nn.Module):
    def __init__(self, layers_dim, num_classes=10, embedding_dim=32, channels=48) -> None:
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(88064, layers_dim[0]))
        layers.append(nn.BatchNorm1d(layers_dim[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
            
        if len(layers_dim) > 1:
            for i in range(len(layers_dim) - 1):
                layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
                layers.append(nn.BatchNorm1d(layers_dim[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout())
        layers.append(nn.Linear(layers_dim[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))
    

class FC_triplane(nn.Module):
    def __init__(self, layers_dim, num_classes=10, embedding_dim=32, channels=48) -> None:
        super().__init__()
        layers = []
        
        layers.append(nn.Conv1d(embedding_dim*embedding_dim, layers_dim[0], 1))
        layers.append(nn.BatchNorm1d(layers_dim[0]))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout())
            
        if len(layers_dim) > 1:
            for i in range(len(layers_dim) - 1):
                # layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
                layers.append(nn.Conv1d(layers_dim[i], layers_dim[i + 1], 1))
                layers.append(nn.BatchNorm1d(layers_dim[i + 1]))
                layers.append(nn.ReLU())
                # layers.append(nn.Dropout())
        self.net = nn.Sequential(*layers)
        self.cls = nn.Linear(layers_dim[-1], num_classes)

    def forward(self, x):
        x = self.net(x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1))
        x, _ = torch.max(x, 2)
        x = self.cls(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, channels=48, **args) -> None:
        super().__init__()
        self.encoder = torchvision.models.resnet50(num_classes=num_classes, pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x
  
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim:int, nhead: int, batch_first=True, num_layers=8, num_classes=10) -> None:
        super().__init__()

        self.projection_layer = nn.Linear(embedding_dim*embedding_dim, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=nhead, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_projection = nn.Linear(projection_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.projection_layer(x)
        x = self.encoder(x)
        x, _ = torch.max(x, 1)
        x = self.final_projection(x)

        return x
    
class Transformer_EncoderDecoder_Seg(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim:int, nhead: int, batch_first=True, num_layers=8, num_classes=10, num_parts=50) -> None:
        super().__init__()

        self.projection_layer = nn.Linear(embedding_dim*embedding_dim, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=nhead, batch_first=batch_first, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.in_layer = nn.Sequential(nn.Linear(63+num_classes, projection_dim), nn.ReLU())        
        decoder_layer = nn.TransformerDecoderLayer(d_model=projection_dim, nhead=nhead//2, batch_first=batch_first, dim_feedforward=1024)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2)        

        self.final_layer = nn.Linear(projection_dim, num_parts)
        
    def forward(self, x: torch.Tensor, class_labels: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.projection_layer(x)
        x = self.encoder(x)
        
        repeated_class_labels = repeat(class_labels, "b d -> b n d", n=coords.shape[1])

        tgt = torch.cat([coords, repeated_class_labels], dim=-1)
        tgt = self.in_layer(tgt)
        tgt = self.transformer_decoder(tgt, x)
        out = self.final_layer(tgt)

        return out 
    
