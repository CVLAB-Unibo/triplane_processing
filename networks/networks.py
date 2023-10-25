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
        # self.class_token = torch.nn.Parameter(
        #     torch.randn(1, 1, embedding_dim)
        # )
        # torch.nn.init.normal_(self.class_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = torch.cat([self.class_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.projection_layer(x)
        x = self.encoder(x)
        x, _ = torch.max(x, 1)
        # x = x[:, 0, :]
        x = self.final_projection(x)

        return x
    
class Transformer_EncoderDecoder_Cls(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim:int, nhead: int, batch_first=True, num_layers=8, num_classes=10) -> None:
        super().__init__()

        self.projection_layer = nn.Linear(embedding_dim*embedding_dim, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=nhead, batch_first=batch_first, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.in_layer = nn.Linear(48, projection_dim)        
        decoder_layer = nn.TransformerDecoderLayer(d_model=projection_dim, nhead=nhead//2, batch_first=batch_first, dim_feedforward=1024)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2)        

        # self.class_token = torch.nn.Parameter(
        #     torch.randn(1, num_classes, projection_dim)
        # )
        # torch.nn.init.normal_(self.class_token, std=0.02)

        self.final_layer = nn.Linear(projection_dim, num_classes)
        
        # self.class_token = nn.Embedding(num_classes, projection_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(x.shape[0], x.shape[1], -1)
        y = self.projection_layer(y)
        y = self.encoder(y)
        
        # repeated_class_tokens = torch.stack([self.class_token[0].clone() for i in range(x.shape[0]) ])
        tgt = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        tgt = self.in_layer(tgt)
        
        # tgt = self.class_token.weight.repeat([x.shape[0], 1, 1], requires_grad=tue)
        tgt = self.transformer_decoder(tgt, y)
        out, _ = torch.max(tgt, 1)
        out = self.final_layer(out)

        return out
    
    
class TransformerSeg(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim:int, nhead: int, batch_first=True, num_layers=8, num_classes=10, num_parts=50) -> None:
        super().__init__()

        self.projection_layer = nn.Linear(embedding_dim*embedding_dim, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=nhead, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.in_layer = nn.Sequential(nn.Linear(projection_dim+63+num_classes, projection_dim), nn.ReLU())
        self.skip_proj = nn.Sequential(nn.Linear(projection_dim+63+num_classes, projection_dim), nn.ReLU())
        before_skip = []
        for _ in range(2):
            before_skip.append(nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.ReLU()))
        self.before_skip = nn.Sequential(*before_skip)

        after_skip = []
        for _ in range(2):
            after_skip.append(nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.ReLU()))
        after_skip.append(nn.Linear(projection_dim, num_parts))
        self.after_skip = nn.Sequential(*after_skip)
        

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.projection_layer(x)
        x = self.encoder(x)
        x, _ = torch.max(x, 1)
        x = torch.cat([x, class_labels], dim=-1)
        
        repeated_embeddings = repeat(x, "b d -> b n d", n=coords.shape[1])
        emb_and_coords = torch.cat([repeated_embeddings, coords], dim=-1)
        
        x = self.in_layer(emb_and_coords)
        x = self.before_skip(x)

        inp_proj = self.skip_proj(emb_and_coords)
        x = x + inp_proj

        x = self.after_skip(x)

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
    
