import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer





class DecisionModelV1(nn.Module):
    def __init__(self):
        super(DecisionModelV1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    

class DecisionModelV1_re(nn.Module):
    def __init__(self):
        super(DecisionModelV1_re, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.tanh(x)

        return self.layers(x)

class DecisionModelV2(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # I doubt the residual connection is not being used 
        
        out = self.layer1(x)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        return self.output_layer(out) 
    

class DecisionModelV2l(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV2l, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512,bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            # nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256,bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            # nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256,bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            # nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # I doubt the residual connection is not being used 
        
        out = self.layer1(x)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        return self.output_layer(out) 

class DecisionModelV5(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV5, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, input_dim, kernel_size=138, padding=4,stride=70,bias=True),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim*10, 1024,bias=True),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512,bias=True),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 1,bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        # print(f"shape of out {out.shape}")
        out = out.view(out.shape[0],-1)
        out = self.layer2(out)
        out = self.layer3(out) 
        return self.output_layer(out)

class DecisionModelV5_b(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV5_b, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, input_dim, kernel_size=138, padding=4,stride=70,bias=True),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim*10, 1024,bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512,bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 1,bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        # print(f"shape of out {out.shape}")
        out = out.view(out.shape[0],-1)
        out = self.layer2(out)
        out = self.layer3(out) 
        return self.output_layer(out)

class DecisionModelV2_rs(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV2_rs, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # I doubt the residual connection is not being used 
        residual = x
        out = self.layer1(x)
        
        # out += residual
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        return self.output_layer(out) 


class DecisionModelV3(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelV3, self).__init__()
        
        # self.input_projection = nn.Linear(input_dim, 256)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Project input to 256 dimensions
        # x = self.input_projection(x)
        
        # Store the projected input as the residual
        # residual = x
        
        out = self.layer1(x)
        out = self.layer2(out)
        # residual = out
        out = self.layer3(out)
        # out += residual
        out = self.layer4(out)
        
        # Add the residual connection
        # out += residual
        
        return self.output_layer(out)

class DecisionModelWithSelfAttn(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3):
        super(DecisionModelWithSelfAttn, self).__init__()
        self.feature = input_dim
        self.key = nn.Linear(input_dim,input_dim)
        self.value = nn.Linear(input_dim,input_dim)
        self.query = nn.Linear(input_dim,input_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate attention weights
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.feature)
        attn_weights = torch.softmax(scores, dim=-1)

        # Calculate weighted sum of values
        x = torch.matmul(attn_weights, v)
        x = self.attn_dropout(x)
        out = self.layer1(x)
        out = self.layer2(out)
        return self.output_layer(out)
        
        
class DecisionModelV4(nn.Module):
    def __init__(self, input_dim=768, dropout_rate=0.3, num_heads=8, hidden_dim=512):
        super(DecisionModelV4, self).__init__()
        self.input_dim = input_dim
        
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2 + out1)
        
        return self.output_layer(out3)
class DecisionModelV6(nn.Module):
    def __init__(self):
        super(DecisionModelV6, self).__init__()
        self.layer1 = nn.Linear(768, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.layernorm1 = nn.LayerNorm(512)
        self.layernorm2 = nn.LayerNorm(512)
        self.layernorm3 = nn.LayerNorm(256)
        self.layernorm4 = nn.LayerNorm(128)
        self.layernorm5 = nn.LayerNorm(64)
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.3)

    def forward(self, x):
        x = self.gelu(self.layer1(x))
        x = self.layernorm1(x)
        x = self.dropout(x)
        
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_output.squeeze(1) + x
        
        x = self.gelu(self.layer2(x))
        x = self.layernorm2(x)
        x = self.dropout(x)
        
        x = self.gelu(self.layer3(x))
        x = self.layernorm3(x)
        x = self.dropout(x)
        
        x = self.gelu(self.layer4(x))
        x = self.layernorm4(x)
        x = self.dropout(x)
        
        x = self.gelu(self.layer5(x))
        x = self.layernorm5(x)
        x = self.dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x
class DecisionModelV0(nn.Module):
    def __init__(self):
        super(DecisionModelV0, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)