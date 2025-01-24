from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(3, 64, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # Apply average pooling to compress global spatial information: (B, C, H, W) --> (B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x))  # Apply max pooling to compress global spatial information: (B, C, H, W) --> (B, C, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x):
        avg_out = torch.mean(x, dim=0, keepdim=True)
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=0)
        x = self.conv1(x)
        return self.sigmoid(x)

class Soft_Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Soft_Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
            # nn.Linear(hidden_size, hidden_size, bias=False)
        )
    def reset_parameters(self):
        for layer in self.project:
            if hasattr(layer, 'reset_parameters'):
                print(layer)
                layer.reset_parameters()
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        beta = beta
        return (beta * z).sum(1), beta

class CategoryAttentionBlock(nn.Module):
    def __init__(self, in_channels, classes, k):
        super(CategoryAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, k * classes, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(k * classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classes = classes
        self.k = k

    def forward(self, inputs):
        chanel, node, feature = inputs.size()

        # Convolution, Batch Normalization, and ReLU
        F = self.conv(inputs)
        F1 = self.relu(F)

        # Global Max Pooling
        x = self.max_pool(F1)
        x = x.view(self.classes,self.k)
        # Compute the attention vector S
        S = x.mean(dim=-1, keepdims=False)
        S = S.view(-1,1,1)

        # Reshape and mean pooling along the 'k' dimension
        x = F1.view(self.classes,self.k,node,feature)

        x = x.mean(dim=1, keepdims=False)
        # Element-wise multiplication
        x = S * x
        # Mean pooling along the 'k' dimension to get the final attention mask M
        M = x.mean(dim=0, keepdims=True)

        # Final multiplication with the input to get the output
        semantic = inputs * M

        return semantic

class Eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FE_GNN(torch.nn.Module):
    def __init__(self,input,hidden_channels):
        super(FE_GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels,2)
        self.ca = ChannelAttention()
        self.sa = SpatialAttention()
        self.ce = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1,padding=3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1,padding=3),
            nn.ReLU(),
        )
        self.conv4 = nn.Conv2d(6, 1, kernel_size=7, stride=1,padding=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def reset_parameters(self):
        #Initialize parameters.
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin.reset_parameters()
        self.ca.reset_parameters()
        self.sa.reset_parameters()
        for layer in self.ce:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.conv4, 'reset_parameters'):
            self.conv4.reset_parameters()

    def forward(self, x, edge_index, batch):
        x11 = self.conv1(x, edge_index)
        x1 = x11.relu()
        x22 = self.conv2(x1, edge_index)
        x2 = x22.relu()
        x3 = self.conv3(x2, edge_index)
        x = torch.stack([x1, x2, x3])
        x_ce = self.ce(x)
        x_ca = self.ca(x)
        x_sa = self.sa(x)
        x = x * x_ca
        x_cabm = x * x_sa
        x_con = torch.concat((x_cabm,x_ce))
        # 2. Readout layer
        x = self.conv4(x_con)
        x = x.squeeze(0)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
