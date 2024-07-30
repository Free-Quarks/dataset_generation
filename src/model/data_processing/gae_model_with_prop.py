import numpy as np
import torch
from torch_geometric.nn import GCNConv, SAGPooling,  GATv2Conv, GAE, InnerProductDecoder, VGAE
from typing import Optional, Tuple


from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F

# Directory with graphs data with csv files

from sklearn.preprocessing import OneHotEncoder

class GraphLinearLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLinearLayer, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass through the model.

        Parameters:
        - x: Node feature matrix of shape [num_nodes, num_node_features].
        - edge_index: Edge index tensor of shape [2, num_edges].

        Returns:
        - Transformed aggregated features of shape [num_nodes, out_features].
        """
        #Finding neighbors and aggregate their features
        row, col = edge_index
        neighbor_features = x[col]  # Features of neighbors
        aggregated_features = neighbor_features.sum(dim=0)  # Summing up neighbor features

        #Apply linear transformation
        transformed_features = self.linear(aggregated_features)

        return transformed_features





######### GAE from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#GAE
class GAEncoder(torch.nn.Module):
    """
    Encoder part of the Auto Encoder
    """
    def __init__(self, properties_dim, edge_dim, in_channels, hidden_channels, out_channels): #, heads=1):
        super().__init__()
        # GATConv has a self sttention
        #self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        #self.graph_lin = GraphLinearLayer(properties_dim, in_channels)

        #self.conv1 = GCNConv(in_channels, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, out_channels)
        #self.conv3 = GCNConv(out_channels, out_channels)

        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        #self.conv = GCNConv(properties_dim, hidden_channels)
        self.conv = GATv2Conv(properties_dim, hidden_channels, edge_dim=edge_dim)

        self.pool = SAGPooling(hidden_channels, ratio=0.5) # Self-Attention Graph Pooling where choosing ratio=0.5 means half of the nodes are kept and rest are merged into smaller graph

        self.linear = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, x2, edge_index, edge_attr ,pe):
        print(f'x.shape= {x.shape}')
        print(f'pe.shape= {pe.shape}')
        x = x+pe
        #x = pe
        print(f'x.shape= {x.shape}')
        print(f'edge_index.shape= {edge_index.shape}')
        print(f'edge_attr.shape= {edge_attr.shape}')

        x = F.relu(self.conv1(x, edge_index, edge_attr)) #.mean(dim=1)

        print(f'-->x.shape= {x.shape}')

        #x2 = self.dropout(x2)
        x2 = F.relu(self.conv(x2, edge_index, edge_attr))

        print(f'-->x2.shape= {x2.shape}')
        #x2 = self.graph_lin(x2)

        print(f'x2:{x2}')
        print(f'x2.shape:{x2.shape}')

        x = x + x2


        print(f'x.shape= {x.shape}')
        print(f'edge_index.shape= {edge_index.shape}')
        x, edge_index,edge_attr, _, _, _ = self.pool(x, edge_index, edge_attr)
        print(f'==> x.shape= {x.shape}')
        print(f'==> edge_index.shape= {edge_index.shape}')

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        print(f'x.shape= {x.shape}')
        x = self.conv3(x, edge_index)
        print(f'x= {x}')
        print(f'x.shape= {x.shape}')

        x = F.relu(self.linear(x))
        print(f'>>>>>>>>> x.shape= {x.shape}')

        x = global_mean_pool(x, batch=None)
        print(f'x.shape= {x.shape}')
        return x


class GADecoder(torch.nn.Module):
    def __init__(self, edge_dim, in_channels, hidden_channels, out_channels):
        super().__init__()
        #self.decoder_conv1 = GCNConv(out_channels, hidden_channels)
        self.decoder_conv1 = GATv2Conv(out_channels, hidden_channels, edge_dim=edge_dim)
        self.decoder_conv2 = GCNConv(hidden_channels, 13)
        self.decoder_conv3 = GCNConv(13, 13)
        self.linear1 = torch.nn.Linear(out_channels, 13)
        self.linear2 = torch.nn.Linear(13, 13)

    def forward(self, z, num_nodes, edge_index, edge_attr,  pe):

        print(f'z.shape:{z.shape}')
        z = torch.tile(z, (num_nodes, 1))

        print(f'z.shape:{z.shape}')

        print(f'pe.shape:{pe.shape}')


        z1 = self.decoder_conv1(z, edge_index, edge_attr)

        z1 = F.relu(z1)
        print(f'z1.shape:{z1.shape}')

        z1 = self.decoder_conv2(z1, edge_index)

        z1 = F.relu(z1)
        print(f'z1.shape:{z1.shape}')
        print(f'----> z1:{z1}')

        z1 = self.decoder_conv3(z1+pe, edge_index)
        z1 = F.relu(z1)
        print(f'z1.shape:{z1.shape}')
        print(f'----> z1:{z1}')

        print()
        return z1









'''class GADecoder(torch.nn.Module):
    """
    Decoder for Auto Encoder the  that takes teh inner product of the latent space matrix, z
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.InnerProductDecoder.html#torch_geometric.nn.models.InnerProductDecoder
    """
    def __init__(self):
        super().__init__()
    def forward(self, z, num_nodes, edge_index, sigmoid: bool = True):
        """
        Forward pass of the decoder which transforms the latent space representation to edge probabilities
        Args:
            z:
            edge_index:
            sigmoid:

        Returns: Probabilities if sigmoid is applied providing edge probabilities or else returns values for

        """

        # Inner product between pairs of nodes that is obtained from edge_index
        # which multiplies the node embeddings connected via edges
        # and then sums the products  along the nodes (via dimension=1)
        #value = (z[edge_index[0]]*z[edge_index[1]]).sum(dim=1)

        print(f'z.size(0) = {z.size(0)}')
        print(f'z.size(1) = {z.size(1)}')
        features_per_node = z.size(0) // num_nodes
        print(f'features_per_node = {features_per_node}')

        # First Initialize the tensor with zeros
        x_unpooled = torch.zeros((num_nodes, z.size(1)), device=z.device)

        # Assign pooled features to each node
        for i in range(num_nodes):
            start_idx = i * 13 // 13 is the number of features per node
            end_idx = start_idx + features_per_node
            x_unpooled[i, :] = z[start_idx:end_idx].mean(dim=0)

        print(f'x_unpooled.shape = {x_unpooled.shape}')

        print(f'x_unpooled = {x_unpooled}')


        projection_layer = torch.nn.Linear(z.shape[1], num_nodes)

        projected_z = projection_layer(z.unsqueeze(0))
        print(f'projected_z= {projected_z}')
        print(f'projected_z.shape= {projected_z.shape}')
        reconstructed_features = projected_z.squeeze().view(num_nodes, -1)
        print(f'reconstructed_features= {reconstructed_features}')
        print(f'reconstructed_features.shape= {reconstructed_features.shape}')

        return reconstructed_features #torch.sigmoid(value) if sigmoid else value'''


class GraphAutoEncoder(torch.nn.Module):
    """
    Graph Auto Encoder
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAE.html#torch_geometric.nn.models.GAE
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder # Encoder module
        self.decoder = decoder # Optional decoder module #InnerProductDecoder() if decoder is None else decoder
        GraphAutoEncoder.reset_parameters(self)
    def reset_parameters(self):
        """
        Resets all learnable parameters of the encoder and decoder module.
        """
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)
    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        EPS = 1e-15
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
