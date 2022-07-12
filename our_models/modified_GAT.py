from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class modified_GAT(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            key_type: int = 0,
            kq_dim: int = 8,
            att_norm: bool = False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.kq_dim = kq_dim
        self.key_type = key_type
        self.att_norm = att_norm

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # query value
        self.query_src = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')
        self.query_dst = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')

        # key value
        if key_type == 0:
            self.key_src = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')
            self.key_dst = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')
        elif key_type == 1:
            self.key_src = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')
            self.key_dst = self.key_src
        elif key_type == 2:
            self.key_dst = Linear(in_channels, heads * kq_dim, bias=False, weight_initializer='glorot')
            self.key_src = self.key_dst
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.query_src.reset_parameters()
        self.query_dst.reset_parameters()
        self.key_src.reset_parameters()
        self.key_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_value = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        # transform source and target node features to query
        query_src = self.query_src(x).view(-1, H, self.kq_dim)
        query_dst = self.query_dst(x).view(-1, H, self.kq_dim)
        query = (query_src, query_dst)

        # transform source and target node features to key
        if self.key_type == 0:
            key_src = self.key_src(x).view(-1, H, self.kq_dim)
            key_dst = self.key_dst(x).view(-1, H, self.kq_dim)
        elif self.key_type == 1:
            key_dst = key_src = self.key_src(x).view(-1, H, self.kq_dim)
        # elif self.key_type == 2:
        else:
            key_dst = key_src = self.key_dst(x).view(-1, H, self.kq_dim)
        key = (key_src, key_dst)

        # x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # alpha_src = (x_src * self.att_src).sum(dim=-1)
        # alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        # alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                if x is not None:
                    num_nodes = min(num_nodes, x.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)

            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        # out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size)
        out = self.propagate(edge_index, x=x_value, query=query, key=key, edge_attr=edge_attr, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, query_j: Tensor, query_i: Tensor, key_j: Tensor, key_i: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        query = F.leaky_relu(query_i + query_j, self.negative_slope)

        if self.key_type == 0:
            key = key_i + key_j
        elif self.key_type == 1:
            key = key_i
        # elif self.key_type == 2:
        else:
            key = key_j

        score = (query * key).sum(-1)

        if self.att_norm:
            score = score / (self.kq_dim ** 0.5)

        alpha = softmax(score, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
