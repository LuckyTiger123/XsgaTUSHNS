from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch_geometric.typing import OptTensor


class feature_propagation(MessagePassing):
    def __init__(self, k: int = 1, eps: float = 1, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)
        self.k = k
        self.eps = eps

    def forward(self, x: Tensor, edge_index: Tensor, k: int, f_mask: Tensor = None, edge_attr: OptTensor = None):

        # remove self loop
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        for i in range(k):
            out = self.propagate(edge_index, x=x)
            if f_mask is None:
                x = (self.eps * x + out) / (1 + self.eps)
            else:
                x = (self.eps * x + out * f_mask) / (1 + self.eps)
        return x
