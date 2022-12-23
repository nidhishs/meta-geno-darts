import torch
import torch.nn as nn
import torch.nn.functional as F

import model.operations as o


class SearchNetwork(nn.Module):
    def __init__(
            self, c_in, c, num_classes, num_layers, num_nodes, primitives, criterion, multiplier=4,
            stem_multiplier=3, dropout_proba=0.2
    ):
        super().__init__()

        self.num_class = num_classes
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.primitives = primitives
        self.criterion = criterion
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.dropout_proba = dropout_proba
        self.cells = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, self.stem_multiplier * c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stem_multiplier * c)
        )
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        c_out = self._compile_network(self.stem_multiplier * c, self.stem_multiplier * c, c, False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_out, self.num_class)

        self._alphas = [p for n, p in self.named_parameters() if "alpha" in n]
        self._weights = [p for n, p in self.named_parameters() if "alpha" not in n]

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, weights_reduce if cell.reduce else weights_normal)
        out = self.global_pool(s1)
        out = out.reshape(out.size(0), -1)
        logits = self.classifier(out)

        return logits

    def _compile_network(self, c_prev_prev, c_prev, c_curr, reduce_prev):
        for i in range(self.num_layers):
            reduce = i in [self.num_layers // 3, 2 * self.num_layers // 3]
            c_curr = c_curr * 2 if reduce else c_curr
            cell = SearchCell(
                self.num_nodes, self.multiplier, c_prev_prev, c_prev, c_curr, reduce, reduce_prev,
                self.primitives, self.dropout_proba
            )
            self.cells.append(cell)

            # The output of each cell is the concat of all internal nodes excluding two input nodes in channel dim.
            c_curr_out = c_curr * self.num_nodes
            c_prev_prev, c_prev = c_prev, c_curr_out
            reduce_prev = reduce

        for i in range(self.num_nodes):
            # The i^th node has i connections from the preceding i nodes.
            # There are two additional connections from C_{k-1} and C_{k-2}.
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, len(self.primitives))))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, len(self.primitives))))

        return c_prev


class SearchCell(nn.Module):
    def __init__(self, num_nodes, multiplier, c_prev_prev, c_prev, c, reduce, reduce_prev, primitives, dropout_proba):
        super().__init__()

        self.preproc0 = o.FactorizedReduce(c_prev_prev, c) if reduce_prev else o.ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preproc1 = o.ReLUConvBN(c_prev, c, 1, 1, 0)
        self.reduce, self.multiplier = reduce, multiplier
        self.dag = nn.ModuleList([nn.ModuleList() for _ in range(num_nodes)])

        for i, node in enumerate(self.dag):
            for j in range(2 + i):
                stride = 2 if (reduce and j < 2) else 1
                edge_op = o.MixedOp(c, stride, primitives, dropout_proba)
                node.append(edge_op)

    def forward(self, s0, s1, weights):
        s0, s1 = self.preproc0(s0), self.preproc1(s1)
        states = [s0, s1]

        for node, node_weights in zip(self.dag, weights):
            s = sum(edge_op(state, weight) for edge_op, state, weight in zip(node, states, node_weights))
            states.append(s)

        return torch.cat(states[-self.multiplier:], dim=1)
