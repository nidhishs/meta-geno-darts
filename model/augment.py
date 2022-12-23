import torch
import torch.nn as nn

import model.operations as o


class AugmentCell(nn.Module):
    def __init__(self, genotype, c_prev_prev, c_prev, c, reduce, reduce_prev, dropout):
        super().__init__()

        self._preproc0 = o.FactorizedReduce(c_prev_prev, c) if reduce_prev else o.ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self._preproc1 = o.ReLUConvBN(c_prev, c, 1, 1, 0)
        self.reduce = reduce

        if reduce:
            self._op_names, self._indices = zip(*genotype.reduce)
            self._concat = genotype.reduce_concat
        else:
            self._op_names, self._indices = zip(*genotype.normal)
            self._concat = genotype.normal_concat

        self.num_nodes = len(self._op_names) // 2
        self.multiplier = len(self._concat)
        self._ops = nn.ModuleList()

    def _compile_cell(self, c, op_names, indices, reduce, dropout_proba):
        for name, index in zip(op_names, indices):
            stride = 2 if (reduce and index < 2) else 1
            op = o.OPERATIONS[name](c, stride, True)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(c, affine=False))
            if isinstance(op, nn.Identity) and dropout_proba > 0:
                op = nn.Sequential(op, nn.Dropout(dropout_proba))
            self._ops.append(op)

    def forward(self, s0, s1):
        s0, s1 = self._preproc0(s0), self._preproc1(s1)
        states = [s0, s1]

        for i in range(self.num_nodes):
            o1, o2 = self._ops[2 * i], self._ops[2 * i + 1]
            x1, x2 = states[self._indices[2 * i]], states[self._indices[2 * i + 1]]
            s = o1(x1) + o2(x2)
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)
