import torch
import torch.nn as nn
import collections

ButcherTableau = collections.namedtuple('ButcherTableau', 'a, b, c, c_err')

# TO DO: re-utilize precomputed `f` evaluations (passed to `g`)
class GenericExplicitButcher(nn.Module):
    def __init__(self, tableau):
        super().__init__()
        self.tab = tableau
        self.order = len(tableau.a)
    
    def forward(self, s, z, f, eps):
        stages = []
        for i, a in enumerate(self.tab.a): 
            stval = torch.zeros_like(z)
            for j, stage in enumerate(stages):
                # determine stage value
                stval += a[j]*stages[j]
            # evaluate f and store stage
            stages += [f(s + self.tab.c[i], z + eps*stval)]
        return sum([self.tab.b[i]*stages[i] for i in range(len(self.tab.b))])
    