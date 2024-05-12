from torch import nn
from .dictionary_iris import *
import snntorch as snn
import torch

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=beta)
        self.mean_spks = 0

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []

        total_spks = int(x.sum())
        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            total_spks += int(spk1.sum() + spk2.sum())

        mem_return = torch.stack(mem1_rec, dim=0), torch.stack(mem2_rec, dim=0)
        self.mean_spks = (total_spks if self.mean_spks==0 else 0.95*self.mean_spks + 0.05*total_spks)
        
        return torch.stack(spk2_rec, dim=0), mem_return