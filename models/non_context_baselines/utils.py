import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR


class WarmupInverseSquareRootSchedule(LambdaLR):
    """Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = warmup_steps**0.5
        super(WarmupInverseSquareRootSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5
    
def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2)][len(str1)])

def collate_wordbased(batch):
    tag_list, text_list = [], []
    for (line, label) in batch:
        text_list.append(line)
        tag_list.append(label)
    return (
        pad_sequence(text_list, padding_value=0),
        pad_sequence(tag_list, padding_value=0)
    )

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if device == 0:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, args):
    src_mask = (src != 0).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(-1) # get seq_len for matrix
        np_mask = nopeak_mask(size, args.device)
        if trg.is_cuda:
            np_mask
        trg_mask = trg_mask.to(args.device) & np_mask.to(args.device)
    else:
        trg_mask = None
    return src_mask, trg_mask

def create_mask(src):
    src_mask = (src > 0).float()
    return src_mask