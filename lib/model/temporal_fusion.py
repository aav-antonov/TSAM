

import torch.nn as nn
import torchvision
import timm


class TemporalShift(nn.Module):
    """
    (n_batch , n_segment, c,h,w) = X.size() 
    n_segment - total number of segments in X video:motion+audio
    n_segment_shifted - number of segments to shift (only segments (0:n_segment_shifted) are shifted); audio segments is not shifted
    shift_depth - extends original TSM implementation to shift/fuse frames/segments with temporal distance > 1, shift_depth = 1 eqvivalent to original implemetation
    f_div - defines the propotion of features to shift
    f_div/shift_depth defines the propotion of features shifted between segments with temporal distance d <= shift_depth
    """

    def __init__(self, layer, n_segment, n_segment_shifted, f_div, shift_depth):
        super(TemporalShift, self).__init__()
        self.layer = layer
        self.in_channels = self.layer.in_channels
        self.f_n = int(self.in_channels // f_div)
        self.shift_depth = shift_depth
        self.n_segment = n_segment
        self.n_shift = n_segment_shifted


    def forward(self, x):
        x = self.shift_temporal(x)
        x = self.layer(x)
        return x

    def shift_temporal(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        f = self.f_n // self.shift_depth
        out = x.clone()
        s = self.n_shift

        for d in range(self.shift_depth):
            l = d + 1
            f1 = 2 * d * f
            f2 = (2 * d + 1) * f
            # print("shift_temporal d", d, f1, f2)
            out[:, 0:s - l, f1:f1 + f] = x[:, l:s, f1:f1 + f]
            out[:, l:s, f2:f2 + f] = x[:, 0:s - l, f2:f2 + f]

        return out.view(nt, c, h, w)


def shift_block(stage, n_segments, n_segment_shifted, f_div, shift_depth, n_insert, m_insert):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i % n_insert == m_insert:
            print(f'shift_block:  stage with {len(blocks)} blocks ')
            blocks[i].conv1 = TemporalShift(b.conv1, n_segments, n_segment_shifted, f_div, shift_depth)

    return stage


def make_Shift(net, n_segments, n_segment_shifted, f_div, shift_depth, n_insert, m_insert):

    if isinstance(net, (torchvision.models.ResNet, timm.models.ResNet)):
        """
        # print out blocks of ResNet
        for i, b in enumerate(net.children()):
            print("net", i, b)
        """

        for i, b in enumerate(net.children()):
        #for name, m in model.named_modules():

            if i in [4, 5, 6, 7]:
                print("make_Shift blocks", i )
                shift_block(b, n_segments, n_segment_shifted, f_div, shift_depth, n_insert, m_insert)

    else:
        raise NotImplementedError("UnKnown Arch (not ResNet )")











