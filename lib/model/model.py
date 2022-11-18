import torch
from torch import nn


class VCM(torch.nn.Module):

    def __init__(self, args, nn_input, nn_backbone):
        super(VCM, self).__init__()

        self.model_layers = {}

        self.nn_input = nn_input
        self.model_layers["nn_input"] = self.nn_input

        self.nn_backbone = nn_backbone
        self.model_layers["nn_backbone"] = self.nn_backbone

        self.num_class = args["model"]["num_class"]

        self.n_video_segments = self.nn_input.n_video_segments  # 8
        self.n_audio_segments = self.nn_input.n_audio_segments



        self.last_num = self.nn_backbone.features_dim_out

        print("last_fc", self.last_num, self.num_class)

        self.last_fc = nn.Linear(self.last_num, self.num_class)
        self.model_layers["last_fc"] = self.last_fc


    def forward(self, x_video, x_audio):

        x = self.nn_input(x_video, x_audio)

        (n_batches, n_samples, n_segment, c, h, w) = x.size()

        x = self.nn_backbone(x)
        x = x.view((-1, n_segment, self.last_num))
        x = x.mean(dim=1)
        x = x.view((-1,self.last_num))
        x = self.last_fc(x)

        return x




