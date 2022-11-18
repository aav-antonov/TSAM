import torch


class x_input(torch.nn.Module):

    def __init__(self, args ):
        super(x_input, self).__init__()

        self.n_video_segments = args["model"]["video_segments"] #[8, 8, 1]
        self.n_audio_segments = args["model"]["audio_segments"]

        self.args = args

        n_segments = self.n_video_segments +  self.n_audio_segments
        n_segment_shifted = self.n_video_segments
        self.config_segments = [n_segments, n_segment_shifted]


    def forward(self, x_video , x_audio):

        with torch.no_grad():

            k_frames = x_video.size()[3]
            x  = x_video[:, :, :, k_frames // 2]


            if self.args["model"]["audio_segments"] > 0:
                x = torch.cat((x, x_audio), 2)


        return x


