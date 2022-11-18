
import torch
from torch import nn
import torchvision
import timm

from lib.model.temporal_fusion import make_Shift
from lib.utils.saveloadmodels import  WeightsData , CheckWeightsExistDownLoadElse


def timm_load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key: continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model



class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




class BackBone(nn.Module):
    def __init__(self, args, config_segments):
        super(BackBone, self).__init__()

        [n_segment,n_segment_shifted] = config_segments
        self.base_model, self.features_dim_out = self.prepare_base_model(args["model"])

        if n_segment_shifted > 0:
            self.base_model = self.insert_shift_temporal(self.base_model, n_segment, n_segment_shifted, args["shift_temporal"])

        self.dropout_last = nn.Dropout(p=args["model"]["dropout"])

    def prepare_base_model(self, param):

        if not 'resnet' in param["arch"]:
            raise ValueError(f'Unknown BackBone base model: {param["arch"]}')

        if "timm" in param["arch"]:
            base_model = timm.create_model('resnet50', pretrained=False)

            model_pretrained_param = WeightsData()
            [file_resnet50_miil_21k_pth, http_link_resnet50_miil_21k_pth] = model_pretrained_param["resnet50_miil_21k"]
            CheckWeightsExistDownLoadElse(file_resnet50_miil_21k_pth, http_link_resnet50_miil_21k_pth)

            base_model = timm_load_model_weights(base_model, file_resnet50_miil_21k_pth)
            print("prepare_base_model timm", file_resnet50_miil_21k_pth)
        else:
            base_model = getattr(torchvision.models, param["arch"])(True if param["pretrain"] == 'imagenet' else False)

        features_dim_out = getattr(base_model, 'fc').in_features
        setattr(base_model, 'fc', Identity())
        return base_model, features_dim_out

    def insert_shift_temporal(self, base_model, n_segment, n_segment_shifted,param):

        f_div = param["f_div"]
        shift_depth = param["shift_depth"]
        n_insert = param["n_insert"]
        m_insert = param["m_insert"]

        print("insert_temporal_shift n_segments, n_segment_shifted", n_segment, n_segment_shifted)
        print(f"make_temporal_shift n_insert={n_insert} m_insert={m_insert} f_div={f_div}")
        make_Shift(base_model, n_segment, n_segment_shifted, f_div, shift_depth, n_insert, m_insert)

        return base_model



    def forward(self, x ): #

        #(n_batches, n_samples, n_segment, c, h, w)  = x.size()

        x = x.reshape((-1,) + x.size()[-3:])
        #(n_batches * n_samples* n_segment, c, h, w)  = x.size()

        x = self.base_model(x)
        x = self.dropout_last(x)

        return x



