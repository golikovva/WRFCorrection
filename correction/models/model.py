from torch import nn
import torch.nn.functional as F
import torch
from correction.models.make_layers import make_layers
from correction.models.model_parts import DeltaConvLayer

class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        o_input = torch.split(input, 3, dim=-3)
        output = output + o_input[0]
        return output


class Predictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = make_layers(params)

    def forward(self, input):
        '''
        input: S*B*1*H*W
        :param input:
        :return:
        '''
        input = input.squeeze(2).permute((1, 0, 2, 3))
        output = self.model(input)
        return output.unsqueeze(2).permute((1, 0, 2, 3, 4))


class Corrector(nn.Module):
    def __init__(self, model, channels=6):
        super().__init__()
        self.channels = channels
        self.unet = model

    def forward(self, x_orig):
        x = x_orig
        unet_out = self.unet(x)
        o_input = torch.split(x_orig, 3, dim=-3)
        return o_input[0] + unet_out.view(x.shape[0], x.shape[1], 3, x.shape[3], x.shape[4])


class LowFreqCorrector(nn.Module):
    def __init__(self, model, in_channels=8, out_channels=3, k=7, inference_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = model
        self.freq_conv = DeltaConvLayer(k, 'gauss')
        self.inference_mode = inference_mode

    def forward(self, x_orig):
        s = x_orig.shape
        with torch.no_grad():
            o_input, metadata = torch.split(x_orig, [self.out_channels, s[-3]-self.out_channels], dim=-3)

            so = o_input.shape
            l_freq = self.freq_conv(o_input.view(-1, *so[-3:])).view(so)

            h_freq = o_input - l_freq
        l_freq_corr = self.model(torch.cat((l_freq, metadata), dim=-3))
        l_freq_corr = l_freq_corr.view(*s[:2], *l_freq_corr.shape[1:])

        return l_freq, l_freq_corr, h_freq

