from experiments.net_params import encoder_params, forecaster_params, \
    conv2d_params, convlstm_encoder_params, convlstm_forecaster_params, unet_params
from correction.models.forecaster import Forecaster
from correction.models.encoder import Encoder
from correction.models.unet import UNet
from correction.models.ConstantBias import ConstantBias
from correction.models.model import EF, Predictor, Corrector, LowFreqCorrector
from correction.models.downscaling_model import LinearDownscaling
import torch


def build_correction_model(cfg):
    if cfg.model_type == "trajGRU":
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.device)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.device)
        model = EF(encoder, forecaster).to(cfg.device)
    elif cfg.model_type == "conv2d":
        model = Predictor(conv2d_params).to(cfg.device)
    elif cfg.model_type == "convLSTM":
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.device)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.device)
        model = EF(encoder, forecaster).to(cfg.device)
    elif cfg.model_type == "BERTunet":
        unet = UNet(*cfg.model_args.BERTunet.values())
        model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == "ConstantBaseline":
        model = ConstantBias(3).to(cfg.device)
    elif cfg.model_type == 'BERTunet_lfreq':
        unet = UNet(*unet_params.values())
        model = LowFreqCorrector(unet).to(cfg.device)
    else:
        raise TypeError
    return model


def build_inference_correction_model(cfg):
    if cfg['model_type'] == "BERTunet":
        unet = UNet(n_channels=9, n_classes=3, bilinear=True)
        model = Corrector(unet).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    else:
        raise TypeError
    return model
