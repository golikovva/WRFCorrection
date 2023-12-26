from experiments.net_params import encoder_params, forecaster_params, \
    conv2d_params, convlstm_encoder_params, convlstm_forecaster_params, unet_params
from correction.models.forecaster import Forecaster
from correction.models.encoder import Encoder
from correction.models.unet import UNet
from correction.models.ConstantBias import ConstantBias
from correction.models.model import EF, Predictor, Corrector
from correction.models.downscaling_model import LinearDownscaling
from correction.config.config import cfg


def build_correction_model(model_type):
    if model_type == "trajGRU":
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
        return encoder_forecaster
    elif model_type == "conv2d":
        model = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
        return model
    elif model_type == "convLSTM":
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
        return encoder_forecaster
    elif model_type == "BERTunet":
        unet = UNet(*unet_params.values())
        model = Corrector(unet).to(cfg.GLOBAL.DEVICE)
        return model
    elif model_type == "ConstantBaseline":
        model = ConstantBias(3).to(cfg.GLOBAL.DEVICE)
        return model
    else:
        raise TypeError


def build_downscaling_model(model_type, metadata):
    if model_type == "Linear":
        model = LinearDownscaling(metadata).to(cfg.GLOBAL.DEVICE)
        return model


def build_scheduler(scheduler_type):
    if scheduler_type == "MultiStepLR":
        pass
