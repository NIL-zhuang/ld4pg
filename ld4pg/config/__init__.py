from collections import namedtuple

DEFAULT_DIM_HEAD = 64
DIFFUSION_EPS = 1e-5
BART_ENCODER_MEAN = 0.002565157
BART_ENCODER_STD = 0.121004902

Intermediates = namedtuple(
    'Intermediates', [
        'pre_softmax_attn',
        'post_softmax_attn'
    ])

LayerIntermediates = namedtuple(
    'Intermediates', [
        'hiddens',
        'attn_intermediates'
    ])
ModelPrediction = namedtuple(
    'ModelPrediction', [
        'pred_noise',
        'pred_x_start'
    ])

DATASET_PATH = "datasets"
