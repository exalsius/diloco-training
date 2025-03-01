from .gcn import get_gcn
from .gpt_neo import get_gpt_neo
from .resnet import get_resnet
from .wav2vec2 import get_wav2vec2

from .big_gan import get_biggan
# from .ppo import get_ppo

MODEL_REGISTRY = {
    "resnet50": lambda: get_resnet("resnet50"),
    "resnet101": lambda: get_resnet("resnet101"),
    "wav2vec2": get_wav2vec2,
    "gpt-neo": get_gpt_neo,
    "gcn": get_gcn,
    "biggan": get_biggan,
    # "ppo": get_ppo,
}
