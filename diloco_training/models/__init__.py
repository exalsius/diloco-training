from typing import Callable, Dict

# from .big_gan import get_biggan
from .gcn import get_gcn
from .gpt_neo import get_gpt_neo, get_tiny_gpt_neo
from .gpt_neo_x import get_gpt_neo_x
from .resnet import get_resnet
from .wav2vec2 import get_wav2vec2

# from .ppo import get_ppo

MODEL_REGISTRY: Dict[str, Callable] = {
    "resnet50": lambda cache_dir=None: get_resnet("resnet50", cache_dir=cache_dir),
    "resnet101": lambda cache_dir=None: get_resnet("resnet101", cache_dir=cache_dir),
    "wav2vec2": get_wav2vec2,
    "gpt-neo": get_gpt_neo,
    "gpt-neo-x": get_gpt_neo_x,
    "gpt-neo-tiny": get_tiny_gpt_neo,
    "gcn": get_gcn,
    # "biggan": get_biggan,
    # "ppo": get_ppo,
}
