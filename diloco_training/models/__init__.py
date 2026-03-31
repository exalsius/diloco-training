from typing import Callable, Dict

from .gpt_neo import get_gpt_neo, get_tiny_gpt_neo, get_small_gpt_neo
from .gpt_neo_x import get_gpt_neo_x

MODEL_REGISTRY: Dict[str, Callable] = {
    "gpt-neo": get_gpt_neo,
    "gpt-neo-x": get_gpt_neo_x,
    "gpt-neo-tiny": get_tiny_gpt_neo,
    "gpt-neo-small": get_small_gpt_neo,
}
