"""
Responsibility
	- Registers every available decoder with models.registry.DECOD
"""


from ..registry import DECODERS
from . import mlp_decoder


DECODERS.register("lfe")(mlp_decoder.LFE)