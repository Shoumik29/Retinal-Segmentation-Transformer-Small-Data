"""
Responsibility
	- Registers every available encoder with models.registry.ENCODERS. 
"""


from ..registry import ENCODERS
from . import Swin_Transformer


ENCODERS.register("swin_transformer")(Swin_Transformer.SwinTransformer())