"""
Responsibility
    - Registry for encoders and decoders.
A component (an encoder or a decoder) registers itself by
name using a decorator, and builder.py looks for components by the name
in the config

Examples:
@ENCODERS.register("my_new_encoder")
    def build_my_new_encoder(x, **kwargs):
        ...
    # configs/experiment_03.yaml
    model:
      encoder: "my_new_encoder"
"""


from typing import Callable, Dict, List


class ComponentRegistry:

    """A name -> callable map for model component (an encoder or a decoder)."""

    def __init__(self, type: str):
        self.type = type
        self._components: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
    
        """
        Decorator: register a function under a name.

        Examples:
            @ENCODERS.register("swin_pretrained")
            def SwinTransformer(x, **kwargs):
                ...
            or,
            ENCODERS.register("swin_pretrained")(SwinTransformer)

        Raises:
            ValueError: if name is already registered.
        """

        def decorator(fn: Callable) -> Callable:

            if name in self._components:
                raise ValueError(
                    f"{self.type} '{name}' is already registered "
                    f"(existing: {self._components[name]}, new: {fn})"
                )
            self._components[name] = fn

            return fn

        return decorator

    def get(self, name: str) -> Callable:

        """
        Search registered component by name.

        Raises:
            KeyError: if name isn't registered, lists what is
                registered.
        """

        if name not in self._components:
            raise KeyError(
                f"Unknown {self.type} '{name}'. Registered {self.type}s: {self.names()}"
            )
        
        return self._components[name]

    def names(self) -> List[str]:

        """All currently registered names."""

        return sorted(self._components)


ENCODERS = ComponentRegistry("encoder")
DECODERS = ComponentRegistry("decoder")
LOSSES = ComponentRegistry("loss")