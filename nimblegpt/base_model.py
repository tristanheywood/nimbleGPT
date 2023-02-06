"""ABCs for each module in the model. This repo contains many implementations of the
various modules, to test different performance strategies, so it is useful to have a
common interface for all of them.

These base-classes also override flax's default auto-naming behaviour, so that e.g.
subclasses of `BaseBlock` will be named `Block_0`, `Block_1`, etc. instead of using the
name of the subclass. This lets all subclasses share the same parameter dict.

In all cases, `x` is the tensor of token embeddings, of shape 
[config.block_size, config.n_embd].
"""

import abc
from functools import partial
from typing import Any

import flax.linen as nn
import jax
from ml_collections import ConfigDict


class CustomPrefixModule(nn.Module):
    """
    Customized version of nn.Module with different autonaming behavior. `nn.Module`
    prefixes auto-names using the name of the class. This class allows subclasses to
    define a custom `prefix` field which will be used instead of the class name.
    """
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # We save the actual class name so it can be used in __repr__, to avoid
        # confusing debug output.
        cls._orig_cls_name = cls.__name__

        if cls.__module__ != "abc":
            # __module__ == 'abc' for classes created using `type()`. Flax transforms
            # like `vmap` create classes using `type()`. We avoid modifying the names of
            # such classes so that e.g. linen.vmap(MySubclass) is named 'VmapMySubclass',
            # instead of 'MySubclass' (since linen.vmap subclasses the input module).
            cls.__name__ = getattr(cls, "prefix", f'{cls.__name__}')

    def __repr__(self) -> str:
        return super().__repr__().replace(self.__class__.__name__,
                                          self._orig_cls_name)


class BaseSingleHeadCausalSelfAttention(CustomPrefixModule, abc.ABC):
    prefix = "SingleHeadCausalSelfAttention"

    n_feat: int

    @abc.abstractmethod
    def __call__(self, x, n_padd: int = 0):
        pass


class BaseCausalSelfAttention(CustomPrefixModule, abc.ABC):
    prefix = "CausalSelfAttention"

    n_head: int

    @abc.abstractmethod
    def __call__(self, x, n_padd: int = 0):
        pass


class BaseBlock(CustomPrefixModule, abc.ABC):
    prefix = "Block"

    n_head: int

    @abc.abstractmethod
    def __call__(self, x, n_padd: int = 0):
        pass


class BaseGPT(CustomPrefixModule, abc.ABC):
    prefix = "GPT"

    C: ConfigDict

    @abc.abstractmethod
    def __call__(self, indices: jax.Array, n_padd: int = 0):
        """
        Parameters
        ----------
        indicies : jnp.ndarray
            Array of token indices of shape (T,). See 'bpe.py' for how text is converted
            into indices.
        n_padd : int
            Number of padding tokens before the data tokens in `indices`.
        """
        pass
