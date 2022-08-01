import functools
from typing import Callable, Protocol

from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from . import ops

# Register custom calls.
for _name, _value in ops.custom_call_targets().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def make_primitive(
    primitive_name: str,
    abstract_eval: Callable,
    translation_rule: xla.TranslationRule,
) -> core.Primitive:
    prim = core.Primitive(primitive_name)
    prim.multiple_results = True
    prim.def_impl(functools.partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(abstract_eval)
    xla.register_translation(prim, translation_rule, platform="gpu")
    return prim
