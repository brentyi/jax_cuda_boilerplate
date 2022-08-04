"""Utilities for making CUDA primitives, defining lowering rules."""


import functools
import itertools
from typing import Any, Sequence, Union, cast

import jax.interpreters.mlir as mlir
import jaxlib.mlir.dialects.mhlo as mhlo
import jaxlib.mlir.ir as ir_
from jax import core
from jax.interpreters import xla
from jaxlib import xla_client
from typing_extensions import TypeAlias

from . import custom_calls_ext, jax_utils_ext

# For suppressing type errors.
ir: Any = ir_
assert hasattr(ir, "StringAttr")
assert hasattr(ir, "IntegerAttr")
assert hasattr(ir, "TupleType")
assert hasattr(ir, "TupleType")

Value = Any  # ir.Value


def make_cuda_primitive(custom_call_target: str) -> core.Primitive:
    """Helper for building JAX primitives from custom call targets.

    Assumes `abstract_eval()` returns a tuple of ShapedArrays.
    """

    capsule = custom_calls_ext.custom_call_targets().get(custom_call_target)
    assert (
        capsule is not None
    ), f"Call target {custom_call_target} not found in registrations!"
    xla_client.register_custom_call_target(
        custom_call_target,
        capsule,
        # Using platform="cuda" here breaks.
        platform="gpu",
    )

    primitive_name = custom_call_target
    prim = core.Primitive(primitive_name)
    prim.multiple_results = True
    prim.def_impl(functools.partial(xla.apply_primitive, prim))
    mlir.register_lowering(
        prim,
        rule=_make_lowering_rule(custom_call_target),
        # Using platform="gpu" here is deprecated.
        platform="cuda",
    )
    return prim


def _make_lowering_rule(custom_call_target: str) -> mlir.LoweringRule:
    def lowering_rule(
        ctx: mlir.LoweringRuleContext,
        *args: Union[Value, Sequence[Value]],
        **kw,
    ) -> Sequence[Union[Value, Sequence[Value]]]:
        out_types = list(map(mlir.aval_to_ir_type, ctx.avals_out))
        shape_descriptor = jax_utils_ext.ShapesDescriptor.construct(
            [
                tuple(cast(core.ShapedArray, x).shape)
                for x in itertools.chain(ctx.avals_in, ctx.avals_out)
            ]
        )
        shape_descriptor = shape_descriptor.as_bytes()

        i32_type = ir.IntegerType.get_signless(32)
        out = mhlo.CustomCallOp(
            result=[ir.TupleType.get_tuple(out_types)],
            operands_=args,
            call_target_name=ir.StringAttr.get(bytes(custom_call_target, "utf-8")),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(shape_descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 2),
        )
        return [
            mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
            for i in range(len(out_types))
        ]

    return lowering_rule
