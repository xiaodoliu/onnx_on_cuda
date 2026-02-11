import argparse
from pathlib import Path


def parse_shape(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty shape")
    shape = [int(p) for p in parts]
    if any(d <= 0 for d in shape):
        raise ValueError("shape dims must be > 0")
    return shape


def main() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency 'numpy'. Install it in your venv:\n"
            "  python -m pip install numpy\n"
        ) from e

    p = argparse.ArgumentParser()
    p.add_argument(
        "--op",
        required=True,
        choices=["add", "mul", "fma", "exp", "sin", "softmax", "layernorm", "rmsnorm", "gelu", "silu"],
        help=(
            "fma is implemented as (x * A) + B per layer; "
            "softmax/layernorm/rmsnorm/gelu/silu are transformer-style elementwise+reduction blocks"
        ),
    )
    p.add_argument("--shape", default="1,4096,4096", help="comma-separated, e.g. 1,4096,4096")
    p.add_argument("--depth", type=int, default=256, help="number of times to apply the op")
    p.add_argument("--out", required=True, type=Path, help="output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--axis", type=int, default=-1, help="axis for softmax/norm (default: last dim)")
    p.add_argument("--epsilon", type=float, default=1e-5, help="epsilon for layernorm/rmsnorm")
    args = p.parse_args()

    if args.depth <= 0:
        raise SystemExit("--depth must be > 0")

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency 'onnx'. Install it in your venv:\n"
            "  python -m pip install onnx\n"
        ) from e

    shape = parse_shape(args.shape)

    x_name = "X"
    y_name = f"Y_{args.op}"

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    cur = x_name

    def add_const(name: str, value: float, *, dtype: np.dtype) -> None:
        arr = np.array([value], dtype=dtype)
        initializers.append(numpy_helper.from_array(arr, name=name))

    if args.op in ("add", "mul", "fma"):
        # Scalars broadcast across the whole tensor.
        add_const("C", 0.1234, dtype=np.float16)

    if args.op == "fma":
        add_const("A", 1.0007, dtype=np.float16)
        add_const("B", 0.4321, dtype=np.float16)

    if args.op in ("layernorm", "rmsnorm"):
        add_const("GAMMA", 1.0, dtype=np.float16)
        add_const("BETA", 0.0, dtype=np.float16)
        # Keep epsilon in fp32 to avoid underflow.
        add_const("EPS", float(args.epsilon), dtype=np.float32)

    if args.op == "gelu":
        # GELU(tanh) constants:
        # 0.5, 1.0, 0.044715, sqrt(2/pi)=0.7978845608028654
        add_const("C0_5", 0.5, dtype=np.float16)
        add_const("C1", 1.0, dtype=np.float16)
        add_const("C0_044715", 0.044715, dtype=np.float16)
        add_const("C0_7978846", 0.7978845608028654, dtype=np.float16)

    # ONNX opset 17 has mixed Reduce* signatures:
    # - ReduceSum: optional `axes` input tensor
    # - ReduceMax/ReduceMean: `axes` attribute
    # Keep an axes initializer for ReduceSum (softmax).
    if args.op in ("softmax",):
        axis_arr = np.array([int(args.axis)], dtype=np.int64)
        initializers.append(numpy_helper.from_array(axis_arr, name="AXIS"))

    for i in range(args.depth):
        if args.op == "add":
            out = f"add_{i}"
            nodes.append(helper.make_node("Add", [cur, "C"], [out], name=f"Add_{i}"))
            cur = out
        elif args.op == "mul":
            out = f"mul_{i}"
            nodes.append(helper.make_node("Mul", [cur, "C"], [out], name=f"Mul_{i}"))
            cur = out
        elif args.op == "fma":
            out_mul = f"mul_{i}"
            out_add = f"add_{i}"
            nodes.append(helper.make_node("Mul", [cur, "A"], [out_mul], name=f"Mul_{i}"))
            nodes.append(helper.make_node("Add", [out_mul, "B"], [out_add], name=f"Add_{i}"))
            cur = out_add
        elif args.op == "exp":
            out = f"exp_{i}"
            nodes.append(helper.make_node("Exp", [cur], [out], name=f"Exp_{i}"))
            cur = out
        elif args.op == "sin":
            out = f"sin_{i}"
            nodes.append(helper.make_node("Sin", [cur], [out], name=f"Sin_{i}"))
            cur = out
        elif args.op == "softmax":
            axis = int(args.axis)
            m = f"softmax_m_{i}"
            x0 = f"softmax_x0_{i}"
            ex = f"softmax_ex_{i}"
            s = f"softmax_s_{i}"
            out = f"softmax_{i}"
            nodes.append(
                helper.make_node("ReduceMax", [cur], [m], name=f"ReduceMax_{i}", axes=[axis], keepdims=1)
            )
            nodes.append(helper.make_node("Sub", [cur, m], [x0], name=f"Sub_{i}"))
            nodes.append(helper.make_node("Exp", [x0], [ex], name=f"Exp_{i}"))
            nodes.append(
                helper.make_node("ReduceSum", [ex, "AXIS"], [s], name=f"ReduceSum_{i}", keepdims=1)
            )
            nodes.append(helper.make_node("Div", [ex, s], [out], name=f"Div_{i}"))
            cur = out
        elif args.op == "layernorm":
            axis = int(args.axis)
            mu = f"ln_mu_{i}"
            xc = f"ln_xc_{i}"
            xc2 = f"ln_xc2_{i}"
            var = f"ln_var_{i}"
            var_f = f"ln_var_f_{i}"
            var_eps = f"ln_var_eps_{i}"
            std = f"ln_std_{i}"
            std_h = f"ln_std_h_{i}"
            y0 = f"ln_y0_{i}"
            y1 = f"ln_y1_{i}"
            out = f"layernorm_{i}"
            nodes.append(
                helper.make_node("ReduceMean", [cur], [mu], name=f"ReduceMean_mu_{i}", axes=[axis], keepdims=1)
            )
            nodes.append(helper.make_node("Sub", [cur, mu], [xc], name=f"Sub_xc_{i}"))
            nodes.append(helper.make_node("Mul", [xc, xc], [xc2], name=f"Mul_xc2_{i}"))
            nodes.append(
                helper.make_node("ReduceMean", [xc2], [var], name=f"ReduceMean_var_{i}", axes=[axis], keepdims=1)
            )
            nodes.append(helper.make_node("Cast", [var], [var_f], name=f"Cast_var_f32_{i}", to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Add", [var_f, "EPS"], [var_eps], name=f"Add_eps_{i}"))
            nodes.append(helper.make_node("Sqrt", [var_eps], [std], name=f"Sqrt_{i}"))
            nodes.append(helper.make_node("Cast", [std], [std_h], name=f"Cast_std_f16_{i}", to=TensorProto.FLOAT16))
            nodes.append(helper.make_node("Div", [xc, std_h], [y0], name=f"Div_norm_{i}"))
            nodes.append(helper.make_node("Mul", [y0, "GAMMA"], [y1], name=f"Mul_gamma_{i}"))
            nodes.append(helper.make_node("Add", [y1, "BETA"], [out], name=f"Add_beta_{i}"))
            cur = out
        elif args.op == "rmsnorm":
            axis = int(args.axis)
            x2 = f"rn_x2_{i}"
            ms = f"rn_ms_{i}"
            ms_f = f"rn_ms_f_{i}"
            ms_eps = f"rn_ms_eps_{i}"
            rms = f"rn_rms_{i}"
            rms_h = f"rn_rms_h_{i}"
            y0 = f"rn_y0_{i}"
            y1 = f"rn_y1_{i}"
            out = f"rmsnorm_{i}"
            nodes.append(helper.make_node("Mul", [cur, cur], [x2], name=f"Mul_x2_{i}"))
            nodes.append(
                helper.make_node("ReduceMean", [x2], [ms], name=f"ReduceMean_ms_{i}", axes=[axis], keepdims=1)
            )
            nodes.append(helper.make_node("Cast", [ms], [ms_f], name=f"Cast_ms_f32_{i}", to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Add", [ms_f, "EPS"], [ms_eps], name=f"Add_eps_{i}"))
            nodes.append(helper.make_node("Sqrt", [ms_eps], [rms], name=f"Sqrt_{i}"))
            nodes.append(helper.make_node("Cast", [rms], [rms_h], name=f"Cast_rms_f16_{i}", to=TensorProto.FLOAT16))
            nodes.append(helper.make_node("Div", [cur, rms_h], [y0], name=f"Div_norm_{i}"))
            nodes.append(helper.make_node("Mul", [y0, "GAMMA"], [y1], name=f"Mul_gamma_{i}"))
            nodes.append(helper.make_node("Add", [y1, "BETA"], [out], name=f"Add_beta_{i}"))
            cur = out
        elif args.op == "gelu":
            # GELU approximate (tanh):
            # y = 0.5*x*(1 + tanh(0.7978846*(x + 0.044715*x^3)))
            x = cur
            x2 = f"gelu_x2_{i}"
            x3 = f"gelu_x3_{i}"
            t0 = f"gelu_t0_{i}"
            t1 = f"gelu_t1_{i}"
            t2 = f"gelu_t2_{i}"
            th = f"gelu_th_{i}"
            one_plus = f"gelu_oneplus_{i}"
            s = f"gelu_s_{i}"
            out = f"gelu_{i}"
            nodes.append(helper.make_node("Mul", [x, x], [x2], name=f"Mul_x2_{i}"))
            nodes.append(helper.make_node("Mul", [x2, x], [x3], name=f"Mul_x3_{i}"))
            nodes.append(helper.make_node("Mul", [x3, "C0_044715"], [t0], name=f"Mul_c_{i}"))
            nodes.append(helper.make_node("Add", [x, t0], [t1], name=f"Add_{i}"))
            nodes.append(helper.make_node("Mul", [t1, "C0_7978846"], [t2], name=f"Mul_k_{i}"))
            nodes.append(helper.make_node("Tanh", [t2], [th], name=f"Tanh_{i}"))
            nodes.append(helper.make_node("Add", [th, "C1"], [one_plus], name=f"Add_1_{i}"))
            nodes.append(helper.make_node("Mul", [x, "C0_5"], [s], name=f"Mul_0_5_{i}"))
            nodes.append(helper.make_node("Mul", [s, one_plus], [out], name=f"Mul_out_{i}"))
            cur = out
        elif args.op == "silu":
            # SiLU / Swish: y = x * sigmoid(x)
            x = cur
            sig = f"silu_sig_{i}"
            out = f"silu_{i}"
            nodes.append(helper.make_node("Sigmoid", [x], [sig], name=f"Sigmoid_{i}"))
            nodes.append(helper.make_node("Mul", [x, sig], [out], name=f"Mul_{i}"))
            cur = out
        else:
            raise AssertionError("unreachable op")

    nodes.append(helper.make_node("Identity", [cur], [y_name], name="Output"))

    graph = helper.make_graph(
        nodes=nodes,
        name=f"fp16_mathbench_{args.op}",
        inputs=[helper.make_tensor_value_info(x_name, TensorProto.FLOAT16, shape)],
        outputs=[helper.make_tensor_value_info(y_name, TensorProto.FLOAT16, shape)],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="onnx_on_cuda/bench",
        opset_imports=[helper.make_opsetid("", args.opset)],
    )

    # onnx.helper does not expose a stable helper for metadata entries across versions,
    # so build StringStringEntryProto directly.
    from onnx.onnx_pb import StringStringEntryProto

    def add_meta(k: str, v: str) -> None:
        e = StringStringEntryProto()
        e.key = k
        e.value = v
        model.metadata_props.append(e)

    add_meta("bench.op", args.op)
    add_meta("bench.depth", str(args.depth))
    add_meta("bench.shape", ",".join(str(d) for d in shape))
    add_meta("bench.axis", str(args.axis))
    add_meta("bench.epsilon", str(args.epsilon))

    # Minimal sanity check.
    onnx.checker.check_model(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    elems = int(np.prod(shape))
    print(f"Wrote: {args.out}")
    print(f"op={args.op} shape={shape} elems={elems} depth={args.depth} dtype=fp16 opset={args.opset}")


if __name__ == "__main__":
    main()

