import argparse
import os
import onnx
import numpy as np


def elem_size(dtype):
    t = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(dtype)
    if t is None:
        return None
    return np.dtype(t).itemsize


def tensor_nbytes(tensor):
    es = elem_size(tensor.data_type)
    if es is None:
        return None
    size = es
    for d in tensor.dims:
        size *= max(1, d)
    return size


def value_info_shape_vi(vi):
    shape = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField('dim_value'):
            shape.append(d.dim_value)
        else:
            shape.append(None)
    return shape


def shape_nbytes(vi):
    dtype = vi.type.tensor_type.elem_type
    es = elem_size(dtype)
    if es is None:
        return None
    shape = value_info_shape_vi(vi)
    if any(s is None for s in shape):
        return None
    n = es
    for s in shape:
        n *= s
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--topk', type=int, default=15)
    args = ap.parse_args()

    m = onnx.load(args.model)

    # Weights size
    weight_bytes = 0
    for init in m.graph.initializer:
        nb = tensor_nbytes(init)
        if nb:
            weight_bytes += nb
    print(f"Weights total: {weight_bytes/1024/1024:.2f} MB")

    # Gather shapes
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        nb = shape_nbytes(vi)
        if nb:
            shapes[vi.name] = nb

    # Add initializers as well (some may not be in value_info)
    for init in m.graph.initializer:
        nb = tensor_nbytes(init)
        if nb:
            shapes.setdefault(init.name, nb)

    # Show top-K biggest tensors
    top = sorted(shapes.items(), key=lambda x: x[1], reverse=True)[: args.topk]
    print(f"Top {len(top)} tensors by size:")
    for name, nb in top:
        print(f"  {name:60s} {nb/1024/1024:.2f} MB")

    # Estimate peak activation (very rough upper bound): sum of all non-initializer tensors
    init_names = {i.name for i in m.graph.initializer}
    act_bytes = sum(nb for n, nb in shapes.items() if n not in init_names)
    print(f"Naive upper-bound activations (sum of non-params): {act_bytes/1024/1024:.2f} MB")

if __name__ == '__main__':
    main()
