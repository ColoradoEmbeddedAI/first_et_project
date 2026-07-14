"""
pt_to_pte.py — Inspect a torch.export'd model and lower it to ExecuTorch .pte

Loads an ExportedProgram saved via torch.export.save(), prints the ops it
uses (both ATen ops from the raw export and edge-dialect ops after
to_edge(), which is what EXECUTORCH_SELECT_OPS_LIST needs), plus other
useful info about its inputs/outputs and parameters. Then lowers it to a
.pte file with executorch.

Usage (from a shell with your ExecuTorch venv active):
    source ~/executorch/et-env/bin/activate
    python3 util/pt_to_pte.py <input.pt> <output.pte>

The input .pt file must be an ExportedProgram saved with torch.export.save()
-- a plain torch.save(model) or torch.save(model.state_dict()) will NOT
load, since those don't carry the captured graph. To produce a compatible
.pt file (e.g. in a Colab notebook, after training):

    model.eval()
    example_input = (torch.randn(1, 1),)  # match your model's actual input shape/dtype
    exported_program = torch.export.export(model, example_input)
    torch.export.save(exported_program, "model.pt")
"""

import argparse
import sys
from collections import Counter

import torch
import executorch.exir as exir


def print_tensor_io(ep: torch.export.ExportedProgram) -> None:
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    outputs = [n for n in ep.graph.nodes if n.op == "output"]

    print("\nPlaceholder tensors:")
    for node in placeholders:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            print(f"  {node.name}: shape={tuple(val.shape)} dtype={val.dtype}")
        else:
            print(f"  {node.name}: {type(val).__name__}")

    print("\nOutput tensors:")
    for node in outputs:
        for arg in node.args[0]:
            if not isinstance(arg, torch.fx.Node):
                continue
            val = arg.meta.get("val")
            if isinstance(val, torch.Tensor):
                print(f"  {arg.name}: shape={tuple(val.shape)} dtype={val.dtype}")


def print_op_histogram(title: str, graph) -> Counter:
    counts = Counter(
        str(node.target) for node in graph.nodes if node.op == "call_function"
    )
    print(f"\n{title}")
    for op, count in sorted(counts.items()):
        print(f"  {count:4d}  {op}")
    return counts


def print_select_ops_list(graph) -> None:
    op_names = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if hasattr(target, "to_out_variant"):
            op_names.add(target.to_out_variant().name())
        else:
            op_names.add(target.name())

    ops_arg = ",".join(sorted(op_names))
    print("\nEXECUTORCH_SELECT_OPS_LIST for CMake:")
    print(f'  EXECUTORCH_SELECT_OPS_LIST="{ops_arg}"')


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a torch.export .pt file and lower it to ExecuTorch .pte"
    )
    parser.add_argument("input_pt", help="Path to a torch.export.save() .pt file")
    parser.add_argument("output_pte", help="Path to write the lowered .pte file")
    args = parser.parse_args()

    print(f"Loading exported program: {args.input_pt}")
    try:
        ep = torch.export.load(args.input_pt)
    except Exception as e:
        print(f"Failed to load '{args.input_pt}' as a torch.export ExportedProgram: {e}")
        sys.exit(1)

    print("\n=== ExportedProgram info ===")
    print_tensor_io(ep)

    param_names = list(ep.graph_signature.parameters)
    buffer_names = list(ep.graph_signature.buffers)
    total_params = sum(ep.state_dict[n].numel() for n in param_names if n in ep.state_dict)
    total_buffers = sum(ep.state_dict[n].numel() for n in buffer_names if n in ep.state_dict)
    print(f"\nParameters: {len(param_names)} tensors, {total_params} elements")
    print(f"Buffers:    {len(buffer_names)} tensors, {total_buffers} elements")

    print_op_histogram("ATen ops (raw torch.export graph):", ep.graph)

    print("\nLowering to edge dialect + ExecuTorch...")
    edge_program = exir.to_edge(ep)
    edge_graph = edge_program.exported_program().graph
    print_op_histogram(
        "Edge dialect ops (use .out variants for EXECUTORCH_SELECT_OPS_LIST):",
        edge_graph,
    )
    print_select_ops_list(edge_graph)

    et_program = edge_program.to_executorch()

    with open(args.output_pte, "wb") as f:
        f.write(et_program.buffer)

    print(f"\nExported to: {args.output_pte}")
    print(f"Model size:  {len(et_program.buffer)} bytes")


if __name__ == "__main__":
    main()
