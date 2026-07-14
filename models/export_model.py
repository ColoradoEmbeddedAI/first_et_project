"""
export_model.py — Export a PyTorch model to ExecuTorch .pte format

Run this from the project root with your ExecuTorch venv active:
    source ~/executorch/et-env/bin/activate
    python3 models/export_model.py

The exported .pte file is written to models/tiny_model.pte.
"""

import os
import torch
import executorch.exir as exir


class TinyModel(torch.nn.Module):
    """
    Minimal test model: f(x) = x * 2 + 1

    For input [1, 2, 3, 4], expected output is [3, 5, 7, 9].

    This model uses:
      - dim_order_ops::_to_dim_order_copy.out  (implicit tensor copies)
      - aten::mul.out                           (element-wise multiply)
      - aten::add.out                           (element-wise add)
    """
    def forward(self, x):
        return x * 2 + 1


def main():
    os.makedirs("models", exist_ok=True)

    model = TinyModel()
    model.eval()

    # Input shape must match what you feed at inference time.
    # The shape is baked into the .pte file — changing it requires re-export.
    example_input = (torch.randn(1, 4),)

    print("Exporting model...")

    # torch.export captures the computation graph
    ep = torch.export.export(model, example_input)

    # Convert to ExecuTorch format
    et_program = (
        exir.to_edge(ep)
        .to_executorch()
    )

    output_path = "models/tiny_model.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to: {output_path}")
    print(f"Model size:  {len(et_program.buffer)} bytes")

    # Print ops used — useful for setting EXECUTORCH_SELECT_OPS_LIST
    print("\nOps in exported graph (use .out variants for SELECT_OPS_LIST):")
    for node in exir.to_edge(ep).exported_program().graph.nodes:
        if node.op == 'call_function':
            print(f"  {node.target}")

    # Quick sanity check on the host
    print("\nRunning host sanity check...")
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        result = model(test_input)
    print(f"Input:    {test_input.tolist()}")
    print(f"Output:   {result.tolist()}")
    print(f"Expected: [[3.0, 5.0, 7.0, 9.0]]")
    assert result.tolist() == [[3.0, 5.0, 7.0, 9.0]], "Model output mismatch!"
    print("Sanity check PASSED")


if __name__ == "__main__":
    main()
