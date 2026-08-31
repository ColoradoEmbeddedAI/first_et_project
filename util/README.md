# util/

## pt_to_pte.py

Inspects a `torch.export`'d model and lowers it to an ExecuTorch `.pte` file.

### What it does

1. Loads an `ExportedProgram` from a `.pt` file.
2. Prints diagnostic info about the graph:
   - Placeholder (input) and output tensor shapes/dtypes
   - Parameter and buffer tensor/element counts
   - ATen op histogram from the raw exported graph
   - Edge-dialect op histogram after `exir.to_edge()`
   - A ready-to-paste `EXECUTORCH_SELECT_OPS_LIST` string for `CMakeLists.txt`,
     built from the edge-dialect `.out` op variants
3. Lowers the program to ExecuTorch and writes the resulting `.pte` buffer to
   disk, printing its final size.

### Usage

```
source ~/executorch/et-env/bin/activate
python3 util/pt_to_pte.py <input.pt> <output.pte>
```

### Input requirements

The input `.pt` file must be an `ExportedProgram` saved with
`torch.export.save()` — a plain `torch.save(model)` or
`torch.save(model.state_dict())` will **not** load, since those don't carry
the captured graph.

To produce a compatible `.pt` file (e.g. in a Colab notebook, after
training):

```python
model.eval()
example_input = (torch.randn(1, 1),)  # match your model's actual input shape/dtype
exported_program = torch.export.export(model, example_input)
torch.export.save(exported_program, "model.pt")
```

### Example output

```
Loading exported program: model.pt

=== ExportedProgram info ===

Placeholder tensors:
  x: shape=(1, 1) dtype=torch.float32

Output tensors:
  add: shape=(1, 1) dtype=torch.float32

Parameters: 2 tensors, 5 elements
Buffers:    0 tensors, 0 elements

ATen ops (raw torch.export graph):
     1  aten.add.Tensor
     1  aten.mul.Tensor

Lowering to edge dialect + ExecuTorch...

Edge dialect ops (use .out variants for EXECUTORCH_SELECT_OPS_LIST):
     1  aten.add.out
     1  aten.mul.out

EXECUTORCH_SELECT_OPS_LIST for CMake:
  EXECUTORCH_SELECT_OPS_LIST="aten::add.out,aten::mul.out"

Exported to: model.pte
Model size:  1234 bytes
```
