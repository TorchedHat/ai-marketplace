#!/usr/bin/env python3
"""
Generate real torch.compile logs for testing parsers.
"""

import os
import torch
import torch._inductor.config as config
from pathlib import Path

# Configure output directory to current working directory
project_root = Path(__file__).parent
os.environ['TORCH_COMPILE_DEBUG_DIR'] = str(project_root)
os.environ['TORCH_COMPILE_DEBUG'] = '1'

# Enable all logging
os.environ['TORCH_LOGS'] = 'graph_breaks,fusion,schedule,dynamo,inductor'

# Enable IR debug output
config.trace.enabled = True
torch._dynamo.config.verbose = True

print("Generating compilation logs...")
print(f"Output will be in: {project_root}/torch_compile_debug/")
print("=" * 60)

# Test 1: Function with graph breaks
@torch.compile
def model_with_breaks(x):
    """This will cause graph breaks."""
    # Break 1: print statement
    print(f"Input shape: {x.shape}")

    y = x.relu()

    # Break 2: item() call (data-dependent)
    if y.sum().item() > 0:
        y = y + 1

    z = y * 2
    return z


# Test 2: Function that should fuse well
@torch.compile
def model_with_fusion(x):
    """This should generate fusion decisions."""
    # Should fuse: relu + add
    y = x.relu()
    y = y + 1.0

    # Should fuse: mul + add
    z = y * 2.0
    z = z + 3.0

    # Reduction - might not fuse with pointwise
    result = z.sum(dim=-1)
    return result


# Test 3: Training model (generates AOT graphs)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x).relu()


def test_compilation():
    """Run compilation tests."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Test 1: Graph breaks
    print("Test 1: Model with graph breaks")
    print("-" * 40)
    x1 = torch.randn(10, 100, device=device)
    try:
        y1 = model_with_breaks(x1)
        print(f"✓ Compiled (with breaks), output shape: {y1.shape}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")

    # Test 2: Fusion
    print("Test 2: Model with fusion opportunities")
    print("-" * 40)
    x2 = torch.randn(10, 100, device=device)
    y2 = model_with_fusion(x2)
    print(f"✓ Compiled, output shape: {y2.shape}\n")

    # Test 3: Training (AOT graphs)
    if device == 'cuda':  # AOT graphs mainly for training on CUDA
        print("Test 3: Training model (AOT graphs)")
        print("-" * 40)
        model = SimpleModel().to(device)
        compiled_model = torch.compile(model)

        x3 = torch.randn(5, 10, device=device, requires_grad=True)
        y3 = compiled_model(x3)
        loss = y3.sum()
        loss.backward()
        print(f"✓ Training compiled, loss: {loss.item():.4f}\n")

    print("=" * 60)
    print(f"Logs saved to: {project_root}/torch_compile_debug/")
    print("\nCheck latest run directory for generated IR files")


if __name__ == "__main__":
    test_compilation()
