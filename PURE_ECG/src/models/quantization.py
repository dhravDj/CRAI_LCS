import torch
import torch.nn as nn

def quantize_model_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Applies PyTorch native dynamic quantization (INT8) to Linear and LSTM layers.
    Reduces model parameter memory footprint by ~75% and accelerates CPU edge inference.
    """
    # Dynamic quantization operates on CPU tensors
    orig_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    model_cpu.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )

    if orig_device.type != "cpu":
        # Keep quantized model on CPU as PyTorch dynamic int8 runs on CPU backend
        pass

    return quantized_model
