import warnings
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )

    return quantized_model
