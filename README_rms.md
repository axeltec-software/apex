# RMS norm optimizations

In order to use the optimized version of the rms kernel, a new class has been added. New class added [here](apex/normalization/). Example usage:

```python
    import apex
    import torch
    apex_rms = apex.normalization.FusedRMSNormOptimized
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.rand(x_shape, dtype=dtype, device=device)
    apex_func = apex_rms(w_shape, eps).to(dtype)
    # run pure rms
    y_apex = apex_func(x)
    # run with residual
    y_apex = apex_func(x, residual)

```