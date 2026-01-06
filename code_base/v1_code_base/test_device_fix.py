#!/usr/bin/env python3
"""
Test script to verify device/dtype fix for MultiModalAlignmentModel.
Run this to ensure no MPS/CPU device mismatches occur.
"""

import torch
from imports.model import MultiModalAlignmentModel, FullAlignmentGraphWrapper

def main():
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create model (use float32 for MPS compatibility)
    dtype = torch.float32 if device.type == "mps" else torch.float16
    print(f"Using dtype: {dtype}")
    
    model = MultiModalAlignmentModel(
        d_shared=512,
        d_latent=512,
        d_align=1024,
        num_latents=32,
        num_layers=2,
        num_heads=4,
        use_perceiver=True,
        dtype=dtype,
        device=device,
    )
    model.eval()
    
    print("\n=== Model created ===")
    
    # Create wrapper
    wrapper = FullAlignmentGraphWrapper(model).to(device)
    wrapper.eval()
    
    print("=== Wrapper created ===")
    
    # Before running forward, explicitly move all components to device
    print(f"\n=== Moving all components to {device} ===")
    
    if hasattr(model.vision_encoder, 'model'):
        model.vision_encoder.model.to(device)
        print(f"✓ Moved vision_encoder.model to {device}")
    
    if hasattr(model.audio_encoder, 'model'):
        model.audio_encoder.model.to(device)
        print(f"✓ Moved audio_encoder.model to {device}")
    
    if model.vision_adapter is not None:
        model.vision_adapter.to(device)
        print(f"✓ Moved vision_adapter to {device}")
    
    if model.audio_adapter is not None:
        model.audio_adapter.to(device)
        print(f"✓ Moved audio_adapter to {device}")
    
    if model.perceiver is not None:
        model.perceiver.to(device)
        print(f"✓ Moved perceiver to {device}")
    
    if model.projector is not None:
        model.projector.to(device)
        print(f"✓ Moved projector to {device}")
    
    # Create dummy inputs
    B = 2
    dummy_pixel_values = torch.randn(B, 3, 224, 224, device=device)
    dummy_waveforms = torch.randn(B, 16000 * 3, device=device)
    
    print(f"\n=== Inputs created ===")
    print(f"dummy_pixel_values: {dummy_pixel_values.device}, {dummy_pixel_values.dtype}")
    print(f"dummy_waveforms: {dummy_waveforms.device}, {dummy_waveforms.dtype}")
    
    # Verify model parameters
    print(f"\n=== Checking model parameter devices (first 5) ===")
    count = 0
    for name, p in model.named_parameters():
        print(f"{name}: {p.device}, {p.dtype}")
        count += 1
        if count >= 5:
            break
    
    # Run forward pass
    print(f"\n=== Running forward pass ===")
    with torch.no_grad():
        try:
            z = wrapper(dummy_pixel_values, dummy_waveforms)
            print(f"✓ Success! Output shape: {z.shape}")
            print(f"✓ Output device: {z.device}, dtype: {z.dtype}")
            return 0
        except RuntimeError as e:
            print(f"✗ Error: {e}")
            return 1

if __name__ == "__main__":
    exit(main())
