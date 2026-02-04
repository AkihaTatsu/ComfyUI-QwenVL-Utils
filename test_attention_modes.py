#!/usr/bin/env python
"""Test script for attention mode selection"""

import sys
from attention import ATTENTION_MODES, resolve_attention_mode, get_attention_availability

print("="*70)
print("Available Attention Modes")
print("="*70)
for i, mode in enumerate(ATTENTION_MODES, 1):
    print(f"{i}. {mode}")

print("\n" + "="*70)
print("Hardware Capability Check")
print("="*70)
avail = get_attention_availability()
for backend, available in avail.items():
    status = "✓ Available" if available else "✗ Not Available"
    print(f"{backend:20s}: {status}")

print("\n" + "="*70)
print("Testing Each Mode")
print("="*70)

test_modes = ['auto', 'sdpa_flash', 'sdpa_math', 'flash_attention_2', 'sage_attention', 'eager']

for mode in test_modes:
    print(f"\n--- Mode: {mode} ---")
    try:
        result = resolve_attention_mode(mode)
        impl, sage = result
        print(f"✓ Resolved to: attn_implementation='{impl}', sage_attention={sage}")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*70)
print("Test Complete")
print("="*70)
