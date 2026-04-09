"""
Dataset proxy module - delegates to vendor/mkb/dataset.

This ensures a single authoritative source for operator definitions.
"""
import sys
import os

# Add vendor path if not already present
vendor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vendor')
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

# Import from authoritative source
from mkb.dataset import dataset, category2exampleop

__all__ = ['dataset', 'category2exampleop']