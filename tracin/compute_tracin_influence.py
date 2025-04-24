#!/usr/bin/env python
"""
Compute TracIn influence scores for training data.

This script is a backward compatibility layer that calls the functionality from the tracin module.
"""

import os
import sys
import argparse

# Ensure the tracin module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new module
from tracin.scripts.compute_influence import run_compute_influence, parse_arguments


def main():
    """Main function."""
    # Display compatibility notice
    print("Note: This script is maintained for backward compatibility. Consider using the tracin module directly:")
    print("  python -m tracin.scripts.compute_influence [arguments]")
    print("Starting TracIn module...\n")
    
    # Call the function from the new module
    run_compute_influence()


if __name__ == "__main__":
    main() 