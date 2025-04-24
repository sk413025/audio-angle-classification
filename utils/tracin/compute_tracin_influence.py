#!/usr/bin/env python
"""
Backward compatibility layer for compute_tracin_influence.py.

This script provides backward compatibility for code that calls the 
original compute_tracin_influence.py script. It forwards all calls
to the new tracin module implementation in tracin/scripts/compute_influence.py.
"""

import os
import sys
import argparse

# Add tracin module to the path
# Get the base directory (assuming this script is in utils/tracin)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

# Import the new module functions
from tracin.scripts.compute_influence import run_compute_influence, parse_arguments


def main():
    """Main function that forwards to the new module implementation."""
    print("********************************************")
    print("* COMPATIBILITY NOTICE                     *")
    print("* Using the new TracIn module implementation *")
    print("* This compatibility layer will be removed in a future version *")
    print("********************************************")
    
    run_compute_influence()


if __name__ == "__main__":
    main() 