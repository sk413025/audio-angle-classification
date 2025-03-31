#!/usr/bin/env python
"""
Visualization Runner

Helper script to run visualization tools from the correct context.
Usage: python run_visualization.py <visualization_script> [args...]
"""

import os
import sys
import importlib
import subprocess
import glob

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_visualization.py <visualization_script> [args...]")
        print("Available visualization scripts:")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_scripts = [os.path.basename(f) for f in glob.glob(os.path.join(script_dir, "*.py"))
                     if os.path.basename(f) not in ["__init__.py", "run_visualization.py", 
                                                   "setup_imports.py", "update_imports.py"]]
        for script in sorted(viz_scripts):
            print(f"  - {script}")
        return
    
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Get the visualization script to run
    viz_script = sys.argv[1]
    if not viz_script.endswith('.py'):
        viz_script += '.py'
    
    # Check if the script exists
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), viz_script)
    if not os.path.exists(script_path):
        print(f"Error: {viz_script} not found in visualizations directory")
        return
    
    # Run the script with the remaining arguments
    cmd = [sys.executable, script_path] + sys.argv[2:]
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, env=env)
    except Exception as e:
        print(f"Error running {viz_script}: {e}")

if __name__ == "__main__":
    main()
