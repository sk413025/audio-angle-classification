#!/usr/bin/env python
"""
Import Updater for Visualization Scripts

This script updates the visualization scripts to use direct imports
from the parent directory for the project's modules.
"""

import os
import glob
import re

def update_file(file_path):
    """Update import statements in the given file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if we need to update this file
    if "IMPORT_CONFIG_COMPLETE" in content:
        print(f"Skipping already updated file: {os.path.basename(file_path)}")
        return
    
    # Add the parent directory path setup at the beginning of imports
    path_setup = """# IMPORT_CONFIG_COMPLETE
# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
"""
    
    # Find where to insert the path setup
    import_section_end = content.find("import sys")
    if import_section_end > 0:
        # Find the next line after "import sys"
        next_line = content.find("\n", import_section_end)
        if next_line > 0:
            # Insert the path setup after import sys
            content = content[:next_line+1] + path_setup + content[next_line+1:]
    
    # Replace any existing path setup code
    path_setup_pattern = re.compile(r'# Add (parent|current) directory.*?sys\.path\..*?\)', re.DOTALL)
    if path_setup_pattern.search(content):
        content = path_setup_pattern.sub('', content)
    
    # Remove try/except import blocks
    import_error_pattern = re.compile(r'try:.*?except ImportError:.*?sys\.exit\(1\)', re.DOTALL)
    if import_error_pattern.search(content):
        # Replace with simple direct imports
        direct_imports = """
# Direct imports from parent directory
from datasets import SpectrogramDatasetWithMaterial
import config
"""
        content = import_error_pattern.sub(direct_imports, content)
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated: {os.path.basename(file_path)}")

def create_init_file():
    """Create or update the __init__.py file"""
    init_path = os.path.join(os.path.dirname(__file__), "__init__.py")
    
    content = """\"\"\"
Visualizations Package for Angle Classification Project

This package contains various visualization utilities for the angle classification project.
\"\"\"

# Add the parent directory to the Python path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
"""
    
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Created/updated __init__.py file")

def create_run_script():
    """Create a helper script to run visualization tools"""
    run_path = os.path.join(os.path.dirname(__file__), "run_visualization.py")
    
    content = """#!/usr/bin/env python
\"\"\"
Visualization Runner

Helper script to run visualization tools from the correct context.
Usage: python run_visualization.py <visualization_script> [args...]
\"\"\"

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
"""
    
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Created run_visualization.py helper script")

def main():
    """Update all visualization files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_files = glob.glob(os.path.join(script_dir, "*.py"))
    
    # Skip non-visualization files
    skip_files = ["setup_imports.py", "update_imports.py", "__init__.py", "run_visualization.py"]
    viz_files = [f for f in viz_files if os.path.basename(f) not in skip_files]
    
    print(f"Found {len(viz_files)} visualization files to update")
    
    # Create package initialization files
    create_init_file()
    create_run_script()
    
    # Update each visualization file
    for file_path in viz_files:
        update_file(file_path)
    
    print("\nSetup complete! To run visualizations, use:")
    print("python visualizations/run_visualization.py <visualization_script> [args...]")

if __name__ == "__main__":
    main() 