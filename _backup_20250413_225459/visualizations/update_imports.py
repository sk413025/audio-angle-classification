#!/usr/bin/env python
"""
Import Path Updater for Visualization Files

This script updates import statements in all visualization files to handle
the new directory structure where visualization files are in their own folder.
"""

import os
import re
import glob

def update_imports(file_path):
    """
    Update import statements in a visualization file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip files that have already been updated
    if "current_dir = os.path.dirname(os.path.abspath(__file__))" in content:
        print(f"Skipping already updated file: {file_path}")
        return
    
    # Pattern to find import sections
    import_pattern = re.compile(r'(# Add current directory to path.*?except ImportError.*?sys\.exit\(1\))', re.DOTALL)
    
    # New import block
    new_import_block = """# Add parent directory to path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from parent directory
try:
    from datasets import SpectrogramDatasetWithMaterial
    import config
    print("Successfully imported modules from parent directory")
except ImportError:
    try:
        # Try from angle_classification_deg6 package
        from angle_classification_deg6.datasets import SpectrogramDatasetWithMaterial
        from angle_classification_deg6 import config
        print("Successfully imported modules from angle classification subdirectory")
    except ImportError:
        print("Could not find required modules. Please check file paths.")
        print("Please make sure datasets.py and config.py are in the parent directory.")
        sys.exit(1)"""
    
    # Replace import section
    updated_content = import_pattern.sub(new_import_block, content)
    
    # If pattern wasn't found, try inserting after common imports
    if updated_content == content:
        import_position = content.find("import sys")
        if import_position > 0:
            # Find the end of the import section (the next empty line after import sys)
            after_import = content.find("\n\n", import_position)
            if after_import > 0:
                updated_content = content[:after_import+2] + new_import_block + content[after_import+2:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    print(f"Updated imports in {file_path}")

def main():
    """Update imports in all visualization files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_files = glob.glob(os.path.join(script_dir, "*.py"))
    
    # Skip this updater script
    viz_files = [f for f in viz_files if os.path.basename(f) != "update_imports.py" 
                 and os.path.basename(f) != "__init__.py"]
    
    print(f"Found {len(viz_files)} visualization files to update")
    
    for file_path in viz_files:
        update_imports(file_path)
    
    print("Import update complete!")

if __name__ == "__main__":
    main() 