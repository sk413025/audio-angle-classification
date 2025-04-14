import os
import torch
import sys

def check_model_file(file_path):
    """Check the structure of a saved model file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        # Load the file
        data = torch.load(file_path)
        
        # Check the type
        print(f"File type: {type(data)}")
        
        # If it's a dictionary, show the keys
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            # Check if there's a 'model' key
            if 'model' in data:
                print(f"Model type: {type(data['model'])}")
            
            # Check for other common keys
            for key in ['epoch', 'optimizer_state_dict', 'model_state_dict', 'training_history']:
                if key in data:
                    print(f"Contains {key}: {type(data[key])}")
        
        # If it's a module, show its structure
        elif hasattr(data, '_modules'):
            print(f"Module structure:")
            for name, module in data._modules.items():
                print(f"  {name}: {type(module)}")
                
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_structure.py <model_file_path>")
        sys.exit(1)
    
    check_model_file(sys.argv[1]) 