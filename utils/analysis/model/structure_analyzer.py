"""
Model Structure Analyzer Module

This module provides tools for analyzing model structure:
- Loading and examining saved PyTorch models
- Identifying model components and layers
- Displaying model information
"""

import os
import torch
import sys

def check_model_structure(file_path):
    """Check the structure of a saved model file.
    
    Args:
        file_path (str): Path to the model file
        
    Returns:
        dict: Dictionary containing model information
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {
            'status': 'error',
            'message': f"File not found: {file_path}"
        }
    
    try:
        # Load the file
        data = torch.load(file_path)
        
        # Dictionary to store model information
        model_info = {
            'status': 'success',
            'file_path': file_path,
            'file_type': str(type(data))
        }
        
        # If it's a dictionary, show the keys
        if isinstance(data, dict):
            model_info['keys'] = list(data.keys())
            
            # Check if there's a 'model' key
            if 'model' in data:
                model_info['model_type'] = str(type(data['model']))
            
            # Check for other common keys
            for key in ['epoch', 'optimizer_state_dict', 'model_state_dict', 'training_history']:
                if key in data:
                    model_info[f'contains_{key}'] = True
                    model_info[f'{key}_type'] = str(type(data[key]))
        
        # If it's a module, show its structure
        elif hasattr(data, '_modules'):
            model_info['is_module'] = True
            model_info['module_structure'] = {}
            
            for name, module in data._modules.items():
                model_info['module_structure'][name] = str(type(module))
                
                # If the module has submodules, list them too
                if hasattr(module, '_modules') and module._modules:
                    model_info['module_structure'][f'{name}_submodules'] = {}
                    for subname, submodule in module._modules.items():
                        model_info['module_structure'][f'{name}_submodules'][subname] = str(type(submodule))
        
        return model_info
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error loading file: {e}"
        }

def format_model_info(model_info):
    """Format model information for display.
    
    Args:
        model_info (dict): Dictionary containing model information
        
    Returns:
        str: Formatted model information
    """
    if model_info.get('status') == 'error':
        return f"Error: {model_info.get('message', 'Unknown error')}"
    
    output = []
    output.append(f"File path: {model_info.get('file_path', 'N/A')}")
    output.append(f"File type: {model_info.get('file_type', 'N/A')}")
    
    if 'keys' in model_info:
        output.append(f"Keys: {model_info['keys']}")
    
    if 'model_type' in model_info:
        output.append(f"Model type: {model_info['model_type']}")
    
    # Display common keys
    for key in ['epoch', 'optimizer_state_dict', 'model_state_dict', 'training_history']:
        if f'contains_{key}' in model_info:
            output.append(f"Contains {key}: {model_info[f'{key}_type']}")
    
    # Display module structure
    if 'module_structure' in model_info:
        output.append("Module structure:")
        for name, type_str in model_info['module_structure'].items():
            if not name.endswith('_submodules'):
                output.append(f"  {name}: {type_str}")
                
                # Add submodules if available
                if f'{name}_submodules' in model_info['module_structure']:
                    for subname, subtype in model_info['module_structure'][f'{name}_submodules'].items():
                        output.append(f"    {subname}: {subtype}")
    
    return '\n'.join(output)

def analyze_model_structure(file_path, output_format='text'):
    """Analyze the structure of a model file and return it in the specified format.
    
    Args:
        file_path (str): Path to the model file
        output_format (str): Format of the output ('text', 'dict')
        
    Returns:
        str or dict: Formatted model information
    """
    model_info = check_model_structure(file_path)
    
    if output_format == 'text':
        return format_model_info(model_info)
    else:
        return model_info