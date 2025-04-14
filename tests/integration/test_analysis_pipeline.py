"""
Integration Test for Analysis Pipeline

This module tests the end-to-end analysis workflow to ensure all components
work together correctly after refactoring.
"""

import os
import sys
import unittest
import tempfile
import subprocess
import numpy as np
import pickle

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestAnalysisPipeline(unittest.TestCase):
    """Integration test suite for the analysis pipeline."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory structure
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock directory structure
        self.saved_models_dir = os.path.join(self.temp_dir.name, 'saved_models')
        self.results_dir = os.path.join(self.temp_dir.name, 'results')
        
        # Create frequency directories
        for freq in ['500hz', '1000hz', '3000hz']:
            # Create training directories
            train_dir = os.path.join(self.saved_models_dir, f'ghm_{freq}_20250101_000000')
            os.makedirs(train_dir, exist_ok=True)
            
            # Create ghm_stats directory
            stats_dir = os.path.join(train_dir, 'ghm_stats')
            os.makedirs(stats_dir, exist_ok=True)
            
            # Create plots directory
            plots_dir = os.path.join(train_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create mock training history file
            self.create_mock_training_history(train_dir)
            
            # Create mock stats files
            self.create_mock_stats_files(stats_dir)
        
        # Create mock model file
        self.create_mock_model_file()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def create_mock_training_history(self, directory):
        """Create a mock training history file."""
        history = {
            'epoch': list(range(1, 11)),
            'train_loss_main': np.random.random(10).tolist(),
            'val_loss_main': np.random.random(10).tolist(),
            'train_accuracy': (np.random.random(10) * 20 + 70).tolist(),  # 70-90%
            'val_accuracy': (np.random.random(10) * 20 + 70).tolist(),    # 70-90%
        }
        
        with open(os.path.join(directory, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
    
    def create_mock_stats_files(self, directory):
        """Create mock GHM stats files."""
        for epoch in range(1, 4):
            for batch in range(2):
                # Create a stats dictionary
                stats = {
                    'bin_counts': np.random.randint(1, 100, size=10),
                    'bin_edges': np.linspace(0, 1, 11),  # 10 bins need 11 edges
                    'mean_gradient': np.random.random(),
                    'median_gradient': np.random.random(),
                    'min_gradient': np.random.random() * 0.1,
                    'max_gradient': np.random.random() * 10,
                    'std_gradient': np.random.random()
                }
                
                # Save it to a file
                filename = f"epoch{epoch}_batch{batch}_stats.npy"
                file_path = os.path.join(directory, filename)
                np.save(file_path, stats)
    
    def create_mock_model_file(self):
        """Create a mock model file for testing."""
        import torch
        import numpy as np
        
        # Simple dictionary to mimic a model checkpoint
        mock_model = {
            'epoch': 10,
            'model_state_dict': {
                'layer1.weight': torch.randn(10, 10),
                'layer1.bias': torch.randn(10)
            },
            'optimizer_state_dict': {'param_groups': []},
            'training_history': {
                'epoch': list(range(1, 11)),
                'train_loss_main': np.random.random(10).tolist(),
                'val_loss_main': np.random.random(10).tolist(),
                'train_accuracy': (np.random.random(10) * 20 + 70).tolist(),
                'val_accuracy': (np.random.random(10) * 20 + 70).tolist(),
            }
        }
        
        model_dir = os.path.join(self.saved_models_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(mock_model, os.path.join(model_dir, 'mock_model.pt'))
    
    def test_analyze_ghm_script(self):
        """Test the analyze_ghm.py script."""
        from scripts.analyze_ghm import main as analyze_ghm_main
        import sys
        
        # Save original argv
        original_argv = sys.argv.copy()
        
        try:
            # Set argv for the script
            sys.argv = [
                'analyze_ghm.py',
                '--base-dir', self.saved_models_dir,
                '--output-dir', os.path.join(self.results_dir, 'ghm'),
                '--frequency', '1000hz'
            ]
            
            # Run the script
            analyze_ghm_main()
            
            # Check if output directory was created
            output_dir = os.path.join(self.results_dir, 'ghm', '1000hz')
            self.assertTrue(os.path.exists(output_dir), f"Output directory {output_dir} not created")
            
            # Check for some expected output files (updated paths)
            expected_files = [
                'early_epoch_ghm.png',
                'mid_epoch_ghm.png',
                'late_epoch_ghm.png',
                'epoch_comparison.png'
            ]
            
            for filename in expected_files:
                file_path = os.path.join(output_dir, '1000hz_ghm_default_analysis', filename)
                self.assertTrue(os.path.exists(file_path), f"Expected file {file_path} not found")
        
        finally:
            # Restore original argv
            sys.argv = original_argv
    
    def test_check_model_script(self):
        """Test the check_model.py script."""
        from scripts.check_model import main as check_model_main
        import sys
        
        # Save original argv
        original_argv = sys.argv.copy()
        
        try:
            # Set up a temporary output file
            output_file = os.path.join(self.temp_dir.name, 'model_check_output.txt')
            
            # Set argv for the script
            sys.argv = [
                'check_model.py',
                '--model-path', os.path.join(self.saved_models_dir, 'models', 'mock_model.pt'),
                '--output-file', output_file
            ]
            
            # Run the script
            check_model_main()
            
            # Check if output file was created
            self.assertTrue(os.path.exists(output_file), f"Output file {output_file} not created")
            
            # Check if the output file contains expected content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn('model_state_dict', content)
                self.assertIn('optimizer_state_dict', content)
                self.assertIn('training_history', content)
        
        finally:
            # Restore original argv
            sys.argv = original_argv

if __name__ == '__main__':
    unittest.main()