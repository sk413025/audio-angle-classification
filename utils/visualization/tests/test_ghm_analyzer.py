"""
Tests for the GHM Analyzer module.

This module tests the functionality of the GHM analysis and visualization tools.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from unittest import mock

# Import the module to be tested
from utils.visualization.ghm_analyzer import (
    load_ghm_stats,
    find_ghm_stats_files,
    parse_epoch_batch_from_filename,
    visualize_ghm_stats,
    compare_ghm_epochs,
    analyze_ghm_training,
    calculate_ghm_statistics_from_criterion,
    visualize_ghm_live
)

class TestGHMFileUtils(unittest.TestCase):
    """Test file utilities for GHM analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample GHM stats data
        self.sample_stats = {
            'bin_counts': [10, 20, 15, 5, 2, 0, 0, 0, 0, 0],
            'bin_edges': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'mean_gradient': 0.25,
            'median_gradient': 0.2,
            'min_gradient': 0.05,
            'max_gradient': 0.45,
            'std_gradient': 0.1
        }
        
        # Save sample stats to files
        for epoch in [1, 2, 3]:
            for batch in [0, 1]:
                filename = os.path.join(self.test_dir, f"epoch{epoch}_batch{batch}_20250414_test.npy")
                np.save(filename, self.sample_stats)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)
    
    def test_load_ghm_stats(self):
        """Test loading GHM statistics from a file."""
        # Test loading valid file
        filename = os.path.join(self.test_dir, "epoch1_batch0_20250414_test.npy")
        stats = load_ghm_stats(filename)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('bin_counts', stats)
        self.assertIn('bin_edges', stats)
        self.assertIn('mean_gradient', stats)
        self.assertEqual(stats['mean_gradient'], self.sample_stats['mean_gradient'])
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            load_ghm_stats("non_existent_file.npy")
    
    def test_find_ghm_stats_files(self):
        """Test finding GHM statistics files."""
        # Test finding all files
        files = find_ghm_stats_files(self.test_dir)
        self.assertEqual(len(files), 6)  # 3 epochs * 2 batches
        
        # Test finding files for specific epoch
        files = find_ghm_stats_files(self.test_dir, epoch=2)
        self.assertEqual(len(files), 2)
        for f in files:
            self.assertIn("epoch2", os.path.basename(f))
        
        # Test finding files for specific batch
        files = find_ghm_stats_files(self.test_dir, batch=1)
        self.assertEqual(len(files), 3)
        for f in files:
            self.assertIn("batch1", os.path.basename(f))
        
        # Test finding files for specific epoch and batch
        files = find_ghm_stats_files(self.test_dir, epoch=3, batch=0)
        self.assertEqual(len(files), 1)
        self.assertIn("epoch3_batch0", os.path.basename(files[0]))
    
    def test_parse_epoch_batch_from_filename(self):
        """Test parsing epoch and batch from filename."""
        # Test valid filename
        epoch, batch = parse_epoch_batch_from_filename("epoch5_batch10_20250414_test.npy")
        self.assertEqual(epoch, 5)
        self.assertEqual(batch, 10)
        
        # Test invalid filename
        with self.assertRaises(ValueError):
            parse_epoch_batch_from_filename("invalid_filename.npy")

class TestGHMVisualization(unittest.TestCase):
    """Test GHM visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create sample GHM stats data
        self.sample_stats = {
            'bin_counts': [10, 20, 15, 5, 2, 0, 0, 0, 0, 0],
            'bin_edges': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'mean_gradient': 0.25,
            'median_gradient': 0.2,
            'min_gradient': 0.05,
            'max_gradient': 0.45,
            'std_gradient': 0.1
        }
        
        # Save sample stats to files
        for epoch in [1, 2, 3]:
            for batch in [0, 1]:
                filename = os.path.join(self.test_dir, f"epoch{epoch}_batch{batch}_20250414_test.npy")
                np.save(filename, self.sample_stats)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(self.test_dir)
    
    @mock.patch('matplotlib.pyplot.savefig')
    def test_visualize_ghm_stats(self, mock_savefig):
        """Test visualizing GHM statistics."""
        # Test visualizing from dictionary
        output_path = os.path.join(self.output_dir, "test_viz.png")
        result = visualize_ghm_stats(self.sample_stats, output_path=output_path, show_plot=False)
        
        self.assertEqual(result, output_path)
        mock_savefig.assert_called_once()
        
        # Test visualizing from file
        mock_savefig.reset_mock()
        filename = os.path.join(self.test_dir, "epoch1_batch0_20250414_test.npy")
        output_path = os.path.join(self.output_dir, "test_viz_file.png")
        result = visualize_ghm_stats(filename, output_path=output_path, show_plot=False)
        
        self.assertEqual(result, output_path)
        mock_savefig.assert_called_once()
    
    @mock.patch('matplotlib.pyplot.savefig')
    def test_compare_ghm_epochs(self, mock_savefig):
        """Test comparing GHM statistics across epochs."""
        # Test comparing multiple epochs
        output_path = os.path.join(self.output_dir, "test_compare.png")
        result = compare_ghm_epochs(
            self.test_dir, 
            epochs=[1, 2, 3], 
            batch_idx=0, 
            output_path=output_path, 
            show_plot=False
        )
        
        self.assertEqual(result, output_path)
        mock_savefig.assert_called_once()
    
    @mock.patch('utils.visualization.ghm_analyzer.visualize_ghm_stats')
    @mock.patch('utils.visualization.ghm_analyzer.compare_ghm_epochs')
    def test_analyze_ghm_training(self, mock_compare, mock_visualize):
        """Test analyzing GHM statistics across a training run."""
        # Set up mocks
        mock_visualize.return_value = "mock_viz_path.png"
        mock_compare.return_value = "mock_compare_path.png"
        
        # Test analyzing training run
        results = analyze_ghm_training(
            self.test_dir,
            output_dir=self.output_dir,
            selected_epochs=[1, 2, 3]
        )
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that visualization functions were called
        self.assertGreaterEqual(mock_visualize.call_count, 3)  # One for each epoch
        self.assertGreaterEqual(mock_compare.call_count, 1)    # At least one comparison

class TestGHMLiveVisualization(unittest.TestCase):
    """Test live GHM visualization during training."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample outputs and targets
        self.outputs1 = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        self.outputs2 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        self.targets = np.array([1, 1, 0, 0, 0])
        
        # Create mock criterion
        self.criterion = mock.MagicMock()
        self.criterion.edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.criterion.bins = 10
    
    @mock.patch('utils.ghm_utils.calculate_ghm_statistics')
    def test_calculate_ghm_statistics_from_criterion(self, mock_calc):
        """Test calculating GHM statistics from a criterion."""
        # Mock the return value
        sample_stats = {
            'bin_counts': [10, 20, 15, 5, 2, 0, 0, 0, 0, 0],
            'bin_edges': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'mean_gradient': 0.25,
            'median_gradient': 0.2,
            'min_gradient': 0.05,
            'max_gradient': 0.45,
            'std_gradient': 0.1
        }
        mock_calc.return_value = sample_stats
        
        # Call the function
        stats = calculate_ghm_statistics_from_criterion(
            self.criterion,
            self.outputs1,
            self.outputs2,
            self.targets
        )
        
        # Check results
        self.assertEqual(stats, sample_stats)
        mock_calc.assert_called_once_with(
            self.criterion,
            self.outputs1,
            self.outputs2,
            self.targets,
            save_dir=None,
            name=None
        )
    
    @mock.patch('utils.visualization.ghm_analyzer.calculate_ghm_statistics_from_criterion')
    @mock.patch('utils.visualization.ghm_analyzer.visualize_ghm_stats')
    def test_visualize_ghm_live(self, mock_visualize, mock_calc):
        """Test live visualization of GHM statistics during training."""
        # Mock the return values
        sample_stats = {
            'bin_counts': [10, 20, 15, 5, 2, 0, 0, 0, 0, 0],
            'bin_edges': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'mean_gradient': 0.25,
            'median_gradient': 0.2,
            'min_gradient': 0.05,
            'max_gradient': 0.45,
            'std_gradient': 0.1
        }
        mock_calc.return_value = sample_stats
        mock_visualize.return_value = "mock_viz_path.png"
        
        # Call the function
        output_path = "test_viz_live.png"
        result = visualize_ghm_live(
            self.criterion,
            self.outputs1,
            self.outputs2,
            self.targets,
            output_path=output_path,
            show_plot=False
        )
        
        # Check results
        self.assertEqual(result, "mock_viz_path.png")
        mock_calc.assert_called_once_with(
            self.criterion,
            self.outputs1,
            self.outputs2,
            self.targets
        )
        mock_visualize.assert_called_once_with(
            sample_stats,
            output_path=output_path,
            show_plot=False
        )

if __name__ == '__main__':
    unittest.main() 