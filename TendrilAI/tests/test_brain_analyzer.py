#!/usr/bin/env python3
"""
Unit Tests for MRI Brain Analysis System
=======================================

This module contains unit tests for the brain analyzer system.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_analyzer import BrainAnalyzer, analyze_mri_scan
from brain_analyzer.segmentation import BrainSegmentation, UNet3D
from brain_analyzer.abnormality_detection import AbnormalityDetector, AbnormalityDetector3D
from brain_analyzer.visualization import BrainVisualizer
from brain_analyzer.reporting import ReportGenerator

class TestBrainAnalyzer(unittest.TestCase):
    """Test cases for the main BrainAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BrainAnalyzer(device='cpu')
        
        # Create sample data
        self.sample_data = np.random.rand(64, 64, 32).astype(np.float32)
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        self.temp_file.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.segmentation)
        self.assertIsNotNone(self.analyzer.abnormality_detector)
        self.assertIsNotNone(self.analyzer.visualizer)
        self.assertIsNotNone(self.analyzer.report_generator)
    
    def test_analyzer_device_selection(self):
        """Test device selection."""
        # Test auto device selection
        analyzer_auto = BrainAnalyzer(device='auto')
        self.assertIsNotNone(analyzer_auto)
        
        # Test CPU device selection
        analyzer_cpu = BrainAnalyzer(device='cpu')
        self.assertIsNotNone(analyzer_cpu)
    
    def test_scan_loading(self):
        """Test scan loading functionality."""
        # This would require actual MRI files to test properly
        # For now, we'll test the method exists
        self.assertTrue(hasattr(self.analyzer, '_load_scan'))
        self.assertTrue(hasattr(self.analyzer, '_load_nifti'))
        self.assertTrue(hasattr(self.analyzer, '_load_dicom'))


class TestBrainSegmentation(unittest.TestCase):
    """Test cases for the brain segmentation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.segmentation = BrainSegmentation(device='cpu')
        self.sample_data = np.random.rand(64, 64, 32).astype(np.float32)
    
    def test_segmentation_initialization(self):
        """Test that segmentation initializes correctly."""
        self.assertIsNotNone(self.segmentation)
        self.assertIsNotNone(self.segmentation.region_labels)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed_data = self.segmentation._preprocess_data(self.sample_data)
        
        # Check that data is normalized to [0, 1]
        self.assertGreaterEqual(processed_data.min(), 0)
        self.assertLessEqual(processed_data.max(), 1)
        
        # Check that data is 3D
        self.assertEqual(len(processed_data.shape), 3)
    
    def test_rule_based_segmentation(self):
        """Test rule-based segmentation."""
        processed_data = self.segmentation._preprocess_data(self.sample_data)
        mask = self.segmentation._rule_based_segmentation(processed_data)
        
        # Check that mask has correct shape
        self.assertEqual(mask.shape, processed_data.shape)
        
        # Check that mask contains expected labels
        unique_labels = np.unique(mask)
        self.assertTrue(all(label in [0, 1, 2, 3] for label in unique_labels))
    
    def test_brain_extraction(self):
        """Test brain extraction."""
        brain_mask = self.segmentation._extract_brain(self.sample_data)
        
        # Check that brain mask is binary
        self.assertTrue(np.all(np.logical_or(brain_mask == 0, brain_mask == 1)))
        
        # Check that brain mask has correct shape
        self.assertEqual(brain_mask.shape, self.sample_data.shape)
    
    def test_tissue_classification(self):
        """Test tissue classification."""
        brain_mask = self.segmentation._extract_brain(self.sample_data)
        
        gray_matter = self.segmentation._classify_gray_matter(self.sample_data, brain_mask)
        white_matter = self.segmentation._classify_white_matter(self.sample_data, brain_mask)
        csf = self.segmentation._classify_csf(self.sample_data, brain_mask)
        
        # Check that classifications are binary
        self.assertTrue(np.all(np.logical_or(gray_matter == 0, gray_matter == 1)))
        self.assertTrue(np.all(np.logical_or(white_matter == 0, white_matter == 1)))
        self.assertTrue(np.all(np.logical_or(csf == 0, csf == 1)))
        
        # Check that classifications have correct shapes
        self.assertEqual(gray_matter.shape, self.sample_data.shape)
        self.assertEqual(white_matter.shape, self.sample_data.shape)
        self.assertEqual(csf.shape, self.sample_data.shape)


class TestAbnormalityDetection(unittest.TestCase):
    """Test cases for the abnormality detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AbnormalityDetector(device='cpu')
        self.sample_data = np.random.rand(64, 64, 32).astype(np.float32)
        self.segmentation_results = {
            'segmentation_mask': np.random.randint(0, 4, (64, 64, 32)),
            'measurements': {
                'total_brain_volume': 1000,
                'csf_volume': 100
            }
        }
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.abnormality_types)
    
    def test_preprocess_data(self):
        """Test data preprocessing for abnormality detection."""
        brain_mask = np.ones_like(self.sample_data)
        processed_data = self.detector._preprocess_data(self.sample_data, brain_mask)
        
        # Check that data is normalized
        self.assertGreaterEqual(processed_data.min(), 0)
        self.assertLessEqual(processed_data.max(), 1)
        
        # Check that data has correct shape
        self.assertEqual(processed_data.shape, self.sample_data.shape)
    
    def test_tumor_detection(self):
        """Test tumor detection."""
        result = self.detector._detect_tumors(self.sample_data)
        
        # Check that result has expected structure
        self.assertIn('detected', result)
        self.assertIn('confidence', result)
        
        # Check that confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_lesion_detection(self):
        """Test lesion detection."""
        result = self.detector._detect_lesions(self.sample_data)
        
        # Check that result has expected structure
        self.assertIn('detected', result)
        self.assertIn('confidence', result)
        
        # Check that confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_atrophy_detection(self):
        """Test atrophy detection."""
        result = self.detector._detect_atrophy(self.sample_data, self.segmentation_results)
        
        # Check that result has expected structure
        self.assertIn('detected', result)
        self.assertIn('confidence', result)
        
        # Check that confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_mass_effect_detection(self):
        """Test mass effect detection."""
        result = self.detector._detect_mass_effect(self.sample_data, self.segmentation_results)
        
        # Check that result has expected structure
        self.assertIn('detected', result)
        self.assertIn('confidence', result)
        
        # Check that confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_edema_detection(self):
        """Test edema detection."""
        result = self.detector._detect_edema(self.sample_data, self.segmentation_results)
        
        # Check that result has expected structure
        self.assertIn('detected', result)
        self.assertIn('confidence', result)
        
        # Check that confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)


class TestBrainVisualizer(unittest.TestCase):
    """Test cases for the brain visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = BrainVisualizer()
        
        # Create mock results
        self.mock_results = type('MockResults', (), {
            'scan_data': {'data': np.random.rand(64, 64, 32)},
            'segmentation_results': {
                'segmentation_mask': np.random.randint(0, 4, (64, 64, 32)),
                'region_mask': np.random.randint(0, 14, (64, 64, 32))
            },
            'abnormality_results': {
                'abnormalities': {}
            }
        })()
    
    def test_visualizer_initialization(self):
        """Test that visualizer initializes correctly."""
        self.assertIsNotNone(self.visualizer)
        self.assertIsNotNone(self.visualizer.colors)
    
    def test_create_visualizations(self):
        """Test visualization creation."""
        visualizations = self.visualizer.create_visualizations(self.mock_results)
        
        # Check that visualizations are created
        self.assertIsInstance(visualizations, dict)
        self.assertGreater(len(visualizations), 0)
    
    def test_slice_visualizations(self):
        """Test slice visualization creation."""
        fig = self.visualizer._create_slice_visualizations(self.mock_results)
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_3d_surface_visualization(self):
        """Test 3D surface visualization creation."""
        fig = self.visualizer._create_3d_surface_visualization(self.mock_results)
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_segmentation_overlay(self):
        """Test segmentation overlay creation."""
        fig = self.visualizer._create_segmentation_overlay(self.mock_results)
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_abnormality_visualization(self):
        """Test abnormality visualization creation."""
        fig = self.visualizer._create_abnormality_visualization(self.mock_results)
        
        # Check that figure is created
        self.assertIsNotNone(fig)


class TestReportGenerator(unittest.TestCase):
    """Test cases for the report generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.report_generator = ReportGenerator()
        
        # Create mock results
        self.mock_results = type('MockResults', (), {
            'scan_path': 'test_scan.nii.gz',
            'scan_data': {
                'shape': (64, 64, 32),
                'voxel_size': [1, 1, 1],
                'orientation': 'RAS'
            },
            'segmentation_results': {
                'measurements': {
                    'total_brain_volume': 1000,
                    'gray_matter_volume': 600,
                    'white_matter_volume': 300,
                    'csf_volume': 100
                },
                'regions': {
                    'frontal_lobe': {'volume': 200, 'percentage': 20},
                    'parietal_lobe': {'volume': 150, 'percentage': 15}
                }
            },
            'abnormality_results': {
                'abnormalities': {},
                'measurements': {
                    'total_abnormalities': 0,
                    'severity_score': 0.0
                }
            },
            'visualizations': {},
            'report': ""
        })()
    
    def test_report_generator_initialization(self):
        """Test that report generator initializes correctly."""
        self.assertIsNotNone(self.report_generator)
        self.assertIsNotNone(self.report_generator.report_templates)
    
    def test_report_generation(self):
        """Test report generation."""
        report = self.report_generator.generate_report(self.mock_results)
        
        # Check that report is generated
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_data_extraction(self):
        """Test data extraction for reports."""
        data = self.report_generator._extract_report_data(self.mock_results)
        
        # Check that data is extracted correctly
        self.assertIn('scan_info', data)
        self.assertIn('segmentation_measurements', data)
        self.assertIn('abnormalities', data)
        self.assertIn('brain_regions', data)
    
    def test_template_filling(self):
        """Test template filling."""
        template = "Test template with {{TOTAL_BRAIN_VOLUME}}"
        data = self.report_generator._extract_report_data(self.mock_results)
        
        filled_template = self.report_generator._fill_template(template, data)
        
        # Check that template is filled
        self.assertNotIn('{{TOTAL_BRAIN_VOLUME}}', filled_template)
        self.assertIn('1000.00', filled_template)
    
    def test_abnormality_summary_generation(self):
        """Test abnormality summary generation."""
        abnormalities = {}
        measurements = {'total_abnormalities': 0, 'severity_score': 0.0}
        
        summary = self.report_generator._generate_abnormality_summary(abnormalities, measurements)
        
        # Check that summary is generated
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_region_summary_generation(self):
        """Test region summary generation."""
        regions = {
            'frontal_lobe': {'volume': 200, 'percentage': 20},
            'parietal_lobe': {'volume': 150, 'percentage': 15}
        }
        
        summary = self.report_generator._generate_region_summary(regions)
        
        # Check that summary is generated
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_recommendations_generation(self):
        """Test recommendations generation."""
        abnormalities = {}
        measurements = {'severity_score': 0.0}
        
        recommendations = self.report_generator._generate_recommendations(abnormalities, measurements)
        
        # Check that recommendations are generated
        self.assertIsInstance(recommendations, str)
        self.assertGreater(len(recommendations), 0)


class TestNeuralNetworks(unittest.TestCase):
    """Test cases for neural network models."""
    
    def test_unet3d_initialization(self):
        """Test UNet3D initialization."""
        model = UNet3D(in_channels=1, out_channels=4)
        
        # Check that model is created
        self.assertIsNotNone(model)
        
        # Check that model has expected structure
        self.assertIsNotNone(model.downs)
        self.assertIsNotNone(model.ups)
        self.assertIsNotNone(model.bottleneck)
        self.assertIsNotNone(model.final_conv)
    
    def test_unet3d_forward_pass(self):
        """Test UNet3D forward pass."""
        model = UNet3D(in_channels=1, out_channels=4)
        
        # Create sample input
        x = torch.randn(1, 1, 32, 32, 16)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 4, 32, 32, 16))
    
    def test_abnormality_detector3d_initialization(self):
        """Test AbnormalityDetector3D initialization."""
        model = AbnormalityDetector3D(in_channels=1, num_classes=6)
        
        # Check that model is created
        self.assertIsNotNone(model)
        
        # Check that model has expected structure
        self.assertIsNotNone(model.features)
        self.assertIsNotNone(model.classifier)
    
    def test_abnormality_detector3d_forward_pass(self):
        """Test AbnormalityDetector3D forward pass."""
        model = AbnormalityDetector3D(in_channels=1, num_classes=6)
        
        # Create sample input
        x = torch.randn(1, 1, 32, 32, 16)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 6))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBrainAnalyzer,
        TestBrainSegmentation,
        TestAbnormalityDetection,
        TestBrainVisualizer,
        TestReportGenerator,
        TestNeuralNetworks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Import torch for neural network tests
    import torch
    
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1) 