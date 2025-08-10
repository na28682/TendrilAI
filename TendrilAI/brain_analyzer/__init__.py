"""
MRI Brain Analysis System
=========================

A comprehensive system for analyzing MRI brain scans to recognize brain regions
and detect abnormalities using deep learning techniques.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import nibabel as nib
from pathlib import Path

from .segmentation import BrainSegmentation
from .abnormality_detection import AbnormalityDetector
from .visualization import BrainVisualizer
from .reporting import ReportGenerator

warnings.filterwarnings('ignore')

class BrainAnalyzer:
    """
    Main class for MRI brain analysis.
    
    This class orchestrates the entire analysis pipeline including:
    - Brain region segmentation
    - Abnormality detection
    - 3D visualization
    - Report generation
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize the brain analyzer.
        
        Args:
            model_path: Path to pre-trained models directory
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Initialize components
        self.segmentation = BrainSegmentation(model_path=self.model_path, device=device)
        self.abnormality_detector = AbnormalityDetector(model_path=self.model_path, device=device)
        self.visualizer = BrainVisualizer()
        self.report_generator = ReportGenerator()
        
        # Analysis results
        self.current_results = None
        
    def analyze_scan(self, 
                    scan_path: Union[str, Path],
                    output_dir: Optional[str] = None,
                    generate_visualizations: bool = True,
                    generate_report: bool = True) -> 'AnalysisResults':
        """
        Perform comprehensive analysis of an MRI brain scan.
        
        Args:
            scan_path: Path to the MRI scan file (.nii.gz, .dcm, etc.)
            output_dir: Directory to save results (optional)
            generate_visualizations: Whether to create 3D visualizations
            generate_report: Whether to generate analysis report
            
        Returns:
            AnalysisResults object containing all analysis results
        """
        print(f"🔍 Analyzing MRI scan: {scan_path}")
        
        # Load and preprocess scan
        scan_data = self._load_scan(scan_path)
        
        # Perform brain segmentation
        print("🧠 Performing brain segmentation...")
        segmentation_results = self.segmentation.segment_brain(scan_data)
        
        # Detect abnormalities
        print("⚠️ Detecting abnormalities...")
        abnormality_results = self.abnormality_detector.detect_abnormalities(
            scan_data, segmentation_results
        )
        
        # Create results object
        self.current_results = AnalysisResults(
            scan_path=scan_path,
            scan_data=scan_data,
            segmentation_results=segmentation_results,
            abnormality_results=abnormality_results
        )
        
        # Generate visualizations if requested
        if generate_visualizations:
            print("📊 Generating visualizations...")
            self.current_results.visualizations = self.visualizer.create_visualizations(
                self.current_results
            )
        
        # Generate report if requested
        if generate_report:
            print("📝 Generating analysis report...")
            self.current_results.report = self.report_generator.generate_report(
                self.current_results
            )
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(output_dir)
        
        print("✅ Analysis complete!")
        return self.current_results
    
    def _load_scan(self, scan_path: Union[str, Path]) -> Dict:
        """
        Load and preprocess MRI scan data.
        
        Args:
            scan_path: Path to the scan file
            
        Returns:
            Dictionary containing scan data and metadata
        """
        scan_path = Path(scan_path)
        
        if not scan_path.exists():
            raise FileNotFoundError(f"Scan file not found: {scan_path}")
        
        # Load based on file extension
        if scan_path.suffix in ['.nii', '.nii.gz']:
            return self._load_nifti(scan_path)
        elif scan_path.suffix in ['.dcm', '.dicom']:
            return self._load_dicom(scan_path)
        else:
            raise ValueError(f"Unsupported file format: {scan_path.suffix}")
    
    def _load_nifti(self, scan_path: Path) -> Dict:
        """Load NIfTI format scan."""
        img = nib.load(str(scan_path))
        data = img.get_fdata()
        
        return {
            'data': data,
            'affine': img.affine,
            'header': img.header,
            'shape': data.shape,
            'voxel_size': img.header.get_zooms(),
            'orientation': nib.aff2axcodes(img.affine)
        }
    
    def _load_dicom(self, scan_path: Path) -> Dict:
        """Load DICOM format scan."""
        import pydicom
        
        if scan_path.is_dir():
            # Load DICOM series
            dicom_files = sorted(scan_path.glob('*.dcm'))
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {scan_path}")
            
            # Load first file to get metadata
            ds = pydicom.dcmread(str(dicom_files[0]))
            
            # Load all slices
            slices = []
            for dcm_file in dicom_files:
                ds_slice = pydicom.dcmread(str(dcm_file))
                slices.append(ds_slice.pixel_array)
            
            data = np.stack(slices, axis=-1)
            
        else:
            # Single DICOM file
            ds = pydicom.dcmread(str(scan_path))
            data = ds.pixel_array
            
        return {
            'data': data,
            'affine': None,  # DICOM doesn't have affine matrix
            'header': ds,
            'shape': data.shape,
            'voxel_size': getattr(ds, 'PixelSpacing', [1, 1]) + [getattr(ds, 'SliceThickness', 1)],
            'orientation': None
        }
    
    def _save_results(self, output_dir: str):
        """Save analysis results to output directory."""
        if self.current_results is None:
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save visualizations
        if self.current_results.visualizations:
            vis_dir = output_path / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for name, viz in self.current_results.visualizations.items():
                if hasattr(viz, 'savefig'):
                    viz.savefig(vis_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        
        # Save report
        if self.current_results.report:
            with open(output_path / 'analysis_report.txt', 'w') as f:
                f.write(self.current_results.report)
        
        print(f"💾 Results saved to: {output_path}")


class AnalysisResults:
    """
    Container for all analysis results.
    """
    
    def __init__(self, 
                 scan_path: Union[str, Path],
                 scan_data: Dict,
                 segmentation_results: Dict,
                 abnormality_results: Dict):
        self.scan_path = scan_path
        self.scan_data = scan_data
        self.segmentation_results = segmentation_results
        self.abnormality_results = abnormality_results
        self.visualizations = {}
        self.report = ""
    
    def get_brain_regions(self) -> Dict:
        """Get segmented brain regions."""
        return self.segmentation_results.get('regions', {})
    
    def get_abnormalities(self) -> Dict:
        """Get detected abnormalities."""
        return self.abnormality_results.get('abnormalities', {})
    
    def get_measurements(self) -> Dict:
        """Get quantitative measurements."""
        return {
            'segmentation': self.segmentation_results.get('measurements', {}),
            'abnormalities': self.abnormality_results.get('measurements', {})
        }
    
    def generate_report(self) -> str:
        """Generate analysis report."""
        return self.report


# Convenience function for quick analysis
def analyze_mri_scan(scan_path: Union[str, Path], 
                    output_dir: Optional[str] = None,
                    **kwargs) -> AnalysisResults:
    """
    Quick function to analyze an MRI scan.
    
    Args:
        scan_path: Path to the MRI scan
        output_dir: Output directory for results
        **kwargs: Additional arguments for BrainAnalyzer
        
    Returns:
        AnalysisResults object
    """
    analyzer = BrainAnalyzer(**kwargs)
    return analyzer.analyze_scan(scan_path, output_dir)


# Version info
__version__ = "1.0.0"
__author__ = "TendrilAI Team"
__email__ = "contact@tendrilai.com" 