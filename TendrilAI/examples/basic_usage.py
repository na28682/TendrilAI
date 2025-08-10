#!/usr/bin/env python3
"""
Basic Usage Example
==================

This script demonstrates how to use the MRI Brain Analysis System
programmatically for analyzing brain scans.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_analyzer import BrainAnalyzer, analyze_mri_scan

def create_sample_data():
    """Create sample MRI data for demonstration."""
    print("📊 Creating sample MRI data...")
    
    # Create a simple 3D brain-like structure
    # This is just for demonstration - real analysis would use actual MRI files
    
    # Create 3D array (height, width, depth)
    data = np.zeros((128, 128, 64), dtype=np.float32)
    
    # Create brain-like structure
    center_y, center_x, center_z = 64, 64, 32
    
    # Create brain mask (ellipsoid)
    for y in range(128):
        for x in range(128):
            for z in range(64):
                # Calculate distance from center
                dy = (y - center_y) / 40
                dx = (x - center_x) / 50
                dz = (z - center_z) / 25
                
                # Ellipsoid equation
                if (dx**2 + dy**2 + dz**2) <= 1:
                    # Brain tissue
                    data[y, x, z] = np.random.normal(0.6, 0.1)
                    
                    # Add some structure
                    if (dx**2 + dy**2 + dz**2) <= 0.7:
                        # Gray matter (darker)
                        data[y, x, z] = np.random.normal(0.4, 0.1)
                    elif (dx**2 + dy**2 + dz**2) <= 0.9:
                        # White matter (lighter)
                        data[y, x, z] = np.random.normal(0.8, 0.1)
    
    # Add some "abnormalities" for demonstration
    # Add a small "tumor" (high intensity region)
    tumor_y, tumor_x, tumor_z = 70, 60, 30
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            for dz in range(-3, 4):
                y, x, z = tumor_y + dy, tumor_x + dx, tumor_z + dz
                if 0 <= y < 128 and 0 <= x < 128 and 0 <= z < 64:
                    if (dy**2 + dx**2 + dz**2) <= 9:
                        data[y, x, z] = np.random.normal(0.9, 0.05)
    
    # Add some "lesions" (small dark spots)
    for i in range(3):
        lesion_y = np.random.randint(40, 90)
        lesion_x = np.random.randint(40, 90)
        lesion_z = np.random.randint(20, 45)
        
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                for dz in range(-2, 3):
                    y, x, z = lesion_y + dy, lesion_x + dx, lesion_z + dz
                    if 0 <= y < 128 and 0 <= x < 128 and 0 <= z < 64:
                        if (dy**2 + dx**2 + dz**2) <= 4:
                            data[y, x, z] = np.random.normal(0.2, 0.05)
    
    return data

def save_sample_nifti(data, output_path):
    """Save sample data as NIfTI file."""
    import nibabel as nib
    
    # Create affine matrix (identity for simplicity)
    affine = np.eye(4)
    affine[0, 0] = 1.0  # x voxel size
    affine[1, 1] = 1.0  # y voxel size
    affine[2, 2] = 1.0  # z voxel size
    
    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)
    
    # Save to file
    nib.save(img, output_path)
    print(f"💾 Saved sample data to: {output_path}")

def example_basic_analysis():
    """Example of basic brain analysis."""
    print("🧠 Example: Basic Brain Analysis")
    print("=" * 40)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save as NIfTI file
    sample_file = Path(__file__).parent / "sample_brain.nii.gz"
    save_sample_nifti(sample_data, sample_file)
    
    # Initialize analyzer
    print("🔧 Initializing brain analyzer...")
    analyzer = BrainAnalyzer(device='cpu')  # Use CPU for demo
    
    # Perform analysis
    print("🔍 Analyzing brain scan...")
    results = analyzer.analyze_scan(
        sample_file,
        generate_visualizations=True,
        generate_report=True
    )
    
    # Display results
    print("\n📊 Analysis Results:")
    print("-" * 20)
    
    # Segmentation results
    seg_measurements = results.segmentation_results.get('measurements', {})
    print(f"Total Brain Volume: {seg_measurements.get('total_brain_volume', 0):.1f} cm³")
    print(f"Gray Matter Volume: {seg_measurements.get('gray_matter_volume', 0):.1f} cm³")
    print(f"White Matter Volume: {seg_measurements.get('white_matter_volume', 0):.1f} cm³")
    print(f"CSF Volume: {seg_measurements.get('csf_volume', 0):.1f} cm³")
    
    # Abnormality results
    ab_measurements = results.abnormality_results.get('measurements', {})
    print(f"Abnormality Severity: {ab_measurements.get('severity_score', 0):.2f}")
    print(f"Total Abnormalities: {ab_measurements.get('total_abnormalities', 0)}")
    
    # Display detected abnormalities
    abnormalities = results.abnormality_results.get('abnormalities', {})
    detected_abnormalities = [ab_type for ab_type, info in abnormalities.items() 
                            if info.get('detected', False)]
    
    if detected_abnormalities:
        print("\n⚠️ Detected Abnormalities:")
        for ab_type in detected_abnormalities:
            info = abnormalities[ab_type]
            confidence = info.get('confidence', 0.0)
            print(f"  - {ab_type.replace('_', ' ').title()}: {confidence:.2f} confidence")
    else:
        print("\n✅ No abnormalities detected")
    
    # Display brain regions
    regions = results.segmentation_results.get('regions', {})
    if regions:
        print("\n🧠 Brain Regions:")
        for region_name, region_info in regions.items():
            volume = region_info.get('volume', 0)
            percentage = region_info.get('percentage', 0)
            print(f"  - {region_name.replace('_', ' ').title()}: {volume:.1f} mm³ ({percentage:.1f}%)")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save report
    if results.report:
        report_file = output_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(results.report)
        print(f"\n📝 Report saved to: {report_file}")
    
    # Save visualizations
    if results.visualizations:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for name, viz in results.visualizations.items():
            if hasattr(viz, 'savefig'):
                viz.savefig(viz_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        
        print(f"📊 Visualizations saved to: {viz_dir}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

def example_quick_analysis():
    """Example of quick analysis using convenience function."""
    print("\n🚀 Example: Quick Analysis")
    print("=" * 30)
    
    # Create sample data
    sample_data = create_sample_data()
    sample_file = Path(__file__).parent / "sample_brain_quick.nii.gz"
    save_sample_nifti(sample_data, sample_file)
    
    # Use convenience function
    print("🔍 Performing quick analysis...")
    results = analyze_mri_scan(sample_file, device='cpu')
    
    # Get results
    brain_regions = results.get_brain_regions()
    abnormalities = results.get_abnormalities()
    measurements = results.get_measurements()
    
    print(f"📊 Found {len(brain_regions)} brain regions")
    print(f"⚠️ Detected {len(abnormalities)} abnormalities")
    print(f"📏 Generated {len(measurements)} measurement categories")
    
    print("✅ Quick analysis complete!")

def example_custom_analysis():
    """Example of custom analysis with specific options."""
    print("\n⚙️ Example: Custom Analysis")
    print("=" * 30)
    
    # Create sample data
    sample_data = create_sample_data()
    sample_file = Path(__file__).parent / "sample_brain_custom.nii.gz"
    save_sample_nifti(sample_data, sample_file)
    
    # Initialize analyzer with custom options
    analyzer = BrainAnalyzer(
        model_path=None,  # Use default model path
        device='cpu'      # Use CPU
    )
    
    # Perform custom analysis
    print("🔍 Performing custom analysis...")
    results = analyzer.analyze_scan(
        sample_file,
        output_dir=Path(__file__).parent / "custom_results",
        generate_visualizations=True,
        generate_report=True
    )
    
    # Access specific results
    seg_mask = results.segmentation_results['segmentation_mask']
    region_mask = results.segmentation_results['region_mask']
    
    print(f"📊 Segmentation mask shape: {seg_mask.shape}")
    print(f"🗺️ Region mask shape: {region_mask.shape}")
    print(f"🧠 Unique regions: {np.unique(region_mask)}")
    
    # Calculate custom metrics
    total_voxels = seg_mask.size
    brain_voxels = np.sum(seg_mask > 0)
    brain_percentage = (brain_voxels / total_voxels) * 100
    
    print(f"📈 Brain occupies {brain_percentage:.1f}% of the scan volume")
    
    print("✅ Custom analysis complete!")

def main():
    """Main function to run all examples."""
    
    print("🧠 MRI Brain Analysis System - Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_analysis()
        example_quick_analysis()
        example_custom_analysis()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📋 Next steps:")
        print("1. Try the web interfaces:")
        print("   - Streamlit: streamlit run app/streamlit_app.py")
        print("   - Gradio: python app/gradio_app.py")
        print("2. Use your own MRI data")
        print("3. Customize the analysis parameters")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 