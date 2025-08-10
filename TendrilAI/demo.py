#!/usr/bin/env python3
"""
MRI Brain Analysis System Demo
=============================

This script demonstrates the MRI brain analysis system capabilities
without requiring all dependencies to be installed.
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_sample_brain_data():
    """Create sample brain-like data for demonstration."""
    print("🧠 Creating sample brain data...")
    
    # Create 3D brain-like structure
    data = np.zeros((128, 128, 64), dtype=np.float32)
    
    # Brain center
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
                    
                    # Add tissue structure
                    if (dx**2 + dy**2 + dz**2) <= 0.7:
                        # Gray matter (darker)
                        data[y, x, z] = np.random.normal(0.4, 0.1)
                    elif (dx**2 + dy**2 + dz**2) <= 0.9:
                        # White matter (lighter)
                        data[y, x, z] = np.random.normal(0.8, 0.1)
    
    # Add simulated abnormalities
    # "Tumor" (high intensity region)
    tumor_y, tumor_x, tumor_z = 70, 60, 30
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            for dz in range(-3, 4):
                y, x, z = tumor_y + dy, tumor_x + dx, tumor_z + dz
                if 0 <= y < 128 and 0 <= x < 128 and 0 <= z < 64:
                    if (dy**2 + dx**2 + dz**2) <= 9:
                        data[y, x, z] = np.random.normal(0.9, 0.05)
    
    # "Lesions" (small dark spots)
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

def simulate_brain_analysis():
    """Simulate brain analysis results."""
    print("🔍 Simulating brain analysis...")
    
    # Create sample data
    brain_data = create_sample_brain_data()
    
    # Simulate segmentation results
    segmentation_results = {
        'segmentation_mask': np.random.randint(0, 4, brain_data.shape),
        'region_mask': np.random.randint(0, 14, brain_data.shape),
        'measurements': {
            'total_brain_volume': 1350.5,
            'gray_matter_volume': 750.2,
            'white_matter_volume': 450.8,
            'csf_volume': 149.5,
            'gray_white_ratio': 1.67
        },
        'regions': {
            'frontal_lobe': {'volume': 280.5, 'percentage': 20.8},
            'parietal_lobe': {'volume': 245.3, 'percentage': 18.2},
            'temporal_lobe': {'volume': 220.1, 'percentage': 16.3},
            'occipital_lobe': {'volume': 185.7, 'percentage': 13.8},
            'cerebellum': {'volume': 150.2, 'percentage': 11.1},
            'brainstem': {'volume': 45.8, 'percentage': 3.4}
        }
    }
    
    # Simulate abnormality detection
    abnormality_results = {
        'abnormalities': {
            'tumor': {
                'detected': True,
                'confidence': 0.85,
                'count': 1,
                'locations': [(70, 60, 30)],
                'sizes': [125],
                'volumes': [125]
            },
            'lesion': {
                'detected': True,
                'confidence': 0.72,
                'count': 3,
                'locations': [(45, 55, 25), (78, 42, 35), (62, 88, 28)],
                'sizes': [45, 32, 38]
            },
            'atrophy': {
                'detected': False,
                'confidence': 0.0
            },
            'mass_effect': {
                'detected': False,
                'confidence': 0.0
            },
            'edema': {
                'detected': True,
                'confidence': 0.68,
                'count': 2,
                'locations': [(68, 58, 29), (72, 62, 31)],
                'volumes': [85, 92]
            }
        },
        'measurements': {
            'total_abnormalities': 3,
            'severity_score': 0.75,
            'abnormality_details': {
                'tumor': {'count': 1, 'total_volume': 125, 'largest_size': 125},
                'lesion': {'count': 3, 'total_volume': 115},
                'edema': {'count': 2, 'total_volume': 177}
            }
        }
    }
    
    return brain_data, segmentation_results, abnormality_results

def display_analysis_results(brain_data, segmentation_results, abnormality_results):
    """Display analysis results in a formatted way."""
    print("\n" + "="*60)
    print("🧠 MRI BRAIN ANALYSIS RESULTS")
    print("="*60)
    
    # Brain measurements
    measurements = segmentation_results['measurements']
    print("\n📊 BRAIN MEASUREMENTS:")
    print(f"  • Total Brain Volume: {measurements['total_brain_volume']:.1f} cm³")
    print(f"  • Gray Matter Volume: {measurements['gray_matter_volume']:.1f} cm³")
    print(f"  • White Matter Volume: {measurements['white_matter_volume']:.1f} cm³")
    print(f"  • CSF Volume: {measurements['csf_volume']:.1f} cm³")
    print(f"  • Gray/White Matter Ratio: {measurements['gray_white_ratio']:.2f}")
    
    # Brain regions
    regions = segmentation_results['regions']
    print("\n🗺️ BRAIN REGIONS:")
    for region_name, region_info in regions.items():
        volume = region_info['volume']
        percentage = region_info['percentage']
        print(f"  • {region_name.replace('_', ' ').title()}: {volume:.1f} cm³ ({percentage:.1f}%)")
    
    # Abnormality analysis
    abnormalities = abnormality_results['abnormalities']
    measurements = abnormality_results['measurements']
    
    print(f"\n⚠️ ABNORMALITY ANALYSIS:")
    print(f"  • Total Abnormalities: {measurements['total_abnormalities']}")
    print(f"  • Severity Score: {measurements['severity_score']:.2f}")
    
    detected_abnormalities = [ab_type for ab_type, info in abnormalities.items() 
                            if info.get('detected', False)]
    
    if detected_abnormalities:
        print("\n🔍 DETECTED ABNORMALITIES:")
        for ab_type in detected_abnormalities:
            info = abnormalities[ab_type]
            confidence = info['confidence']
            status = "🔴 HIGH RISK" if confidence > 0.7 else "🟡 MEDIUM RISK" if confidence > 0.4 else "🟢 LOW RISK"
            
            print(f"  • {ab_type.replace('_', ' ').title()}: {confidence:.2f} confidence - {status}")
            
            if ab_type == 'tumor':
                count = info.get('count', 0)
                total_volume = sum(info.get('volumes', []))
                print(f"    - Count: {count}, Total Volume: {total_volume} mm³")
            elif ab_type == 'lesion':
                count = info.get('count', 0)
                print(f"    - Count: {count}")
            elif ab_type == 'edema':
                count = info.get('count', 0)
                total_volume = sum(info.get('volumes', []))
                print(f"    - Count: {count}, Total Volume: {total_volume} mm³")
    else:
        print("\n✅ No abnormalities detected")
    
    # Clinical recommendations
    severity_score = measurements['severity_score']
    print(f"\n💡 CLINICAL RECOMMENDATIONS:")
    
    if severity_score > 0.7:
        print("  • URGENT: High severity abnormalities detected")
        print("  • Immediate clinical evaluation recommended")
        print("  • Consider emergency imaging protocols")
    elif severity_score > 0.4:
        print("  • MODERATE: Significant abnormalities detected")
        print("  • Prompt clinical evaluation recommended")
        print("  • Consider specialized imaging studies")
    else:
        print("  • MILD: Minor abnormalities detected")
        print("  • Routine clinical follow-up recommended")
    
    # Specific recommendations
    if 'tumor' in detected_abnormalities:
        print("  • Tumor detected: Consider biopsy and histopathological analysis")
        print("  • Consult with neuro-oncology specialist")
        print("  • Consider advanced imaging (contrast-enhanced MRI, spectroscopy)")
    
    if 'lesion' in detected_abnormalities:
        print("  • Lesions detected: Consider differential diagnosis")
        print("  • Rule out demyelinating disease, infection, or vascular causes")
        print("  • Consider follow-up imaging to assess progression")
    
    if 'edema' in detected_abnormalities:
        print("  • Edema detected: Consider underlying cause evaluation")
        print("  • Monitor for neurological deterioration")
        print("  • Consider anti-edema therapy if indicated")
    
    print("\n" + "="*60)
    print("📋 TECHNICAL NOTES:")
    print("  • Analysis performed using deep learning-based segmentation")
    print("  • Abnormality detection uses both rule-based and AI methods")
    print("  • All measurements are approximate and should be validated clinically")
    print("  • This demo is for research and educational purposes only")
    print("="*60)

def show_system_capabilities():
    """Show the system's capabilities."""
    print("\n🚀 MRI BRAIN ANALYSIS SYSTEM CAPABILITIES")
    print("="*50)
    
    capabilities = {
        "🧠 Brain Region Recognition": [
            "Automatic segmentation of brain structures",
            "Gray matter/white matter differentiation", 
            "Cortical regions identification (frontal, parietal, temporal, occipital lobes)",
            "Subcortical structures detection (thalamus, basal ganglia, hippocampus)",
            "Ventricles and CSF identification"
        ],
        "⚠️ Abnormality Detection": [
            "Tumor detection with size and location estimation",
            "Lesion identification (MS plaques, infarcts, hemorrhages)",
            "Atrophy detection (cortical thinning, ventricular enlargement)",
            "Mass effect assessment",
            "Edema detection and quantification"
        ],
        "📊 Analysis & Visualization": [
            "3D interactive visualization of brain structures",
            "Quantitative measurements (volumes, ratios, symmetry)",
            "Automated report generation with findings and recommendations",
            "Longitudinal analysis for tracking changes over time"
        ],
        "🔧 Technical Features": [
            "Deep learning-based segmentation (U-Net 3D)",
            "Attention mechanisms for abnormality detection",
            "Multi-modal analysis support",
            "GPU acceleration for faster processing",
            "Comprehensive API for programmatic access"
        ]
    }
    
    for category, features in capabilities.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  • {feature}")
    
    print("\n📋 Supported File Formats:")
    print("  • NIfTI (.nii, .nii.gz) - Research standard")
    print("  • DICOM (.dcm, .dicom) - Clinical standard")
    
    print("\n🎯 Recommended Scan Types:")
    print("  • T1-weighted MRI (best for structural analysis)")
    print("  • T2-weighted MRI (good for lesion detection)")
    print("  • FLAIR (fluid-attenuated inversion recovery)")

def show_usage_examples():
    """Show usage examples."""
    print("\n💻 USAGE EXAMPLES")
    print("="*30)
    
    print("\n1️⃣ Basic Usage:")
    print("""
from brain_analyzer import BrainAnalyzer

# Initialize analyzer
analyzer = BrainAnalyzer()

# Analyze MRI scan
results = analyzer.analyze_scan("path/to/mri_scan.nii.gz")

# Get results
brain_regions = results.get_brain_regions()
abnormalities = results.get_abnormalities()
report = results.generate_report()
""")
    
    print("\n2️⃣ Web Interface (Streamlit):")
    print("streamlit run app/streamlit_app.py")
    
    print("\n3️⃣ Web Interface (Gradio):")
    print("python app/gradio_app.py")
    
    print("\n4️⃣ Quick Analysis:")
    print("""
from brain_analyzer import analyze_mri_scan

results = analyze_mri_scan("path/to/mri_scan.nii.gz")
""")

def main():
    """Main demo function."""
    print("🧠 MRI BRAIN ANALYSIS SYSTEM DEMO")
    print("="*50)
    
    # Show system capabilities
    show_system_capabilities()
    
    # Simulate analysis
    print("\n" + "="*50)
    print("🔬 DEMONSTRATION: SIMULATED BRAIN ANALYSIS")
    print("="*50)
    
    brain_data, segmentation_results, abnormality_results = simulate_brain_analysis()
    
    # Display results
    display_analysis_results(brain_data, segmentation_results, abnormality_results)
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 DEMO COMPLETE!")
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download models: python scripts/download_models.py")
    print("3. Run web interface: streamlit run app/streamlit_app.py")
    print("4. Try with your own MRI data")
    print("5. Check out the examples in examples/ directory")
    
    print("\n⚠️ MEDICAL DISCLAIMER:")
    print("This system is for research and educational purposes only.")
    print("It should not be used for clinical diagnosis without proper validation.")
    print("Always consult qualified medical professionals for medical decisions.")

if __name__ == "__main__":
    main() 