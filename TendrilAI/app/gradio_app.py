"""
Gradio Web Application for MRI Brain Analysis
============================================

This module provides a Gradio-based interface for uploading and analyzing MRI brain scans.
"""

import gradio as gr
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_analyzer import BrainAnalyzer, analyze_mri_scan

def analyze_brain_scan(file_path, device='auto', generate_viz=True, generate_report=True):
    """
    Analyze brain scan using the BrainAnalyzer.
    
    Args:
        file_path: Path to the MRI file
        device: Computing device ('auto', 'cpu', 'cuda')
        generate_viz: Whether to generate visualizations
        generate_report: Whether to generate report
        
    Returns:
        Dictionary containing analysis results and outputs
    """
    try:
        # Initialize analyzer
        analyzer = BrainAnalyzer(device=device)
        
        # Perform analysis
        results = analyzer.analyze_scan(
            file_path,
            generate_visualizations=generate_viz,
            generate_report=generate_report
        )
        
        # Prepare outputs
        outputs = {}
        
        # Summary metrics
        seg_measurements = results.segmentation_results.get('measurements', {})
        ab_measurements = results.abnormality_results.get('measurements', {})
        
        summary_text = f"""
        **Brain Analysis Results**
        
        **Volume Measurements:**
        - Total Brain Volume: {seg_measurements.get('total_brain_volume', 0):.1f} cm³
        - Gray Matter Volume: {seg_measurements.get('gray_matter_volume', 0):.1f} cm³
        - White Matter Volume: {seg_measurements.get('white_matter_volume', 0):.1f} cm³
        - CSF Volume: {seg_measurements.get('csf_volume', 0):.1f} cm³
        
        **Abnormality Analysis:**
        - Severity Score: {ab_measurements.get('severity_score', 0):.2f}
        - Total Abnormalities: {ab_measurements.get('total_abnormalities', 0)}
        """
        
        outputs['summary'] = summary_text
        
        # Abnormality details
        abnormalities = results.abnormality_results.get('abnormalities', {})
        detected_abnormalities = [ab_type for ab_type, info in abnormalities.items() 
                                if info.get('detected', False)]
        
        if detected_abnormalities:
            abnormality_text = "**Detected Abnormalities:**\n\n"
            for ab_type in detected_abnormalities:
                info = abnormalities[ab_type]
                confidence = info.get('confidence', 0.0)
                abnormality_text += f"- **{ab_type.replace('_', ' ').title()}**: Confidence {confidence:.2f}\n"
        else:
            abnormality_text = "✅ **No abnormalities detected in this scan.**"
        
        outputs['abnormalities'] = abnormality_text
        
        # Report
        if results.report:
            outputs['report'] = results.report
        else:
            outputs['report'] = "No report generated."
        
        # Visualizations
        if results.visualizations and generate_viz:
            # Create a combined visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Brain Analysis Results', fontsize=16)
            
            # Get middle slice for visualization
            data = results.scan_data['data']
            mid_slice = data.shape[2] // 2
            
            # Original scan
            axes[0, 0].imshow(data[:, :, mid_slice], cmap='gray')
            axes[0, 0].set_title('Original Scan')
            axes[0, 0].axis('off')
            
            # Segmentation overlay
            seg_mask = results.segmentation_results['segmentation_mask']
            axes[0, 1].imshow(data[:, :, mid_slice], cmap='gray', alpha=0.7)
            axes[0, 1].imshow(seg_mask[:, :, mid_slice], alpha=0.3, cmap='tab10')
            axes[0, 1].set_title('Segmentation')
            axes[0, 1].axis('off')
            
            # Region overlay
            region_mask = results.segmentation_results['region_mask']
            axes[1, 0].imshow(data[:, :, mid_slice], cmap='gray', alpha=0.7)
            axes[1, 0].imshow(region_mask[:, :, mid_slice], alpha=0.3, cmap='tab20')
            axes[1, 0].set_title('Brain Regions')
            axes[1, 0].axis('off')
            
            # Volume chart
            volumes = [
                seg_measurements.get('gray_matter_volume', 0),
                seg_measurements.get('white_matter_volume', 0),
                seg_measurements.get('csf_volume', 0)
            ]
            labels = ['Gray Matter', 'White Matter', 'CSF']
            axes[1, 1].pie(volumes, labels=labels, autopct='%1.1f%%')
            axes[1, 1].set_title('Tissue Distribution')
            
            plt.tight_layout()
            outputs['visualization'] = fig
        else:
            outputs['visualization'] = None
        
        # Severity gauge
        severity_score = ab_measurements.get('severity_score', 0.0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=severity_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Abnormality Severity Score"},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        outputs['severity_gauge'] = fig
        
        return outputs
        
    except Exception as e:
        error_text = f"❌ Analysis failed: {str(e)}"
        return {
            'summary': error_text,
            'abnormalities': error_text,
            'report': error_text,
            'visualization': None,
            'severity_gauge': None
        }

def create_interface():
    """Create the Gradio interface."""
    
    # Define the interface
    with gr.Blocks(
        title="🧠 MRI Brain Analysis System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🧠 MRI Brain Analysis System</h1>
            <p>Advanced deep learning-based brain scan analysis and abnormality detection</p>
        </div>
        """)
        
        # Warning disclaimer
        gr.HTML("""
        <div class="warning">
            <strong>⚠️ Medical Disclaimer:</strong> This system is for research and educational purposes only. 
            It should not be used for clinical diagnosis without proper validation and regulatory approval. 
            Always consult qualified medical professionals for medical decisions.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📁 Upload MRI Scan")
                
                file_input = gr.File(
                    label="Choose MRI file",
                    file_types=[".nii", ".nii.gz", ".dcm", ".dicom"],
                    type="filepath"
                )
                
                gr.Markdown("### ⚙️ Analysis Options")
                
                device_select = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Computing Device",
                    info="Select computing device for analysis"
                )
                
                generate_viz = gr.Checkbox(
                    label="Generate Visualizations",
                    value=True,
                    info="Create 3D visualizations and plots"
                )
                
                generate_report = gr.Checkbox(
                    label="Generate Report",
                    value=True,
                    info="Create detailed analysis report"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Brain Scan",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📊 Analysis Results")
                
                with gr.Tabs():
                    with gr.TabItem("📈 Summary"):
                        summary_output = gr.Markdown(label="Summary")
                        severity_gauge = gr.Plot(label="Severity Score")
                    
                    with gr.TabItem("⚠️ Abnormalities"):
                        abnormality_output = gr.Markdown(label="Abnormality Details")
                    
                    with gr.TabItem("📊 Visualizations"):
                        visualization_output = gr.Plot(label="Brain Analysis Visualization")
                    
                    with gr.TabItem("📝 Report"):
                        report_output = gr.Textbox(
                            label="Detailed Report",
                            lines=20,
                            max_lines=30
                        )
        
        # Function to handle analysis
        def process_analysis(file_path, device, gen_viz, gen_report):
            if file_path is None:
                return {
                    summary_output: "Please upload an MRI file first.",
                    abnormality_output: "Please upload an MRI file first.",
                    report_output: "Please upload an MRI file first.",
                    visualization_output: None,
                    severity_gauge: None
                }
            
            results = analyze_brain_scan(file_path, device, gen_viz, gen_report)
            
            return {
                summary_output: results['summary'],
                abnormality_output: results['abnormalities'],
                report_output: results['report'],
                visualization_output: results['visualization'],
                severity_gauge: results['severity_gauge']
            }
        
        # Connect the button to the function
        analyze_btn.click(
            fn=process_analysis,
            inputs=[file_input, device_select, generate_viz, generate_report],
            outputs=[summary_output, abnormality_output, report_output, visualization_output, severity_gauge]
        )
        
        # Information section
        with gr.Accordion("ℹ️ System Information", open=False):
            gr.Markdown("""
            ### 🧠 Brain Analysis Features
            
            **Brain Region Recognition:**
            - Gray matter and white matter segmentation
            - Cortical regions identification (frontal, parietal, temporal, occipital lobes)
            - Subcortical structures detection (thalamus, basal ganglia, hippocampus)
            - Ventricles and CSF identification
            
            **Abnormality Detection:**
            - Tumors and masses with size and location estimation
            - Lesions (MS plaques, infarcts, hemorrhages)
            - Atrophy detection (cortical thinning, ventricular enlargement)
            - Mass effect and midline shift assessment
            - Edema detection and quantification
            
            **Analysis Features:**
            - 3D interactive visualization of brain structures
            - Quantitative measurements (volumes, ratios, symmetry)
            - Automated report generation with findings and recommendations
            - Severity scoring for detected abnormalities
            
            ### 📋 Supported File Formats
            
            **NIfTI (.nii, .nii.gz):**
            - Most common format for research
            - Contains 3D volume data
            - Includes metadata and orientation information
            
            **DICOM (.dcm, .dicom):**
            - Clinical standard format
            - Can be single file or series
            - Rich metadata included
            
            ### 🎯 Recommended Scan Types
            
            **T1-weighted MRI:**
            - Best for structural analysis
            - Good gray/white matter contrast
            - Standard for brain segmentation
            
            **T2-weighted MRI:**
            - Good for lesion detection
            - Shows fluid and pathology well
            - Useful for abnormality detection
            
            **FLAIR:**
            - Fluid-attenuated inversion recovery
            - Excellent for lesion detection
            - Suppresses CSF signal
            """)
    
    return interface

def main():
    """Main function to run the Gradio app."""
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main() 