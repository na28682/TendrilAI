"""
Streamlit Web Application for MRI Brain Analysis
===============================================

This module provides a web-based interface for uploading and analyzing MRI brain scans.
"""

import streamlit as st
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

# Configure page
st.set_page_config(
    page_title="MRI Brain Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 MRI Brain Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload MRI Scan")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an MRI file",
            type=['nii', 'nii.gz', 'dcm', 'dicom'],
            help="Supported formats: NIfTI (.nii, .nii.gz) and DICOM (.dcm)"
        )
        
        # Analysis options
        st.header("⚙️ Analysis Options")
        
        generate_visualizations = st.checkbox(
            "Generate Visualizations",
            value=True,
            help="Create 3D visualizations and plots"
        )
        
        generate_report = st.checkbox(
            "Generate Report",
            value=True,
            help="Create detailed analysis report"
        )
        
        device = st.selectbox(
            "Computing Device",
            options=['auto', 'cpu', 'cuda'],
            help="Select computing device for analysis"
        )
        
        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze Brain Scan",
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.metric("File Type", uploaded_file.type or "Unknown")
        
        if analyze_button:
            with st.spinner("🧠 Analyzing brain scan..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize analyzer
                    analyzer = BrainAnalyzer(device=device)
                    
                    # Perform analysis
                    results = analyzer.analyze_scan(
                        tmp_path,
                        generate_visualizations=generate_visualizations,
                        generate_report=generate_report
                    )
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    # Display results
                    display_results(results)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the MRI Brain Analysis System
        
        This system uses advanced deep learning techniques to analyze MRI brain scans and detect:
        
        ### 🧠 Brain Region Recognition
        - **Gray matter and white matter** segmentation
        - **Cortical regions** identification (frontal, parietal, temporal, occipital lobes)
        - **Subcortical structures** detection (thalamus, basal ganglia, hippocampus)
        - **Ventricles and CSF** identification
        
        ### ⚠️ Abnormality Detection
        - **Tumors and masses** with size and location estimation
        - **Lesions** (MS plaques, infarcts, hemorrhages)
        - **Atrophy** detection (cortical thinning, ventricular enlargement)
        - **Mass effect** and midline shift assessment
        - **Edema** detection and quantification
        
        ### 📊 Analysis Features
        - **3D interactive visualization** of brain structures
        - **Quantitative measurements** (volumes, ratios, symmetry)
        - **Automated report generation** with findings and recommendations
        - **Severity scoring** for detected abnormalities
        
        ### 🚀 Getting Started
        1. Upload your MRI scan file (NIfTI or DICOM format)
        2. Configure analysis options in the sidebar
        3. Click "Analyze Brain Scan" to begin processing
        4. Review results, visualizations, and generated report
        
        ---
        
        **⚠️ Medical Disclaimer**: This system is for research and educational purposes only. 
        It should not be used for clinical diagnosis without proper validation and regulatory approval. 
        Always consult qualified medical professionals for medical decisions.
        """)
        
        # Sample data info
        with st.expander("📋 Sample Data Information"):
            st.markdown("""
            **Supported File Formats:**
            - NIfTI (.nii, .nii.gz) - Most common for research
            - DICOM (.dcm, .dicom) - Clinical standard
            
            **Recommended Scan Types:**
            - T1-weighted MRI (best for structural analysis)
            - T2-weighted MRI (good for lesion detection)
            - FLAIR (fluid-attenuated inversion recovery)
            
            **Image Requirements:**
            - 3D volume data
            - Proper orientation (RAS or similar)
            - Adequate resolution (1mm³ or better)
            - Minimal artifacts
            """)

def display_results(results):
    """Display analysis results."""
    
    st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Summary", 
        "🧠 Segmentation", 
        "⚠️ Abnormalities", 
        "📊 Visualizations", 
        "📝 Report"
    ])
    
    with tab1:
        display_summary(results)
    
    with tab2:
        display_segmentation(results)
    
    with tab3:
        display_abnormalities(results)
    
    with tab4:
        display_visualizations(results)
    
    with tab5:
        display_report(results)

def display_summary(results):
    """Display summary metrics."""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    seg_measurements = results.segmentation_results.get('measurements', {})
    ab_measurements = results.abnormality_results.get('measurements', {})
    
    with col1:
        st.metric(
            "Total Brain Volume",
            f"{seg_measurements.get('total_brain_volume', 0):.1f} cm³"
        )
    
    with col2:
        st.metric(
            "Gray Matter Volume",
            f"{seg_measurements.get('gray_matter_volume', 0):.1f} cm³"
        )
    
    with col3:
        st.metric(
            "White Matter Volume",
            f"{seg_measurements.get('white_matter_volume', 0):.1f} cm³"
        )
    
    with col4:
        st.metric(
            "Abnormality Severity",
            f"{ab_measurements.get('severity_score', 0):.2f}"
        )
    
    # Abnormality summary
    abnormalities = results.abnormality_results.get('abnormalities', {})
    detected_abnormalities = [ab_type for ab_type, info in abnormalities.items() 
                            if info.get('detected', False)]
    
    if detected_abnormalities:
        st.markdown('<h3>⚠️ Detected Abnormalities</h3>', unsafe_allow_html=True)
        
        for ab_type in detected_abnormalities:
            info = abnormalities[ab_type]
            confidence = info.get('confidence', 0.0)
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>{ab_type.replace('_', ' ').title()}</strong><br>
                Confidence: {confidence:.2f}<br>
                Status: {'🔴 High Risk' if confidence > 0.7 else '🟡 Medium Risk' if confidence > 0.4 else '🟢 Low Risk'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No abnormalities detected in this scan.")

def display_segmentation(results):
    """Display segmentation results."""
    
    seg_measurements = results.segmentation_results.get('measurements', {})
    regions = results.segmentation_results.get('regions', {})
    
    # Volume chart
    st.markdown('<h3>📊 Brain Tissue Volumes</h3>', unsafe_allow_html=True)
    
    # Create pie chart
    labels = ['Gray Matter', 'White Matter', 'CSF']
    values = [
        seg_measurements.get('gray_matter_volume', 0),
        seg_measurements.get('white_matter_volume', 0),
        seg_measurements.get('csf_volume', 0)
    ]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="Brain Tissue Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Region analysis
    if regions:
        st.markdown('<h3>🧠 Brain Region Analysis</h3>', unsafe_allow_html=True)
        
        region_data = []
        for region_name, region_info in regions.items():
            volume = region_info.get('volume', 0)
            percentage = region_info.get('percentage', 0)
            region_data.append({
                'Region': region_name.replace('_', ' ').title(),
                'Volume (mm³)': volume,
                'Percentage (%)': percentage
            })
        
        st.dataframe(region_data, use_container_width=True)

def display_abnormalities(results):
    """Display abnormality details."""
    
    abnormalities = results.abnormality_results.get('abnormalities', {})
    measurements = results.abnormality_results.get('measurements', {})
    
    if not abnormalities:
        st.success("✅ No abnormalities detected in this scan.")
        return
    
    # Severity gauge
    severity_score = measurements.get('severity_score', 0.0)
    
    st.markdown('<h3>⚠️ Abnormality Severity</h3>', unsafe_allow_html=True)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=severity_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Severity Score"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed abnormality information
    st.markdown('<h3>📋 Abnormality Details</h3>', unsafe_allow_html=True)
    
    for abnormality_type, info in abnormalities.items():
        if info.get('detected', False):
            with st.expander(f"{abnormality_type.replace('_', ' ').title()}"):
                confidence = info.get('confidence', 0.0)
                st.metric("Confidence", f"{confidence:.2f}")
                
                # Display specific details
                if abnormality_type == 'tumor':
                    count = info.get('count', 0)
                    total_volume = sum(info.get('volumes', []))
                    st.metric("Count", count)
                    st.metric("Total Volume", f"{total_volume:.2f} mm³")
                
                elif abnormality_type == 'lesion':
                    count = info.get('count', 0)
                    st.metric("Count", count)
                
                elif abnormality_type == 'atrophy':
                    brain_volume = info.get('brain_volume', 0)
                    csf_ratio = info.get('csf_ratio', 0)
                    st.metric("Brain Volume", f"{brain_volume:.2f} cm³")
                    st.metric("CSF Ratio", f"{csf_ratio:.2f}")

def display_visualizations(results):
    """Display visualizations."""
    
    if not results.visualizations:
        st.info("No visualizations available. Enable visualization generation in analysis options.")
        return
    
    st.markdown('<h3>📊 Analysis Visualizations</h3>', unsafe_allow_html=True)
    
    # Display available visualizations
    viz_names = list(results.visualizations.keys())
    
    if 'slices' in viz_names:
        st.markdown('<h4>🧠 Multi-planar Views</h4>', unsafe_allow_html=True)
        fig = results.visualizations['slices']
        st.pyplot(fig)
    
    if 'segmentation_overlay' in viz_names:
        st.markdown('<h4>🎨 Segmentation Overlay</h4>', unsafe_allow_html=True)
        fig = results.visualizations['segmentation_overlay']
        st.pyplot(fig)
    
    if 'abnormalities' in viz_names:
        st.markdown('<h4>⚠️ Abnormality Visualization</h4>', unsafe_allow_html=True)
        fig = results.visualizations['abnormalities']
        st.pyplot(fig)
    
    if 'interactive' in viz_names:
        st.markdown('<h4>🖱️ Interactive Visualization</h4>', unsafe_allow_html=True)
        fig = results.visualizations['interactive']
        st.plotly_chart(fig, use_container_width=True)

def display_report(results):
    """Display generated report."""
    
    if not results.report:
        st.info("No report available. Enable report generation in analysis options.")
        return
    
    st.markdown('<h3>📝 Analysis Report</h3>', unsafe_allow_html=True)
    
    # Display report in expandable sections
    report_lines = results.report.split('\n')
    
    current_section = ""
    section_content = []
    
    for line in report_lines:
        if line.startswith('===') or line.startswith('---'):
            # End of section
            if current_section and section_content:
                with st.expander(current_section):
                    st.text('\n'.join(section_content))
            current_section = ""
            section_content = []
        elif line.strip() and not line.startswith(' '):
            # New section header
            if current_section and section_content:
                with st.expander(current_section):
                    st.text('\n'.join(section_content))
            current_section = line.strip()
            section_content = []
        else:
            section_content.append(line)
    
    # Display last section
    if current_section and section_content:
        with st.expander(current_section):
            st.text('\n'.join(section_content))
    
    # Download button for report
    st.download_button(
        label="📥 Download Report",
        data=results.report,
        file_name="brain_analysis_report.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main() 