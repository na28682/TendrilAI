"""
Brain Visualization Module
=========================

This module provides 3D interactive visualizations of brain structures
and abnormalities using various visualization libraries.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pyvista as pv
from pyvista import themes

warnings.filterwarnings('ignore')

# Set PyVista theme
themes.DefaultTheme().show_edges = True
themes.DefaultTheme().color = 'white'

class BrainVisualizer:
    """
    Creates 3D visualizations of brain structures and abnormalities.
    
    Supports multiple visualization types:
    - 3D surface rendering
    - Multi-planar reconstruction (MPR)
    - Segmentation overlays
    - Abnormality highlighting
    - Interactive plots
    """
    
    def __init__(self):
        """Initialize brain visualizer."""
        self.colors = {
            'gray_matter': '#8B4513',
            'white_matter': '#F5F5DC',
            'csf': '#87CEEB',
            'frontal_lobe': '#FF6B6B',
            'parietal_lobe': '#4ECDC4',
            'temporal_lobe': '#45B7D1',
            'occipital_lobe': '#96CEB4',
            'cerebellum': '#FFEAA7',
            'brainstem': '#DDA0DD',
            'tumor': '#FF0000',
            'lesion': '#FF8C00',
            'edema': '#FFD700',
            'atrophy': '#800080'
        }
    
    def create_visualizations(self, results: 'AnalysisResults') -> Dict:
        """
        Create comprehensive visualizations for brain analysis results.
        
        Args:
            results: AnalysisResults object containing all analysis data
            
        Returns:
            Dictionary containing various visualization objects
        """
        visualizations = {}
        
        # Create 2D slice visualizations
        visualizations['slices'] = self._create_slice_visualizations(results)
        
        # Create 3D surface visualization
        visualizations['surface_3d'] = self._create_3d_surface_visualization(results)
        
        # Create segmentation overlay
        visualizations['segmentation_overlay'] = self._create_segmentation_overlay(results)
        
        # Create abnormality visualization
        visualizations['abnormalities'] = self._create_abnormality_visualization(results)
        
        # Create interactive plotly visualization
        visualizations['interactive'] = self._create_interactive_visualization(results)
        
        # Create PyVista 3D visualization
        visualizations['pyvista_3d'] = self._create_pyvista_visualization(results)
        
        return visualizations
    
    def _create_slice_visualizations(self, results: 'AnalysisResults') -> plt.Figure:
        """Create 2D slice visualizations."""
        data = results.scan_data['data']
        segmentation_mask = results.segmentation_results['segmentation_mask']
        region_mask = results.segmentation_results['region_mask']
        
        # Create figure with subplots for different views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Brain MRI Analysis - Multi-planar Views', fontsize=16)
        
        # Get middle slices for each view
        height, width, depth = data.shape
        mid_slice = depth // 2
        mid_height = height // 2
        mid_width = width // 2
        
        # Axial view (top-down)
        axes[0, 0].imshow(data[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Axial View (Top-Down)')
        axes[0, 0].axis('off')
        
        # Coronal view (front-back)
        axes[0, 1].imshow(data[mid_height, :, :], cmap='gray')
        axes[0, 1].set_title('Coronal View (Front-Back)')
        axes[0, 1].axis('off')
        
        # Sagittal view (left-right)
        axes[0, 2].imshow(data[:, mid_width, :], cmap='gray')
        axes[0, 2].set_title('Sagittal View (Left-Right)')
        axes[0, 2].axis('off')
        
        # Segmentation overlay - Axial
        seg_axial = segmentation_mask[:, :, mid_slice]
        axes[1, 0].imshow(data[:, :, mid_slice], cmap='gray', alpha=0.7)
        axes[1, 0].imshow(seg_axial, alpha=0.3, cmap='tab10')
        axes[1, 0].set_title('Segmentation Overlay - Axial')
        axes[1, 0].axis('off')
        
        # Region overlay - Coronal
        region_coronal = region_mask[mid_height, :, :]
        axes[1, 1].imshow(data[mid_height, :, :], cmap='gray', alpha=0.7)
        axes[1, 1].imshow(region_coronal, alpha=0.3, cmap='tab20')
        axes[1, 1].set_title('Region Overlay - Coronal')
        axes[1, 1].axis('off')
        
        # Abnormality overlay - Sagittal
        axes[1, 2].imshow(data[:, mid_width, :], cmap='gray')
        axes[1, 2].set_title('Sagittal View')
        axes[1, 2].axis('off')
        
        # Add abnormality markers if detected
        abnormalities = results.abnormality_results.get('abnormalities', {})
        if abnormalities:
            self._add_abnormality_markers(axes[1, 2], abnormalities, 'sagittal')
        
        plt.tight_layout()
        return fig
    
    def _create_3d_surface_visualization(self, results: 'AnalysisResults') -> plt.Figure:
        """Create 3D surface visualization."""
        data = results.scan_data['data']
        segmentation_mask = results.segmentation_results['segmentation_mask']
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface from brain mask
        brain_mask = segmentation_mask > 0
        
        # Get surface coordinates
        x, y, z = np.where(brain_mask)
        
        # Sample points for visualization (to avoid overcrowding)
        if len(x) > 1000:
            indices = np.random.choice(len(x), 1000, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=data[x, y, z], cmap='gray', alpha=0.6, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Brain Surface Visualization')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Intensity')
        
        return fig
    
    def _create_segmentation_overlay(self, results: 'AnalysisResults') -> plt.Figure:
        """Create segmentation overlay visualization."""
        data = results.scan_data['data']
        segmentation_mask = results.segmentation_results['segmentation_mask']
        region_mask = results.segmentation_results['region_mask']
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Brain Segmentation Analysis', fontsize=16)
        
        # Original scan
        mid_slice = data.shape[2] // 2
        axes[0, 0].imshow(data[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Original MRI Scan')
        axes[0, 0].axis('off')
        
        # Tissue segmentation
        seg_slice = segmentation_mask[:, :, mid_slice]
        axes[0, 1].imshow(data[:, :, mid_slice], cmap='gray', alpha=0.7)
        axes[0, 1].imshow(seg_slice, alpha=0.5, cmap='tab10')
        axes[0, 1].set_title('Tissue Segmentation')
        axes[0, 1].axis('off')
        
        # Region segmentation
        region_slice = region_mask[:, :, mid_slice]
        axes[1, 0].imshow(data[:, :, mid_slice], cmap='gray', alpha=0.7)
        axes[1, 0].imshow(region_slice, alpha=0.5, cmap='tab20')
        axes[1, 0].set_title('Brain Region Segmentation')
        axes[1, 0].axis('off')
        
        # Volume measurements
        measurements = results.segmentation_results.get('measurements', {})
        self._create_volume_chart(axes[1, 1], measurements)
        
        plt.tight_layout()
        return fig
    
    def _create_abnormality_visualization(self, results: 'AnalysisResults') -> plt.Figure:
        """Create abnormality visualization."""
        data = results.scan_data['data']
        abnormalities = results.abnormality_results.get('abnormalities', {})
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Abnormality Detection Results', fontsize=16)
        
        # Original scan
        mid_slice = data.shape[2] // 2
        axes[0, 0].imshow(data[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Original Scan')
        axes[0, 0].axis('off')
        
        # Abnormality overlay
        axes[0, 1].imshow(data[:, :, mid_slice], cmap='gray')
        axes[0, 1].set_title('Detected Abnormalities')
        axes[0, 1].axis('off')
        
        # Add abnormality markers
        if abnormalities:
            self._add_abnormality_markers(axes[0, 1], abnormalities, 'axial')
        
        # Abnormality summary
        self._create_abnormality_summary(axes[1, 0], abnormalities)
        
        # Severity score
        measurements = results.abnormality_results.get('measurements', {})
        self._create_severity_chart(axes[1, 1], measurements)
        
        plt.tight_layout()
        return fig
    
    def _create_interactive_visualization(self, results: 'AnalysisResults') -> go.Figure:
        """Create interactive Plotly visualization."""
        data = results.scan_data['data']
        segmentation_mask = results.segmentation_results['segmentation_mask']
        abnormalities = results.abnormality_results.get('abnormalities', {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Original Scan', 'Segmentation', 'Abnormalities', '3D View'),
            specs=[[{"type": "image"}, {"type": "image"}],
                   [{"type": "image"}, {"type": "scatter3d"}]]
        )
        
        # Get middle slice
        mid_slice = data.shape[2] // 2
        
        # Original scan
        fig.add_trace(
            go.Image(z=data[:, :, mid_slice], colorscale='gray', name='Original'),
            row=1, col=1
        )
        
        # Segmentation overlay
        seg_slice = segmentation_mask[:, :, mid_slice]
        fig.add_trace(
            go.Image(z=seg_slice, colorscale='viridis', name='Segmentation'),
            row=1, col=2
        )
        
        # Abnormality overlay
        fig.add_trace(
            go.Image(z=data[:, :, mid_slice], colorscale='gray', name='Abnormalities'),
            row=2, col=1
        )
        
        # 3D scatter plot
        brain_mask = segmentation_mask > 0
        x, y, z = np.where(brain_mask)
        
        # Sample points for performance
        if len(x) > 1000:
            indices = np.random.choice(len(x), 1000, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=1,
                    color=data[x, y, z],
                    colorscale='gray',
                    opacity=0.6
                ),
                name='3D Brain'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Brain Analysis",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def _create_pyvista_visualization(self, results: 'AnalysisResults') -> pv.Plotter:
        """Create PyVista 3D visualization."""
        data = results.scan_data['data']
        segmentation_mask = results.segmentation_results['segmentation_mask']
        
        # Create PyVista plotter
        plotter = pv.Plotter()
        
        # Create brain surface
        brain_mask = segmentation_mask > 0
        
        # Convert to PyVista grid
        grid = pv.wrap(data)
        
        # Extract brain surface
        brain_surface = grid.extract_geometry()
        
        # Add brain surface to plotter
        plotter.add_mesh(brain_surface, color='lightblue', opacity=0.7, name='Brain')
        
        # Add segmentation regions
        for label in np.unique(segmentation_mask):
            if label == 0:
                continue
            
            region_mask = segmentation_mask == label
            if np.any(region_mask):
                region_grid = pv.wrap(region_mask.astype(float))
                region_surface = region_grid.extract_geometry()
                
                if region_surface.n_points > 0:
                    color = self._get_region_color(label)
                    plotter.add_mesh(region_surface, color=color, opacity=0.5, name=f'Region_{label}')
        
        # Add abnormalities if detected
        abnormalities = results.abnormality_results.get('abnormalities', {})
        if abnormalities:
            self._add_abnormality_to_pyvista(plotter, abnormalities, data)
        
        # Set camera position
        plotter.camera_position = 'iso'
        plotter.add_title("3D Brain Visualization", font_size=16)
        
        return plotter
    
    def _add_abnormality_markers(self, ax, abnormalities: Dict, view: str):
        """Add abnormality markers to plot."""
        for abnormality_type, info in abnormalities.items():
            if info.get('detected', False):
                if abnormality_type == 'tumor':
                    color = self.colors.get('tumor', 'red')
                    marker = 'o'
                elif abnormality_type == 'lesion':
                    color = self.colors.get('lesion', 'orange')
                    marker = 's'
                elif abnormality_type == 'edema':
                    color = self.colors.get('edema', 'yellow')
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'x'
                
                # Add markers at abnormality locations
                locations = info.get('locations', [])
                for loc in locations:
                    if view == 'axial':
                        ax.plot(loc[1], loc[0], marker, color=color, markersize=8, markeredgewidth=2)
                    elif view == 'sagittal':
                        ax.plot(loc[2], loc[0], marker, color=color, markersize=8, markeredgewidth=2)
    
    def _create_volume_chart(self, ax, measurements: Dict):
        """Create volume measurement chart."""
        if not measurements:
            ax.text(0.5, 0.5, 'No measurements available', ha='center', va='center')
            ax.set_title('Volume Measurements')
            return
        
        # Extract volume data
        volumes = []
        labels = []
        
        if 'total_brain_volume' in measurements:
            volumes.append(measurements['total_brain_volume'])
            labels.append('Total Brain')
        
        if 'gray_matter_volume' in measurements:
            volumes.append(measurements['gray_matter_volume'])
            labels.append('Gray Matter')
        
        if 'white_matter_volume' in measurements:
            volumes.append(measurements['white_matter_volume'])
            labels.append('White Matter')
        
        if 'csf_volume' in measurements:
            volumes.append(measurements['csf_volume'])
            labels.append('CSF')
        
        if volumes:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            ax.pie(volumes, labels=labels, colors=colors[:len(volumes)], autopct='%1.1f%%')
            ax.set_title('Brain Tissue Volumes')
        else:
            ax.text(0.5, 0.5, 'No volume data available', ha='center', va='center')
            ax.set_title('Volume Measurements')
    
    def _create_abnormality_summary(self, ax, abnormalities: Dict):
        """Create abnormality summary chart."""
        if not abnormalities:
            ax.text(0.5, 0.5, 'No abnormalities detected', ha='center', va='center')
            ax.set_title('Abnormality Summary')
            return
        
        detected_abnormalities = [ab_type for ab_type, info in abnormalities.items() 
                                if info.get('detected', False)]
        
        if detected_abnormalities:
            # Create bar chart of abnormality types
            counts = [abnormalities[ab_type].get('count', 1) for ab_type in detected_abnormalities]
            colors = [self.colors.get(ab_type, 'gray') for ab_type in detected_abnormalities]
            
            bars = ax.bar(detected_abnormalities, counts, color=colors)
            ax.set_title('Detected Abnormalities')
            ax.set_ylabel('Count')
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No abnormalities detected', ha='center', va='center')
            ax.set_title('Abnormality Summary')
    
    def _create_severity_chart(self, ax, measurements: Dict):
        """Create severity score chart."""
        severity_score = measurements.get('severity_score', 0.0)
        
        # Create gauge chart
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Draw gauge
        gauge = patches.Wedge((0.5, 0.5), 0.4, 0, 180, width=0.1, 
                            color='lightgray', alpha=0.3)
        ax.add_patch(gauge)
        
        # Draw severity indicator
        angle = severity_score * 180
        indicator = patches.Wedge((0.5, 0.5), 0.4, 0, angle, width=0.1,
                                color='red' if severity_score > 0.5 else 'orange' if severity_score > 0.2 else 'green')
        ax.add_patch(indicator)
        
        # Add text
        ax.text(0.5, 0.3, f'Severity: {severity_score:.2f}', ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.2, 'Low' if severity_score < 0.3 else 'Medium' if severity_score < 0.7 else 'High',
                ha='center', va='center', fontsize=10)
        
        ax.set_title('Abnormality Severity Score')
        ax.axis('off')
    
    def _get_region_color(self, label: int) -> str:
        """Get color for brain region."""
        region_colors = {
            1: '#87CEEB',  # CSF
            2: '#8B4513',  # Gray matter
            3: '#F5F5DC',  # White matter
            4: '#FF6B6B',  # Frontal lobe
            5: '#4ECDC4',  # Parietal lobe
            6: '#45B7D1',  # Temporal lobe
            7: '#96CEB4',  # Occipital lobe
            8: '#FFEAA7',  # Cerebellum
            9: '#DDA0DD',  # Brainstem
        }
        return region_colors.get(label, '#808080')
    
    def _add_abnormality_to_pyvista(self, plotter: pv.Plotter, abnormalities: Dict, data: np.ndarray):
        """Add abnormality markers to PyVista visualization."""
        for abnormality_type, info in abnormalities.items():
            if info.get('detected', False):
                locations = info.get('locations', [])
                
                for loc in locations:
                    # Create sphere at abnormality location
                    sphere = pv.Sphere(radius=2, center=loc)
                    color = self.colors.get(abnormality_type, 'red')
                    plotter.add_mesh(sphere, color=color, name=f'{abnormality_type}_{loc}')
    
    def save_visualizations(self, visualizations: Dict, output_dir: str):
        """Save all visualizations to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, viz in visualizations.items():
            if hasattr(viz, 'savefig'):
                # Matplotlib figure
                viz.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
            elif hasattr(viz, 'write_html'):
                # Plotly figure
                viz.write_html(os.path.join(output_dir, f'{name}.html'))
            elif hasattr(viz, 'screenshot'):
                # PyVista plotter
                viz.screenshot(os.path.join(output_dir, f'{name}.png'))
        
        print(f"💾 Visualizations saved to: {output_dir}") 