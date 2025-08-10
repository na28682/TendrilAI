"""
Brain Segmentation Module
========================

This module handles the segmentation of brain regions using deep learning models.
It identifies different brain structures including gray matter, white matter,
cortical regions, and subcortical structures.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure, morphology
import nibabel as nib

warnings.filterwarnings('ignore')

class UNet3D(nn.Module):
    """
    3D U-Net architecture for brain segmentation.
    """
    
    def __init__(self, in_channels=1, out_channels=8, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Downsampling
        in_channels_temp = in_channels
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels_temp, feature))
            in_channels_temp = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        
        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(features[-1] * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with batch normalization and ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class BrainSegmentation:
    """
    Brain segmentation using deep learning models.
    
    This class handles the segmentation of brain regions including:
    - Gray matter and white matter
    - Cortical regions (frontal, parietal, temporal, occipital)
    - Subcortical structures (thalamus, basal ganglia, hippocampus)
    - Ventricles and CSF
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize brain segmentation.
        
        Args:
            model_path: Path to pre-trained models
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.segmentation_model = None
        self.region_model = None
        self._load_models()
        
        # Brain region labels
        self.region_labels = {
            0: 'background',
            1: 'cerebrospinal_fluid',
            2: 'gray_matter',
            3: 'white_matter',
            4: 'frontal_lobe',
            5: 'parietal_lobe',
            6: 'temporal_lobe',
            7: 'occipital_lobe',
            8: 'cerebellum',
            9: 'brainstem',
            10: 'thalamus',
            11: 'basal_ganglia',
            12: 'hippocampus',
            13: 'ventricles'
        }
    
    def _load_models(self):
        """Load pre-trained segmentation models."""
        try:
            # Load main segmentation model
            model_file = os.path.join(self.model_path, 'brain_segmentation_model.pth')
            if os.path.exists(model_file):
                self.segmentation_model = UNet3D(in_channels=1, out_channels=4)
                self.segmentation_model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.segmentation_model.to(self.device)
                self.segmentation_model.eval()
                print("✅ Loaded brain segmentation model")
            else:
                print("⚠️ No pre-trained segmentation model found. Using rule-based segmentation.")
                self.segmentation_model = None
            
            # Load region-specific model
            region_model_file = os.path.join(self.model_path, 'brain_regions_model.pth')
            if os.path.exists(region_model_file):
                self.region_model = UNet3D(in_channels=1, out_channels=len(self.region_labels))
                self.region_model.load_state_dict(torch.load(region_model_file, map_location=self.device))
                self.region_model.to(self.device)
                self.region_model.eval()
                print("✅ Loaded brain regions model")
            else:
                print("⚠️ No pre-trained regions model found. Using atlas-based segmentation.")
                self.region_model = None
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.segmentation_model = None
            self.region_model = None
    
    def segment_brain(self, scan_data: Dict) -> Dict:
        """
        Perform brain segmentation on MRI scan.
        
        Args:
            scan_data: Dictionary containing scan data and metadata
            
        Returns:
            Dictionary containing segmentation results
        """
        data = scan_data['data']
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Perform segmentation
        if self.segmentation_model is not None:
            segmentation_mask = self._deep_learning_segmentation(processed_data)
        else:
            segmentation_mask = self._rule_based_segmentation(processed_data)
        
        # Identify brain regions
        if self.region_model is not None:
            region_mask = self._deep_learning_regions(processed_data)
        else:
            region_mask = self._atlas_based_regions(processed_data, segmentation_mask)
        
        # Post-process results
        processed_mask = self._postprocess_segmentation(segmentation_mask)
        processed_regions = self._postprocess_regions(region_mask)
        
        # Calculate measurements
        measurements = self._calculate_measurements(processed_mask, processed_regions, scan_data)
        
        return {
            'segmentation_mask': processed_mask,
            'region_mask': processed_regions,
            'regions': self._extract_regions(processed_regions),
            'measurements': measurements
        }
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess MRI data for segmentation."""
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min())
        
        # Apply Gaussian smoothing to reduce noise
        data = ndimage.gaussian_filter(data, sigma=0.5)
        
        # Ensure 3D data
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        
        return data
    
    def _deep_learning_segmentation(self, data: np.ndarray) -> np.ndarray:
        """Perform segmentation using deep learning model."""
        # Prepare input tensor
        input_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            mask = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
        
        return mask
    
    def _rule_based_segmentation(self, data: np.ndarray) -> np.ndarray:
        """Perform rule-based brain segmentation."""
        # Simple thresholding-based segmentation
        mask = np.zeros_like(data, dtype=np.uint8)
        
        # Brain extraction using Otsu thresholding
        brain_mask = self._extract_brain(data)
        
        # Tissue classification
        gray_matter = self._classify_gray_matter(data, brain_mask)
        white_matter = self._classify_white_matter(data, brain_mask)
        csf = self._classify_csf(data, brain_mask)
        
        # Combine masks
        mask[csf] = 1
        mask[gray_matter] = 2
        mask[white_matter] = 3
        
        return mask
    
    def _extract_brain(self, data: np.ndarray) -> np.ndarray:
        """Extract brain from skull using morphological operations."""
        # Apply Otsu thresholding
        threshold = self._otsu_threshold(data)
        brain_mask = data > threshold
        
        # Morphological operations to clean up
        brain_mask = morphology.binary_opening(brain_mask, morphology.ball(3))
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(5))
        
        # Fill holes
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        return brain_mask
    
    def _otsu_threshold(self, data: np.ndarray) -> float:
        """Calculate Otsu threshold for image."""
        from skimage.filters import threshold_otsu
        return threshold_otsu(data)
    
    def _classify_gray_matter(self, data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Classify gray matter using intensity thresholds."""
        # Gray matter typically has intermediate intensity
        gray_threshold_low = np.percentile(data[brain_mask], 30)
        gray_threshold_high = np.percentile(data[brain_mask], 70)
        
        gray_matter = (data >= gray_threshold_low) & (data <= gray_threshold_high) & brain_mask
        return gray_matter
    
    def _classify_white_matter(self, data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Classify white matter using intensity thresholds."""
        # White matter typically has higher intensity
        white_threshold = np.percentile(data[brain_mask], 80)
        
        white_matter = (data > white_threshold) & brain_mask
        return white_matter
    
    def _classify_csf(self, data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Classify CSF using intensity thresholds."""
        # CSF typically has lower intensity
        csf_threshold = np.percentile(data[brain_mask], 20)
        
        csf = (data < csf_threshold) & brain_mask
        return csf
    
    def _deep_learning_regions(self, data: np.ndarray) -> np.ndarray:
        """Identify brain regions using deep learning."""
        # Prepare input tensor
        input_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.region_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            regions = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
        
        return regions
    
    def _atlas_based_regions(self, data: np.ndarray, segmentation_mask: np.ndarray) -> np.ndarray:
        """Identify brain regions using atlas-based approach."""
        # Create a simple atlas-based segmentation
        regions = np.zeros_like(segmentation_mask, dtype=np.uint8)
        
        # Get brain mask
        brain_mask = segmentation_mask > 0
        
        # Simple region assignment based on position
        height, width, depth = data.shape
        
        # Frontal lobe (anterior portion)
        frontal_mask = brain_mask.copy()
        frontal_mask[:, :width//3, :] = False
        regions[frontal_mask] = 4
        
        # Parietal lobe (superior portion)
        parietal_mask = brain_mask.copy()
        parietal_mask[:height//2, :, :] = False
        parietal_mask[:, width//3:2*width//3, :] = False
        regions[parietal_mask] = 5
        
        # Temporal lobe (lateral portions)
        temporal_mask = brain_mask.copy()
        temporal_mask[:, :width//4, :] = False
        temporal_mask[:, 3*width//4:, :] = False
        regions[temporal_mask] = 6
        
        # Occipital lobe (posterior portion)
        occipital_mask = brain_mask.copy()
        occipital_mask[:, 2*width//3:, :] = False
        regions[occipital_mask] = 7
        
        # Cerebellum (inferior portion)
        cerebellum_mask = brain_mask.copy()
        cerebellum_mask[height//2:, :, :] = False
        regions[cerebellum_mask] = 8
        
        return regions
    
    def _postprocess_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """Post-process segmentation mask."""
        # Remove small objects
        for label in np.unique(mask):
            if label == 0:
                continue
            label_mask = mask == label
            label_mask = morphology.remove_small_objects(label_mask, min_size=100)
            mask[label_mask] = label
        
        return mask
    
    def _postprocess_regions(self, regions: np.ndarray) -> np.ndarray:
        """Post-process region mask."""
        # Apply morphological operations to smooth regions
        processed_regions = regions.copy()
        
        for label in np.unique(regions):
            if label == 0:
                continue
            label_mask = regions == label
            label_mask = morphology.binary_closing(label_mask, morphology.ball(2))
            processed_regions[label_mask] = label
        
        return processed_regions
    
    def _extract_regions(self, region_mask: np.ndarray) -> Dict:
        """Extract information about each brain region."""
        regions = {}
        
        for label, name in self.region_labels.items():
            if label == 0:  # Skip background
                continue
            
            region_mask_binary = region_mask == label
            if np.any(region_mask_binary):
                # Calculate region properties
                props = measure.regionprops(region_mask_binary.astype(int))[0]
                
                regions[name] = {
                    'volume': props.area,
                    'centroid': props.centroid,
                    'bbox': props.bbox,
                    'extent': props.extent,
                    'solidity': props.solidity
                }
        
        return regions
    
    def _calculate_measurements(self, 
                              segmentation_mask: np.ndarray, 
                              region_mask: np.ndarray,
                              scan_data: Dict) -> Dict:
        """Calculate quantitative measurements."""
        voxel_size = scan_data.get('voxel_size', [1, 1, 1])
        voxel_volume = np.prod(voxel_size)
        
        measurements = {
            'total_brain_volume': np.sum(segmentation_mask > 0) * voxel_volume,
            'gray_matter_volume': np.sum(segmentation_mask == 2) * voxel_volume,
            'white_matter_volume': np.sum(segmentation_mask == 3) * voxel_volume,
            'csf_volume': np.sum(segmentation_mask == 1) * voxel_volume,
            'gray_white_ratio': np.sum(segmentation_mask == 2) / max(np.sum(segmentation_mask == 3), 1),
            'brain_regions': {}
        }
        
        # Calculate region volumes
        for label, name in self.region_labels.items():
            if label == 0:
                continue
            volume = np.sum(region_mask == label) * voxel_volume
            measurements['brain_regions'][name] = {
                'volume': volume,
                'percentage': volume / max(measurements['total_brain_volume'], 1) * 100
            }
        
        return measurements 