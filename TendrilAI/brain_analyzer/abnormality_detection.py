"""
Abnormality Detection Module
===========================

This module detects various brain abnormalities including:
- Tumors and masses
- Lesions (MS plaques, infarcts, hemorrhages)
- Atrophy and volume loss
- Mass effect and midline shift
- Edema and swelling
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.feature import peak_local_maxima
import cv2

warnings.filterwarnings('ignore')

class AbnormalityDetector3D(nn.Module):
    """
    3D CNN for abnormality detection.
    """
    
    def __init__(self, in_channels=1, num_classes=6):
        super(AbnormalityDetector3D, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AttentionModule(nn.Module):
    """
    Attention module for focusing on suspicious regions.
    """
    
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights, attention_weights


class AbnormalityDetector:
    """
    Detects various brain abnormalities using deep learning and computer vision.
    
    Detected abnormalities include:
    - Tumors and masses
    - Lesions (MS plaques, infarcts, hemorrhages)
    - Atrophy and volume loss
    - Mass effect and midline shift
    - Edema and swelling
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize abnormality detector.
        
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
        self.abnormality_model = None
        self.attention_model = None
        self._load_models()
        
        # Abnormality types
        self.abnormality_types = {
            0: 'normal',
            1: 'tumor',
            2: 'lesion',
            3: 'atrophy',
            4: 'mass_effect',
            5: 'edema'
        }
    
    def _load_models(self):
        """Load pre-trained abnormality detection models."""
        try:
            # Load main abnormality detection model
            model_file = os.path.join(self.model_path, 'abnormality_detection_model.pth')
            if os.path.exists(model_file):
                self.abnormality_model = AbnormalityDetector3D(in_channels=1, num_classes=6)
                self.abnormality_model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.abnormality_model.to(self.device)
                self.abnormality_model.eval()
                print("✅ Loaded abnormality detection model")
            else:
                print("⚠️ No pre-trained abnormality model found. Using rule-based detection.")
                self.abnormality_model = None
            
            # Load attention model
            attention_model_file = os.path.join(self.model_path, 'attention_model.pth')
            if os.path.exists(attention_model_file):
                self.attention_model = AttentionModule(in_channels=1)
                self.attention_model.load_state_dict(torch.load(attention_model_file, map_location=self.device))
                self.attention_model.to(self.device)
                self.attention_model.eval()
                print("✅ Loaded attention model")
            else:
                print("⚠️ No pre-trained attention model found.")
                self.attention_model = None
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.abnormality_model = None
            self.attention_model = None
    
    def detect_abnormalities(self, 
                           scan_data: Dict, 
                           segmentation_results: Dict) -> Dict:
        """
        Detect abnormalities in brain scan.
        
        Args:
            scan_data: Dictionary containing scan data
            segmentation_results: Results from brain segmentation
            
        Returns:
            Dictionary containing detected abnormalities
        """
        data = scan_data['data']
        segmentation_mask = segmentation_results['segmentation_mask']
        
        # Preprocess data
        processed_data = self._preprocess_data(data, segmentation_mask)
        
        # Detect abnormalities
        if self.abnormality_model is not None:
            abnormalities = self._deep_learning_detection(processed_data)
        else:
            abnormalities = self._rule_based_detection(processed_data, segmentation_results)
        
        # Calculate measurements
        measurements = self._calculate_abnormality_measurements(abnormalities, scan_data)
        
        return {
            'abnormalities': abnormalities,
            'measurements': measurements
        }
    
    def _preprocess_data(self, data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Preprocess data for abnormality detection."""
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min())
        
        # Apply brain mask
        data = data * brain_mask
        
        # Apply Gaussian smoothing
        data = ndimage.gaussian_filter(data, sigma=1.0)
        
        return data
    
    def _deep_learning_detection(self, data: np.ndarray) -> Dict:
        """Detect abnormalities using deep learning model."""
        # Prepare input tensor
        input_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.abnormality_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            confidence_scores = probabilities.cpu().numpy()[0]
        
        # Apply attention if available
        attention_weights = None
        if self.attention_model is not None:
            with torch.no_grad():
                _, attention_weights = self.attention_model(input_tensor)
                attention_weights = attention_weights.cpu().numpy()[0, 0]
        
        # Extract abnormalities
        abnormalities = {}
        for i, (abnormality_type, confidence) in enumerate(zip(self.abnormality_types.values(), confidence_scores)):
            if confidence > 0.3:  # Threshold for detection
                abnormalities[abnormality_type] = {
                    'confidence': float(confidence),
                    'detected': True,
                    'attention_weights': attention_weights if attention_weights is not None else None
                }
        
        return abnormalities
    
    def _rule_based_detection(self, data: np.ndarray, segmentation_results: Dict) -> Dict:
        """Detect abnormalities using rule-based approach."""
        abnormalities = {}
        
        # Detect tumors/masses
        tumors = self._detect_tumors(data)
        if tumors['detected']:
            abnormalities['tumor'] = tumors
        
        # Detect lesions
        lesions = self._detect_lesions(data)
        if lesions['detected']:
            abnormalities['lesion'] = lesions
        
        # Detect atrophy
        atrophy = self._detect_atrophy(data, segmentation_results)
        if atrophy['detected']:
            abnormalities['atrophy'] = atrophy
        
        # Detect mass effect
        mass_effect = self._detect_mass_effect(data, segmentation_results)
        if mass_effect['detected']:
            abnormalities['mass_effect'] = mass_effect
        
        # Detect edema
        edema = self._detect_edema(data, segmentation_results)
        if edema['detected']:
            abnormalities['edema'] = edema
        
        return abnormalities
    
    def _detect_tumors(self, data: np.ndarray) -> Dict:
        """Detect tumors and masses."""
        # Apply intensity thresholding for potential tumors
        # Tumors often appear as hyperintense or hypointense regions
        high_intensity_threshold = np.percentile(data[data > 0], 95)
        low_intensity_threshold = np.percentile(data[data > 0], 5)
        
        # Find high intensity regions (potential tumors)
        high_intensity_regions = data > high_intensity_threshold
        low_intensity_regions = data < low_intensity_threshold
        
        # Morphological operations to clean up
        high_intensity_regions = morphology.remove_small_objects(high_intensity_regions, min_size=50)
        low_intensity_regions = morphology.remove_small_objects(low_intensity_regions, min_size=50)
        
        # Combine potential tumor regions
        potential_tumors = high_intensity_regions | low_intensity_regions
        
        if np.any(potential_tumors):
            # Calculate tumor properties
            tumor_props = measure.regionprops(potential_tumors.astype(int))
            
            tumor_info = {
                'detected': True,
                'confidence': 0.7,  # Moderate confidence for rule-based
                'count': len(tumor_props),
                'locations': [],
                'sizes': [],
                'volumes': []
            }
            
            for prop in tumor_props:
                tumor_info['locations'].append(prop.centroid)
                tumor_info['sizes'].append(prop.area)
                tumor_info['volumes'].append(prop.area)  # Assuming 1mm³ voxels
            
            return tumor_info
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_lesions(self, data: np.ndarray) -> Dict:
        """Detect lesions (MS plaques, infarcts, etc.)."""
        # Lesions often appear as small, well-defined regions with different intensity
        # Apply edge detection to find well-defined boundaries
        edges = filters.sobel(data)
        
        # Find regions with high edge density (potential lesions)
        edge_threshold = np.percentile(edges[edges > 0], 90)
        potential_lesions = edges > edge_threshold
        
        # Apply morphological operations
        potential_lesions = morphology.remove_small_objects(potential_lesions, min_size=10)
        potential_lesions = morphology.binary_closing(potential_lesions, morphology.ball(2))
        
        if np.any(potential_lesions):
            lesion_props = measure.regionprops(potential_lesions.astype(int))
            
            lesion_info = {
                'detected': True,
                'confidence': 0.6,
                'count': len(lesion_props),
                'locations': [prop.centroid for prop in lesion_props],
                'sizes': [prop.area for prop in lesion_props]
            }
            
            return lesion_info
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_atrophy(self, data: np.ndarray, segmentation_results: Dict) -> Dict:
        """Detect brain atrophy."""
        # Atrophy manifests as volume loss and ventricular enlargement
        measurements = segmentation_results.get('measurements', {})
        
        # Check for reduced brain volume
        total_brain_volume = measurements.get('total_brain_volume', 0)
        expected_volume = 1400  # Expected brain volume in cm³
        
        # Check for increased CSF volume (ventricular enlargement)
        csf_volume = measurements.get('csf_volume', 0)
        csf_ratio = csf_volume / max(total_brain_volume, 1)
        
        atrophy_detected = False
        atrophy_indicators = []
        
        if total_brain_volume < expected_volume * 0.9:  # 10% volume loss
            atrophy_detected = True
            atrophy_indicators.append('reduced_brain_volume')
        
        if csf_ratio > 0.15:  # Increased CSF ratio
            atrophy_detected = True
            atrophy_indicators.append('ventricular_enlargement')
        
        if atrophy_detected:
            return {
                'detected': True,
                'confidence': 0.8,
                'indicators': atrophy_indicators,
                'brain_volume': total_brain_volume,
                'csf_ratio': csf_ratio
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_mass_effect(self, data: np.ndarray, segmentation_results: Dict) -> Dict:
        """Detect mass effect and midline shift."""
        # Mass effect causes displacement of brain structures
        # Check for asymmetry in brain hemispheres
        
        # Calculate midline
        height, width, depth = data.shape
        midline = width // 2
        
        # Calculate left and right hemisphere volumes
        left_hemisphere = data[:, :midline, :]
        right_hemisphere = data[:, midline:, :]
        
        left_volume = np.sum(left_hemisphere > 0)
        right_volume = np.sum(right_hemisphere > 0)
        
        # Calculate asymmetry
        total_volume = left_volume + right_volume
        asymmetry_ratio = abs(left_volume - right_volume) / max(total_volume, 1)
        
        if asymmetry_ratio > 0.1:  # 10% asymmetry threshold
            return {
                'detected': True,
                'confidence': 0.7,
                'asymmetry_ratio': asymmetry_ratio,
                'left_volume': left_volume,
                'right_volume': right_volume
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_edema(self, data: np.ndarray, segmentation_results: Dict) -> Dict:
        """Detect brain edema (swelling)."""
        # Edema appears as increased signal intensity around lesions or tumors
        # Look for regions with increased intensity and blurring
        
        # Apply Gaussian blur to detect edema-like regions
        blurred = ndimage.gaussian_filter(data, sigma=2.0)
        
        # Find regions where original data is significantly different from blurred
        # (indicating edema-like changes)
        difference = np.abs(data - blurred)
        edema_threshold = np.percentile(difference[difference > 0], 85)
        
        potential_edema = difference > edema_threshold
        
        # Apply morphological operations
        potential_edema = morphology.remove_small_objects(potential_edema, min_size=100)
        potential_edema = morphology.binary_closing(potential_edema, morphology.ball(3))
        
        if np.any(potential_edema):
            edema_props = measure.regionprops(potential_edema.astype(int))
            
            return {
                'detected': True,
                'confidence': 0.6,
                'count': len(edema_props),
                'locations': [prop.centroid for prop in edema_props],
                'volumes': [prop.area for prop in edema_props]
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _calculate_abnormality_measurements(self, abnormalities: Dict, scan_data: Dict) -> Dict:
        """Calculate quantitative measurements for abnormalities."""
        measurements = {
            'total_abnormalities': len(abnormalities),
            'severity_score': 0.0,
            'abnormality_details': {}
        }
        
        severity_score = 0.0
        
        for abnormality_type, info in abnormalities.items():
            if info.get('detected', False):
                confidence = info.get('confidence', 0.0)
                severity_score += confidence
                
                # Calculate specific measurements for each abnormality type
                if abnormality_type == 'tumor':
                    measurements['abnormality_details']['tumor'] = {
                        'count': info.get('count', 0),
                        'total_volume': sum(info.get('volumes', [])),
                        'largest_size': max(info.get('sizes', [0]))
                    }
                
                elif abnormality_type == 'lesion':
                    measurements['abnormality_details']['lesion'] = {
                        'count': info.get('count', 0),
                        'total_volume': sum(info.get('sizes', []))
                    }
                
                elif abnormality_type == 'atrophy':
                    measurements['abnormality_details']['atrophy'] = {
                        'brain_volume': info.get('brain_volume', 0),
                        'csf_ratio': info.get('csf_ratio', 0),
                        'indicators': info.get('indicators', [])
                    }
                
                elif abnormality_type == 'mass_effect':
                    measurements['abnormality_details']['mass_effect'] = {
                        'asymmetry_ratio': info.get('asymmetry_ratio', 0),
                        'left_volume': info.get('left_volume', 0),
                        'right_volume': info.get('right_volume', 0)
                    }
                
                elif abnormality_type == 'edema':
                    measurements['abnormality_details']['edema'] = {
                        'count': info.get('count', 0),
                        'total_volume': sum(info.get('volumes', []))
                    }
        
        measurements['severity_score'] = min(severity_score, 1.0)
        
        return measurements 