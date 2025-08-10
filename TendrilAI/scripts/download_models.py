#!/usr/bin/env python3
"""
Model Download Script
====================

This script downloads pre-trained models for the brain analysis system.
Since we don't have actual pre-trained models, this script creates placeholder
models and demonstrates the expected structure.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_analyzer.segmentation import UNet3D
from brain_analyzer.abnormality_detection import AbnormalityDetector3D, AttentionModule

def create_placeholder_models():
    """Create placeholder models for demonstration."""
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Creating models directory: {models_dir}")
    
    # Create brain segmentation model
    print("🧠 Creating brain segmentation model...")
    seg_model = UNet3D(in_channels=1, out_channels=4)
    
    # Initialize with random weights
    for param in seg_model.parameters():
        nn.init.xavier_uniform_(param)
    
    # Save model
    seg_model_path = models_dir / "brain_segmentation_model.pth"
    torch.save(seg_model.state_dict(), seg_model_path)
    print(f"✅ Saved segmentation model: {seg_model_path}")
    
    # Create brain regions model
    print("🗺️ Creating brain regions model...")
    regions_model = UNet3D(in_channels=1, out_channels=14)  # 14 brain regions
    
    # Initialize with random weights
    for param in regions_model.parameters():
        nn.init.xavier_uniform_(param)
    
    # Save model
    regions_model_path = models_dir / "brain_regions_model.pth"
    torch.save(regions_model.state_dict(), regions_model_path)
    print(f"✅ Saved regions model: {regions_model_path}")
    
    # Create abnormality detection model
    print("⚠️ Creating abnormality detection model...")
    ab_model = AbnormalityDetector3D(in_channels=1, num_classes=6)
    
    # Initialize with random weights
    for param in ab_model.parameters():
        nn.init.xavier_uniform_(param)
    
    # Save model
    ab_model_path = models_dir / "abnormality_detection_model.pth"
    torch.save(ab_model.state_dict(), ab_model_path)
    print(f"✅ Saved abnormality model: {ab_model_path}")
    
    # Create attention model
    print("👁️ Creating attention model...")
    attention_model = AttentionModule(in_channels=1)
    
    # Initialize with random weights
    for param in attention_model.parameters():
        nn.init.xavier_uniform_(param)
    
    # Save model
    attention_model_path = models_dir / "attention_model.pth"
    torch.save(attention_model.state_dict(), attention_model_path)
    print(f"✅ Saved attention model: {attention_model_path}")
    
    # Create model info file
    model_info = {
        "models": {
            "brain_segmentation_model.pth": {
                "type": "UNet3D",
                "input_channels": 1,
                "output_channels": 4,
                "description": "Brain tissue segmentation (CSF, gray matter, white matter)",
                "version": "1.0.0"
            },
            "brain_regions_model.pth": {
                "type": "UNet3D",
                "input_channels": 1,
                "output_channels": 14,
                "description": "Brain region segmentation (lobes, subcortical structures)",
                "version": "1.0.0"
            },
            "abnormality_detection_model.pth": {
                "type": "AbnormalityDetector3D",
                "input_channels": 1,
                "num_classes": 6,
                "description": "Abnormality detection (tumor, lesion, atrophy, etc.)",
                "version": "1.0.0"
            },
            "attention_model.pth": {
                "type": "AttentionModule",
                "input_channels": 1,
                "description": "Attention mechanism for focusing on suspicious regions",
                "version": "1.0.0"
            }
        },
        "total_size_mb": 0,  # Will be calculated
        "created_date": "2024-01-01",
        "framework": "PyTorch",
        "notes": "These are placeholder models for demonstration. Real models would be trained on large datasets."
    }
    
    import json
    model_info_path = models_dir / "model_info.json"
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Saved model info: {model_info_path}")
    
    # Calculate total size
    total_size = 0
    for model_file in models_dir.glob("*.pth"):
        total_size += model_file.stat().st_size
    
    print(f"📊 Total model size: {total_size / 1024 / 1024:.1f} MB")
    
    return models_dir

def verify_models():
    """Verify that models can be loaded correctly."""
    
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        print("❌ Models directory not found. Run create_placeholder_models() first.")
        return False
    
    print("🔍 Verifying models...")
    
    try:
        # Test segmentation model
        seg_model = UNet3D(in_channels=1, out_channels=4)
        seg_model.load_state_dict(torch.load(models_dir / "brain_segmentation_model.pth"))
        print("✅ Segmentation model loaded successfully")
        
        # Test regions model
        regions_model = UNet3D(in_channels=1, out_channels=14)
        regions_model.load_state_dict(torch.load(models_dir / "brain_regions_model.pth"))
        print("✅ Regions model loaded successfully")
        
        # Test abnormality model
        ab_model = AbnormalityDetector3D(in_channels=1, num_classes=6)
        ab_model.load_state_dict(torch.load(models_dir / "abnormality_detection_model.pth"))
        print("✅ Abnormality model loaded successfully")
        
        # Test attention model
        attention_model = AttentionModule(in_channels=1)
        attention_model.load_state_dict(torch.load(models_dir / "attention_model.pth"))
        print("✅ Attention model loaded successfully")
        
        print("🎉 All models verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main function."""
    
    print("🚀 MRI Brain Analysis - Model Download Script")
    print("=" * 50)
    
    # Create models directory
    models_dir = create_placeholder_models()
    
    # Verify models
    if verify_models():
        print("\n✅ Model setup completed successfully!")
        print(f"📁 Models are available in: {models_dir}")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the web interface: streamlit run app/streamlit_app.py")
        print("3. Or run the Gradio interface: python app/gradio_app.py")
        print("4. Or use the Python API directly")
    else:
        print("\n❌ Model setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 