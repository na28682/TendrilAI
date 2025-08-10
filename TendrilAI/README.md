# MRI Brain Analysis System

A comprehensive deep learning system for analyzing MRI brain scans to recognize brain regions and detect abnormalities.

## Features

### 🧠 Brain Region Recognition
- **Automatic segmentation** of brain structures (cerebrum, cerebellum, brainstem, ventricles)
- **Gray matter/white matter** differentiation
- **Cortical regions** identification (frontal, parietal, temporal, occipital lobes)
- **Subcortical structures** detection (thalamus, basal ganglia, hippocampus)

### ⚠️ Abnormality Detection
- **Tumor detection** with size and location estimation
- **Lesion identification** (MS plaques, infarcts, hemorrhages)
- **Atrophy detection** (cortical thinning, ventricular enlargement)
- **Mass effect** assessment
- **Edema detection** and quantification

### 📊 Analysis & Visualization
- **3D interactive visualization** of brain structures
- **Quantitative measurements** (volumes, ratios, symmetry)
- **Automated report generation** with findings and recommendations
- **Longitudinal analysis** for tracking changes over time

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TendrilAI

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if available)
python scripts/download_models.py
```

## Usage

### Basic Usage
```python
from brain_analyzer import BrainAnalyzer

# Initialize analyzer
analyzer = BrainAnalyzer()

# Analyze MRI scan
results = analyzer.analyze_scan("path/to/mri_scan.nii.gz")

# Get brain regions
regions = results.get_brain_regions()

# Get abnormalities
abnormalities = results.get_abnormalities()

# Generate report
report = results.generate_report()
```

### Web Interface
```bash
# Start Streamlit app
streamlit run app/streamlit_app.py

# Start Gradio interface
python app/gradio_app.py
```

## Project Structure

```
TendrilAI/
├── brain_analyzer/          # Core analysis modules
│   ├── __init__.py
│   ├── segmentation.py      # Brain region segmentation
│   ├── abnormality_detection.py  # Abnormality detection
│   ├── visualization.py     # 3D visualization
│   └── reporting.py        # Report generation
├── models/                  # Pre-trained models
├── data/                   # Sample data and datasets
├── app/                    # Web applications
│   ├── streamlit_app.py
│   └── gradio_app.py
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## Models

The system uses several pre-trained deep learning models:

1. **U-Net for Brain Segmentation** - Identifies brain regions
2. **3D CNN for Abnormality Detection** - Detects tumors, lesions, etc.
3. **Attention-based Model** - Focuses on suspicious areas
4. **Ensemble Model** - Combines multiple models for accuracy

## Medical Disclaimer

⚠️ **IMPORTANT**: This system is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult qualified medical professionals for medical decisions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{brain_analyzer,
  title={MRI Brain Analysis System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TendrilAI}
}
``` 