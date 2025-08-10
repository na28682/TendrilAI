"""
Report Generation Module
=======================

This module generates comprehensive medical reports based on brain analysis results.
Reports include findings, measurements, abnormalities, and recommendations.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

class ReportGenerator:
    """
    Generates comprehensive medical reports for brain analysis results.
    
    Report types include:
    - Summary report
    - Detailed analysis report
    - Abnormality-specific report
    - Longitudinal comparison report
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.report_templates = self._load_report_templates()
    
    def _load_report_templates(self) -> Dict:
        """Load report templates."""
        return {
            'summary': self._get_summary_template(),
            'detailed': self._get_detailed_template(),
            'abnormality': self._get_abnormality_template(),
            'longitudinal': self._get_longitudinal_template()
        }
    
    def generate_report(self, 
                       results: 'AnalysisResults',
                       report_type: str = 'detailed',
                       include_visualizations: bool = True) -> str:
        """
        Generate analysis report.
        
        Args:
            results: AnalysisResults object
            report_type: Type of report ('summary', 'detailed', 'abnormality', 'longitudinal')
            include_visualizations: Whether to include visualization references
            
        Returns:
            Formatted report string
        """
        if report_type not in self.report_templates:
            report_type = 'detailed'
        
        template = self.report_templates[report_type]
        
        # Extract data for report
        report_data = self._extract_report_data(results)
        
        # Generate report using template
        report = self._fill_template(template, report_data)
        
        # Add visualization references if requested
        if include_visualizations and results.visualizations:
            report += self._add_visualization_references(results.visualizations)
        
        return report
    
    def _extract_report_data(self, results: 'AnalysisResults') -> Dict:
        """Extract data for report generation."""
        scan_data = results.scan_data
        segmentation_results = results.segmentation_results
        abnormality_results = results.abnormality_results
        
        # Basic scan information
        scan_info = {
            'scan_path': str(results.scan_path),
            'scan_shape': scan_data.get('shape', 'Unknown'),
            'voxel_size': scan_data.get('voxel_size', [1, 1, 1]),
            'orientation': scan_data.get('orientation', 'Unknown')
        }
        
        # Segmentation measurements
        seg_measurements = segmentation_results.get('measurements', {})
        
        # Abnormality information
        abnormalities = abnormality_results.get('abnormalities', {})
        ab_measurements = abnormality_results.get('measurements', {})
        
        # Brain regions
        regions = segmentation_results.get('regions', {})
        
        return {
            'scan_info': scan_info,
            'segmentation_measurements': seg_measurements,
            'abnormalities': abnormalities,
            'abnormality_measurements': ab_measurements,
            'brain_regions': regions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_version': '1.0.0'
        }
    
    def _fill_template(self, template: str, data: Dict) -> str:
        """Fill template with data."""
        report = template
        
        # Replace placeholders with actual data
        report = report.replace('{{TIMESTAMP}}', data['timestamp'])
        report = report.replace('{{VERSION}}', data['analysis_version'])
        
        # Scan information
        scan_info = data['scan_info']
        report = report.replace('{{SCAN_PATH}}', scan_info['scan_path'])
        report = report.replace('{{SCAN_SHAPE}}', str(scan_info['scan_shape']))
        report = report.replace('{{VOXEL_SIZE}}', str(scan_info['voxel_size']))
        
        # Segmentation measurements
        seg_meas = data['segmentation_measurements']
        report = report.replace('{{TOTAL_BRAIN_VOLUME}}', f"{seg_meas.get('total_brain_volume', 0):.2f}")
        report = report.replace('{{GRAY_MATTER_VOLUME}}', f"{seg_meas.get('gray_matter_volume', 0):.2f}")
        report = report.replace('{{WHITE_MATTER_VOLUME}}', f"{seg_meas.get('white_matter_volume', 0):.2f}")
        report = report.replace('{{CSF_VOLUME}}', f"{seg_meas.get('csf_volume', 0):.2f}")
        report = report.replace('{{GRAY_WHITE_RATIO}}', f"{seg_meas.get('gray_white_ratio', 0):.2f}")
        
        # Abnormality information
        abnormalities = data['abnormalities']
        ab_meas = data['abnormality_measurements']
        
        # Generate abnormality summary
        abnormality_summary = self._generate_abnormality_summary(abnormalities, ab_meas)
        report = report.replace('{{ABNORMALITY_SUMMARY}}', abnormality_summary)
        
        # Generate brain region summary
        region_summary = self._generate_region_summary(data['brain_regions'])
        report = report.replace('{{REGION_SUMMARY}}', region_summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(abnormalities, ab_meas)
        report = report.replace('{{RECOMMENDATIONS}}', recommendations)
        
        return report
    
    def _generate_abnormality_summary(self, abnormalities: Dict, measurements: Dict) -> str:
        """Generate abnormality summary text."""
        if not abnormalities:
            return "No abnormalities detected in this scan."
        
        summary = []
        total_abnormalities = measurements.get('total_abnormalities', 0)
        severity_score = measurements.get('severity_score', 0.0)
        
        summary.append(f"Total abnormalities detected: {total_abnormalities}")
        summary.append(f"Overall severity score: {severity_score:.2f}")
        summary.append("")
        
        for abnormality_type, info in abnormalities.items():
            if info.get('detected', False):
                confidence = info.get('confidence', 0.0)
                summary.append(f"- {abnormality_type.replace('_', ' ').title()}:")
                summary.append(f"  Confidence: {confidence:.2f}")
                
                # Add specific details for each abnormality type
                if abnormality_type == 'tumor':
                    count = info.get('count', 0)
                    total_volume = sum(info.get('volumes', []))
                    summary.append(f"  Count: {count}")
                    summary.append(f"  Total volume: {total_volume:.2f} mm³")
                
                elif abnormality_type == 'lesion':
                    count = info.get('count', 0)
                    summary.append(f"  Count: {count}")
                
                elif abnormality_type == 'atrophy':
                    brain_volume = info.get('brain_volume', 0)
                    csf_ratio = info.get('csf_ratio', 0)
                    indicators = info.get('indicators', [])
                    summary.append(f"  Brain volume: {brain_volume:.2f} cm³")
                    summary.append(f"  CSF ratio: {csf_ratio:.2f}")
                    summary.append(f"  Indicators: {', '.join(indicators)}")
                
                elif abnormality_type == 'mass_effect':
                    asymmetry_ratio = info.get('asymmetry_ratio', 0)
                    summary.append(f"  Asymmetry ratio: {asymmetry_ratio:.2f}")
                
                elif abnormality_type == 'edema':
                    count = info.get('count', 0)
                    total_volume = sum(info.get('volumes', []))
                    summary.append(f"  Count: {count}")
                    summary.append(f"  Total volume: {total_volume:.2f} mm³")
                
                summary.append("")
        
        return "\n".join(summary)
    
    def _generate_region_summary(self, regions: Dict) -> str:
        """Generate brain region summary."""
        if not regions:
            return "No brain region information available."
        
        summary = []
        summary.append("Brain Region Analysis:")
        summary.append("")
        
        for region_name, region_info in regions.items():
            volume = region_info.get('volume', 0)
            percentage = region_info.get('percentage', 0)
            summary.append(f"- {region_name.replace('_', ' ').title()}:")
            summary.append(f"  Volume: {volume:.2f} mm³")
            summary.append(f"  Percentage: {percentage:.1f}%")
            summary.append("")
        
        return "\n".join(summary)
    
    def _generate_recommendations(self, abnormalities: Dict, measurements: Dict) -> str:
        """Generate clinical recommendations."""
        recommendations = []
        
        if not abnormalities:
            recommendations.append("- No immediate clinical concerns detected.")
            recommendations.append("- Consider routine follow-up imaging as per standard protocols.")
            return "\n".join(recommendations)
        
        severity_score = measurements.get('severity_score', 0.0)
        
        if severity_score > 0.7:
            recommendations.append("- URGENT: High severity abnormalities detected.")
            recommendations.append("- Immediate clinical evaluation recommended.")
            recommendations.append("- Consider emergency imaging protocols.")
        elif severity_score > 0.4:
            recommendations.append("- MODERATE: Significant abnormalities detected.")
            recommendations.append("- Prompt clinical evaluation recommended.")
            recommendations.append("- Consider specialized imaging studies.")
        else:
            recommendations.append("- MILD: Minor abnormalities detected.")
            recommendations.append("- Routine clinical follow-up recommended.")
        
        # Specific recommendations based on abnormality types
        for abnormality_type, info in abnormalities.items():
            if info.get('detected', False):
                if abnormality_type == 'tumor':
                    recommendations.append("- Tumor detected: Consider biopsy and histopathological analysis.")
                    recommendations.append("- Consult with neuro-oncology specialist.")
                    recommendations.append("- Consider advanced imaging (contrast-enhanced MRI, spectroscopy).")
                
                elif abnormality_type == 'lesion':
                    recommendations.append("- Lesions detected: Consider differential diagnosis.")
                    recommendations.append("- Rule out demyelinating disease, infection, or vascular causes.")
                    recommendations.append("- Consider follow-up imaging to assess progression.")
                
                elif abnormality_type == 'atrophy':
                    recommendations.append("- Brain atrophy detected: Consider neurodegenerative workup.")
                    recommendations.append("- Evaluate for cognitive decline and functional status.")
                    recommendations.append("- Consider neuropsychological assessment.")
                
                elif abnormality_type == 'mass_effect':
                    recommendations.append("- Mass effect detected: Monitor for neurological symptoms.")
                    recommendations.append("- Consider intracranial pressure monitoring if indicated.")
                    recommendations.append("- Evaluate for surgical intervention if necessary.")
                
                elif abnormality_type == 'edema':
                    recommendations.append("- Edema detected: Consider underlying cause evaluation.")
                    recommendations.append("- Monitor for neurological deterioration.")
                    recommendations.append("- Consider anti-edema therapy if indicated.")
        
        recommendations.append("")
        recommendations.append("DISCLAIMER: This report is for research purposes only.")
        recommendations.append("Clinical decisions should be made by qualified medical professionals.")
        
        return "\n".join(recommendations)
    
    def _add_visualization_references(self, visualizations: Dict) -> str:
        """Add visualization references to report."""
        refs = []
        refs.append("")
        refs.append("VISUALIZATIONS:")
        refs.append("The following visualizations were generated:")
        
        for name, viz in visualizations.items():
            if hasattr(viz, 'savefig'):
                refs.append(f"- {name.replace('_', ' ').title()}: Static image")
            elif hasattr(viz, 'write_html'):
                refs.append(f"- {name.replace('_', ' ').title()}: Interactive HTML")
            elif hasattr(viz, 'screenshot'):
                refs.append(f"- {name.replace('_', ' ').title()}: 3D visualization")
        
        return "\n".join(refs)
    
    def _get_summary_template(self) -> str:
        """Get summary report template."""
        return """BRAIN MRI ANALYSIS REPORT - SUMMARY
Generated: {{TIMESTAMP}}
Analysis Version: {{VERSION}}

SCAN INFORMATION:
- File: {{SCAN_PATH}}
- Dimensions: {{SCAN_SHAPE}}
- Voxel Size: {{VOXEL_SIZE}} mm

KEY FINDINGS:
{{ABNORMALITY_SUMMARY}}

BRAIN MEASUREMENTS:
- Total Brain Volume: {{TOTAL_BRAIN_VOLUME}} cm³
- Gray Matter Volume: {{GRAY_MATTER_VOLUME}} cm³
- White Matter Volume: {{WHITE_MATTER_VOLUME}} cm³
- CSF Volume: {{CSF_VOLUME}} cm³
- Gray/White Matter Ratio: {{GRAY_WHITE_RATIO}}

RECOMMENDATIONS:
{{RECOMMENDATIONS}}
"""
    
    def _get_detailed_template(self) -> str:
        """Get detailed report template."""
        return """BRAIN MRI ANALYSIS REPORT - DETAILED
Generated: {{TIMESTAMP}}
Analysis Version: {{VERSION}}

SCAN INFORMATION:
- File: {{SCAN_PATH}}
- Dimensions: {{SCAN_SHAPE}}
- Voxel Size: {{VOXEL_SIZE}} mm
- Orientation: {{ORIENTATION}}

BRAIN SEGMENTATION RESULTS:
- Total Brain Volume: {{TOTAL_BRAIN_VOLUME}} cm³
- Gray Matter Volume: {{GRAY_MATTER_VOLUME}} cm³
- White Matter Volume: {{WHITE_MATTER_VOLUME}} cm³
- CSF Volume: {{CSF_VOLUME}} cm³
- Gray/White Matter Ratio: {{GRAY_WHITE_RATIO}}

BRAIN REGION ANALYSIS:
{{REGION_SUMMARY}}

ABNORMALITY DETECTION RESULTS:
{{ABNORMALITY_SUMMARY}}

CLINICAL RECOMMENDATIONS:
{{RECOMMENDATIONS}}

TECHNICAL NOTES:
- Analysis performed using deep learning-based segmentation
- Abnormality detection uses both rule-based and AI methods
- All measurements are approximate and should be validated clinically
- This report is for research and educational purposes only

QUALITY ASSURANCE:
- Segmentation quality: Automated assessment
- Abnormality confidence: Based on model predictions
- Report generated: {{TIMESTAMP}}
"""
    
    def _get_abnormality_template(self) -> str:
        """Get abnormality-specific report template."""
        return """BRAIN MRI ABNORMALITY REPORT
Generated: {{TIMESTAMP}}
Analysis Version: {{VERSION}}

FOCUSED ABNORMALITY ANALYSIS:
{{ABNORMALITY_SUMMARY}}

DETAILED MEASUREMENTS:
- Total abnormalities: {{TOTAL_ABNORMALITIES}}
- Severity score: {{SEVERITY_SCORE}}
- Confidence levels: See detailed breakdown above

CLINICAL IMPLICATIONS:
{{RECOMMENDATIONS}}

FOLLOW-UP PROTOCOLS:
- Immediate actions based on severity
- Recommended imaging intervals
- Specialist consultations as indicated

DISCLAIMER:
This automated analysis is for screening purposes only.
All findings require clinical correlation and professional interpretation.
"""
    
    def _get_longitudinal_template(self) -> str:
        """Get longitudinal comparison template."""
        return """BRAIN MRI LONGITUDINAL ANALYSIS REPORT
Generated: {{TIMESTAMP}}
Analysis Version: {{VERSION}}

LONGITUDINAL COMPARISON:
{{LONGITUDINAL_SUMMARY}}

CHANGE ANALYSIS:
{{CHANGE_ANALYSIS}}

PROGRESSION ASSESSMENT:
{{PROGRESSION_ASSESSMENT}}

CLINICAL RECOMMENDATIONS:
{{RECOMMENDATIONS}}

TREND ANALYSIS:
- Volume changes over time
- Abnormality progression
- Response to treatment (if applicable)

DISCLAIMER:
Longitudinal analysis requires consistent imaging protocols.
Changes may reflect technical factors rather than true progression.
"""
    
    def save_report(self, report: str, output_path: str):
        """Save report to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"💾 Report saved to: {output_path}")
    
    def export_json(self, results: 'AnalysisResults', output_path: str):
        """Export results as JSON for further processing."""
        data = self._extract_report_data(results)
        
        # Add raw data
        data['raw_results'] = {
            'segmentation_mask': results.segmentation_results.get('segmentation_mask', []).tolist(),
            'region_mask': results.segmentation_results.get('region_mask', []).tolist(),
            'abnormalities': results.abnormality_results.get('abnormalities', {}),
            'measurements': results.segmentation_results.get('measurements', {})
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 JSON export saved to: {output_path}") 