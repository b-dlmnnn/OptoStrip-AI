# 🧬 OptoStrip-AI: Single-Image Assay Reader & White-Balance Calibrator

A computational computer vision web application designed for **reference-card white-balance calibration ($\Delta E$)**, **automated tilt rectification**, and **8-well colorimetric LAMP/PCR strip quality control (QC)**.

## Key Engineering Features
- **Pre-Flight Image Quality Gate:** Automatically detects motion blur (Laplacian variance), specular clipping (>252 AU), and underexposure prior to measurement.
- **Contour-Based Tilt Rectification:** Locates the left white reference card and corrects camera tilt angles via affine transformation.
- **Reference-Card Color Calibration:** Computes per-channel RGB gain factors ($Gain_c = 240 / \mu_{ref,c}$) to neutralize ambient lighting shifts (tungsten, cool shade, dim lighting).
- **Automated Batch QC & CSV Export:** Validates Positive Control (`T1`) and No-Template Control (`T2 / NTC`) before reporting sample results (`T3–T8`).
- **Multi-Lighting Statistical Benchmark:** Evaluates $\Delta E$ color error reduction and classification accuracy across 80 multi-lighting trials.

## Academic Transparency & Limitations
- **Disclaimer:** This is a computational and optical validation prototype using physics-informed synthetic strip images and non-biological colored solutions. No biological nucleic acids or wet-lab experiments were conducted.
- **Known Limitations:** Direct flash specular clipping (>252 AU) cannot be recovered mathematically; steep camera tilt (>25°) requires manual re-alignment; currently calibrated for Phenol Red (yellow/pink) colorimetric assays.
