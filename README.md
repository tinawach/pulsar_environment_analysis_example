# Pulsar Environment Analysis with IACTs – Example Project

This repository showcases **example workflows and code skeletons** developed during my research on high-energy gamma-ray emission from pulsar environments, using data from the H.E.S.S. (High Energy Stereoscopic System) telescope array.
The purpose of this repo is to present the structure and methodology of my work, while omitting proprietary data, custom tools, and unpublished analysis code.

---

## 🧠 Project Overview

My research focused on analyzing pulsar environments in the TeV (tera-electronvolt) energy regime, which required custom techniques to overcome instrumental limitations due to the limited field of view of the telescopes compared to the extension of the sources.
The full analysis spans from raw observational data to physical modeling and population-level insights.

This repository includes **representative examples and code skeletons** across the key phases of this work:

---

## 📁 What's Inside

### 1. **example_source_analysis** 
- Code for the background estimation technique developed to improve background estimation in imaging atmospheric Cherenkov telescope (IACT) data.
- Enables analysis of extended gamma-ray structures beyond traditional IACT capabilities.
- Code shows dataset construction, masking, safe cut application, and model stacking.
-  Demonstrates the application of the background method to a well-known gamma-ray source.
- Includes spatial and spectral data analysis, significance map generation, and signal-to-background evaluation.

> 📰 Related publication: 	https://doi.org/10.1051/0004-6361/202451020


---

### 2. **example_source_modeling**
- Example of modeling a pulsar wind nebula (PWN) spectrum using leptonic and hadronic particle models to describe the gamma-ray emission via [GAMERA](https://github.com/libgamera/GAMERA).
- Fits are performed using Bayesian inference (MCMC chains) for parameter estimation.
- Shows integration of physical modeling with statistical inference.

> 📰 Related publication: https://doi.org/10.1051/0004-6361/202348374

---

### 5. **PWN Population Study Skeleton**
- Outlines the pipeline used to search for TeV gamma-ray emission around all pulsars in the H.E.S.S. field of view (spanning 19 years of data).
- While the full implementation is proprietary, the skeleton illustrates:
  - Automated dataset building
  - Mask generation
  - Model fitting and validation
  - Batch analysis workflows

---

## Disclaimer

This repository does not include:
- Real observational data
- Full analysis scripts
- Custom internal tooling
- Source-specific results that are not already published

All examples are simplified and restructured to respect data privacy and collaboration policies. 
They are not capable of reproducing scientific results and can not be run.



