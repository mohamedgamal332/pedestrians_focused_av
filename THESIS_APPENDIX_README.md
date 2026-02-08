# Thesis Appendix Documentation

This directory contains a comprehensive LaTeX appendix for your thesis on the Pedestrian-Focused Autonomous Vehicle Research Platform.

## Files

- `thesis_appendix.tex` - Main LaTeX appendix document

## Compilation Instructions

### Using pdflatex (Recommended)

```bash
# Compile the document (run twice for proper references)
pdflatex thesis_appendix.tex
pdflatex thesis_appendix.tex
```

### Using LaTeX editors

The document can be compiled using any LaTeX editor:
- **Overleaf**: Upload `thesis_appendix.tex` to Overleaf (online, no installation needed)
- **TeXstudio**: Open and compile with F5
- **TeXmaker**: Open and compile with F1
- **VS Code**: Use LaTeX Workshop extension

### Online Compilation

You can use Overleaf (https://www.overleaf.com) to compile this document without installing LaTeX:
1. Create a new project on Overleaf
2. Upload `thesis_appendix.tex`
3. Click "Recompile"

## Document Contents

The appendix includes the following sections:

1. **Project Overview and Architecture**
   - System architecture
   - Directory structure

2. **Data Collection with CARLA Simulation**
   - Simulation environment
   - Stereo camera setup
   - Data collection pipeline

3. **Pose Estimation System**
   - RTMPose architecture
   - 17 COCO keypoints
   - Pose estimation pipeline

4. **Graph Convolutional Networks for Trajectory Prediction**
   - ST-GCN (Spatial-Temporal GCN)
   - CTR-GCN (Channel-wise Topology Refinement GCN)
   - CTR-GCN with Motion Stream
   - TE-GCN (Temporal Enhanced GCN)
   - SHT (Spatial-Hierarchical Transformer)
   - Model comparison and specifications

5. **Training Pipeline**
   - Data preparation
   - Training configuration
   - Training scripts

6. **Stereo Matching and Triangulation**
   - Pedestrian matching problem
   - Hungarian algorithm
   - 3D triangulation
   - Baseline performance

7. **Risk Assessment and Scoring**
   - Risk score formulation
   - Proximity, velocity, trajectory, and behavior risk

8. **Governor-Reflex Autonomous Control**
   - Architecture overview
   - Control pipeline

9. **Evaluation Metrics and Results**
   - Pose estimation metrics
   - Trajectory prediction metrics
   - Comparative results

10. **Implementation Details**
    - Software dependencies
    - Hardware requirements
    - Setup and installation

11. **Code Examples**
    - Loading and running GCN models
    - Stereo matching and triangulation
    - Risk score computation

12. **Experimental Results and Analysis**
    - Dataset statistics
    - Bone length consistency
    - Confidence vs. error correlation
    - Training convergence

13. **Discussion and Future Work**
    - Current limitations
    - Proposed improvements
    - Future research directions

## Customization

You can customize the document by:

1. **Modifying the title**: Edit the `\title{}` command
2. **Adding your name**: Edit the `\author{}` command
3. **Adding more sections**: Use `\section{}` and `\subsection{}`
4. **Including figures**: Place image files in the same directory and use:
   ```latex
   \begin{figure}[h]
   \centering
   \includegraphics[width=0.8\textwidth]{your_image.png}
   \caption{Your caption}
   \label{fig:your_label}
   \end{figure}
   ```

## Required LaTeX Packages

The document uses the following packages (usually included in standard LaTeX distributions):
- inputenc, babel
- amsmath, amsfonts, amssymb
- graphicx
- listings, xcolor
- hyperref
- geometry
- booktabs, longtable
- algorithm, algpseudocode
- subcaption

## Integration with Your Thesis

To integrate this appendix into your main thesis document:

1. **Option 1 - Include as separate document**:
   - Compile separately and include PDF in your thesis

2. **Option 2 - Copy sections**:
   - Copy relevant sections from `thesis_appendix.tex` into your main thesis LaTeX file
   - Place after your main content, before references

3. **Option 3 - Use \input or \include**:
   ```latex
   % In your main thesis file
   \appendix
   \input{thesis_appendix}
   ```

## Tips for Thesis Writing

1. **Adjust formatting**: Match the appendix formatting to your thesis template
2. **Add figure references**: Include actual figures from your work
3. **Update results**: Replace placeholder results with your actual experimental results
4. **Cite properly**: Ensure all references are properly cited in your bibliography
5. **Review with advisor**: Have your thesis advisor review the appendix structure

## Support

If you encounter any issues with compilation:
1. Ensure you have a complete LaTeX distribution installed (TeX Live, MiKTeX)
2. Check that all required packages are installed
3. Run pdflatex multiple times if references don't appear correctly
4. Check the `.log` file for specific error messages

## License

This appendix document is part of your thesis work and follows the same license as the main repository.
