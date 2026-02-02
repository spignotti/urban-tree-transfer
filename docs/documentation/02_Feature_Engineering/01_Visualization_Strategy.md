# Phase 2 Visualization Strategy

**Phase:** 2 - Feature Engineering  
**Created:** 2026-01-28  
**Purpose:** Documentation and publication-quality plots

---

## Overview

All exploratory notebooks must produce publication-quality visualizations with consistent styling. Plots are saved both to Google Drive (during Colab execution) and committed to Git (for documentation and presentations).

---

## Plot Style Configuration

### Using the Plotting Utilities

**Always** import and setup at the beginning of each notebook:

```python
from urban_tree_transfer.utils.plotting import setup_plotting, save_figure, PUBLICATION_STYLE

# Setup consistent style
setup_plotting()
```

### Style Specifications

**Defined in:** `src/urban_tree_transfer/utils/plotting.py`

```python
PUBLICATION_STYLE = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (12, 7),
    "dpi_export": 300,
    "palette": "Set2",
}
```

**Applied settings:**

- Style: `seaborn-v0_8-whitegrid` (clean, professional)
- Figure size: 12×7 inches (good for presentations)
- Export DPI: 300 (publication quality)
- Color palette: `Set2` (colorblind-friendly)
- Tight bbox: Removes excess whitespace

---

## File Organization

### Directory Structure

```
Google Drive (during Colab execution):
/content/drive/MyDrive/urban-tree-transfer/
└── outputs/phase_2/
    ├── figures/                          # All plots
    │   ├── exp_01_temporal/              # Temporal analysis plots
    │   │   ├── jm_heatmap_berlin.png
    │   │   ├── jm_heatmap_leipzig.png
    │   │   ├── jm_monthly_comparison.png
    │   │   ├── jm_feature_groups.png
    │   │   └── temporal_selection_summary.png
    │   ├── exp_02_chm/                   # CHM assessment plots
    │   │   ├── chm_boxplot_per_genus.png
    │   │   ├── chm_cadastre_correlation.png
    │   │   ├── eta2_comparison.png
    │   │   ├── cohens_d_forest_plot.png
    │   │   └── chm_distribution_cities.png
    │   ├── exp_03_correlation/           # Correlation analysis plots
    │   │   ├── correlation_heatmap_spectral.png
    │   │   ├── correlation_heatmap_broadband.png
    │   │   ├── correlation_heatmap_rededge.png
    │   │   ├── correlation_heatmap_water.png
    │   │   ├── redundant_features_scatter.png
    │   │   └── variance_comparison.png
    │   ├── exp_04_outliers/              # Outlier analysis plots
    │   │   ├── outlier_rates_per_method.png
    │   │   ├── outlier_venn_diagram.png
    │   │   ├── zscore_sensitivity.png
    │   │   ├── mahalanobis_distribution.png
    │   │   ├── iqr_boxplots_per_genus.png
    │   │   └── hierarchical_decision_flow.png
    │   └── exp_05_spatial/               # Spatial analysis plots
    │       ├── morans_i_decay_curves.png
    │       ├── spatial_autocorrelation_heatmap.png
    │       ├── block_overlay_map_berlin.png
    │       ├── block_overlay_map_leipzig.png
    │       └── autocorrelation_comparison.png
    └── metadata/                          # JSON configs (separate from plots)

Git Repository (after manual sync):
outputs/phase_2/
├── figures/                               # Copy from Drive
│   └── [same structure as above]
└── metadata/                              # JSON configs
```

### Naming Convention

**Pattern:** `{analysis_type}_{plot_type}_{specifics}.png`

**Examples:**

- `jm_heatmap_berlin.png` - JM distance heatmap for Berlin
- `chm_boxplot_per_genus.png` - CHM distribution by genus
- `correlation_heatmap_spectral.png` - Spectral band correlations
- `morans_i_decay_curves.png` - Spatial autocorrelation decay

**Rules:**

- Lowercase with underscores
- Descriptive but concise
- Include city name if city-specific
- PNG format (universal, good for presentations)

---

## Standard Plot Types by Notebook

### exp_01_temporal_analysis.ipynb

**Required Plots:**

1. **JM Distance Heatmap (per city)**
   - X-axis: Months (1-12)
   - Y-axis: Feature groups (Spectral, Broadband VI, Red-edge VI, Water VI)
   - Color: Mean JM distance
   - Annotations: Selected months marked

2. **Monthly JM Comparison (Berlin vs Leipzig)**
   - Line plot with two lines (Berlin, Leipzig)
   - X-axis: Months
   - Y-axis: Mean JM distance
   - Error bars: Std across features

3. **Feature Group Discriminability**
   - Box plots per feature group
   - Comparison across months
   - Highlights best-performing groups

4. **Temporal Selection Summary**
   - Bar chart showing JM distance per month
   - Threshold line
   - Selected months highlighted

**Example Code:**

```python
from urban_tree_transfer.utils.plotting import setup_plotting, save_figure
import matplotlib.pyplot as plt

setup_plotting()

# Create plot
fig, ax = plt.subplots()
# ... plotting code ...

# Save to Drive
output_path = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal/jm_heatmap_berlin.png")
save_figure(fig, output_path)

print(f"✓ Saved: {output_path}")
```

---

### exp_02_chm_assessment.ipynb

**Required Plots:**

1. **CHM Distribution by Genus (per city)**
   - Box plots or violin plots
   - Grouped by genus
   - Faceted by city

2. **CHM vs Cadastre Height Correlation**
   - Scatter plot with regression line
   - Annotated with correlation coefficient
   - Separate colors per city

3. **Discriminative Power (η²) Comparison**
   - Bar chart comparing η² per city
   - Effect size interpretation guidelines

4. **Cohen's d Forest Plot**
   - Forest plot showing Cohen's d per genus
   - Confidence intervals
   - Reference lines for small/medium/large effects

5. **CHM Distribution Comparison (Cities)**
   - Overlaid histograms or density plots
   - Berlin vs Leipzig CHM distributions

---

### exp_03_correlation_analysis.ipynb

**Required Plots:**

1. **Correlation Heatmaps (per feature group)**
   - Spectral bands
   - Broadband VIs
   - Red-edge VIs
   - Water VIs
   - Annotated with correlation coefficients
   - Mask for |r| > 0.95 (redundancy threshold)

2. **Redundant Feature Pairs Scatter Plots**
   - Selected high-correlation pairs (r > 0.95)
   - Shows why features are redundant

3. **Variance Comparison**
   - Bar chart comparing variance of redundant pairs
   - Justifies which feature to keep

4. **Temporal Consistency**
   - Line plot showing correlation stability across months
   - Validates removal decisions

---

### exp_04_outlier_thresholds.ipynb

**Required Plots:**

1. **Outlier Detection Rates by Method**
   - Bar chart comparing Z-score, Mahalanobis, IQR
   - Per city and overall

2. **Venn Diagram**
   - Overlap between detection methods
   - Shows CRITICAL outliers (intersection)

3. **Z-Score Sensitivity Analysis**
   - Line plot: Removal rate vs threshold
   - Multiple lines for min_feature_count values

4. **Mahalanobis Distance Distribution**
   - Histogram with chi-squared critical value line
   - Per genus (faceted)

5. **IQR Box Plots (CHM)**
   - Box plots per genus × city
   - Tukey fences visualized

6. **Hierarchical Decision Flow**
   - Sankey diagram or flowchart
   - Shows how trees are categorized (CRITICAL, WARNING, OK)

---

### exp_05_spatial_autocorrelation.ipynb

**Required Plots:**

1. **Moran's I Decay Curves**
   - Line plot: Moran's I vs distance
   - Multiple lines for different features
   - Threshold line (Moran's I = 0.05)

2. **Spatial Autocorrelation Heatmap**
   - X-axis: Features
   - Y-axis: Distance lags
   - Color: Moran's I value

3. **Block Overlay Maps (per city)**
   - Map of tree locations
   - 500×500m grid overlay
   - Color-coded by train/val assignment

4. **Autocorrelation Comparison (Berlin vs Leipzig)**
   - Faceted plot comparing decay patterns
   - Validates consistent block size

---

## Plotting Best Practices

### 1. Always Use setup_plotting()

```python
from urban_tree_transfer.utils.plotting import setup_plotting

# At the beginning of visualization section
setup_plotting()
```

### 2. Consistent Color Usage

**For Cities:**

```python
city_colors = {
    "berlin": "#1f77b4",  # Blue
    "leipzig": "#ff7f0e",  # Orange
}
```

**For Genera (if needed):**

```python
# Use PUBLICATION_STYLE palette (Set2)
# Automatically handled by setup_plotting()
```

### 3. Informative Titles and Labels

```python
ax.set_title("JM Distance Across Months (Berlin)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Mean JM Distance", fontsize=12)
```

### 4. Legends and Annotations

```python
# Clear legend
ax.legend(loc="best", frameon=True, fontsize=10)

# Annotations for key findings
ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
```

### 5. Save with Metadata

```python
from pathlib import Path
from urban_tree_transfer.utils.plotting import save_figure

output_path = Path("/content/drive/.../exp_01_temporal/jm_heatmap_berlin.png")
save_figure(fig, output_path, dpi=300)

# Optional: Save vector version for editing
svg_path = output_path.with_suffix(".svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
```

---

## Manual Sync Process for Plots

### After Running Exploratory Notebooks

**Step 1: Verify plots on Drive**

```bash
ls ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/figures/
```

**Step 2: Copy to local repo**

```bash
# Copy entire figures directory
cp -r ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/figures/ \
       outputs/phase_2/

# Or copy specific notebook's plots
cp -r ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal/ \
       outputs/phase_2/figures/
```

**Step 3: Review plots**

```bash
# Open in Preview/Image viewer
open outputs/phase_2/figures/exp_01_temporal/*.png
```

**Step 4: Commit to Git**

```bash
git add outputs/phase_2/figures/
git commit -m "feat: Add exploratory analysis plots (exp_01 temporal)"
git push
```

**Step 5: Reference in documentation**

```markdown
![JM Distance Heatmap](../../outputs/phase_2/figures/exp_01_temporal/jm_heatmap_berlin.png)
_Figure 1: JM distance heatmap showing temporal discriminability patterns for Berlin._
```

---

## Quality Checklist

Before committing plots to Git:

- [ ] All plots use `setup_plotting()` (consistent style)
- [ ] File naming follows convention
- [ ] DPI = 300 (publication quality)
- [ ] Titles, labels, legends are clear and informative
- [ ] Colors are colorblind-friendly (Set2 palette)
- [ ] Annotations highlight key findings
- [ ] File sizes reasonable (<5MB per PNG)
- [ ] Plots saved in correct subdirectory
- [ ] Key plots also saved as SVG (optional, for editing)

---

## Integration with Notebooks

### Standard Visualization Section Template

```python
# ============================================================================
# VISUALIZATION
# ============================================================================

from urban_tree_transfer.utils.plotting import setup_plotting, save_figure, PUBLICATION_STYLE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
setup_plotting()
FIGURES_DIR = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot 1: JM Distance Heatmap
fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
# ... plotting code ...
save_figure(fig, FIGURES_DIR / "jm_heatmap_berlin.png")

# Plot 2: Monthly Comparison
fig, ax = plt.subplots()
# ... plotting code ...
save_figure(fig, FIGURES_DIR / "jm_monthly_comparison.png")

# ... more plots ...

print(f"\n✓ All plots saved to: {FIGURES_DIR}")
print(f"✓ Total plots: {len(list(FIGURES_DIR.glob('*.png')))}")
```

---

## Troubleshooting

### Problem: Plots look different from legacy

**Solution:** Ensure `setup_plotting()` is called before any plotting

### Problem: DPI too low

**Solution:** Use `save_figure(fig, path, dpi=300)` instead of `fig.savefig()`

### Problem: Colors inconsistent

**Solution:** Don't manually set colors unless city-specific, use palette

### Problem: File size too large

**Solution:**

- Reduce figure complexity (fewer data points)
- Use vector format (SVG) for complex plots
- Compress PNGs: `optipng outputs/phase_2/figures/**/*.png`

---

## Example: Complete Plotting Workflow

```python
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from urban_tree_transfer.utils.plotting import setup_plotting, save_figure, PUBLICATION_STYLE

# 1. Setup
setup_plotting()
FIGURES_DIR = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 2. Create plot
fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])

# Example: JM distance heatmap
sns.heatmap(
    jm_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax,
    cbar_kws={"label": "JM Distance"}
)

ax.set_title("JM Distance: Feature Discriminability Across Months (Berlin)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Feature Group", fontsize=12)

# 3. Save
save_figure(fig, FIGURES_DIR / "jm_heatmap_berlin.png")

# 4. Optional: Save SVG for editing
fig.savefig(FIGURES_DIR / "jm_heatmap_berlin.svg", format="svg", bbox_inches="tight")

print(f"✓ Saved: jm_heatmap_berlin.png")

# 5. Cleanup
plt.close("all")

# ============================================================================
# MANUAL SYNC REMINDER
# ============================================================================
print("\n" + "="*70)
print("⚠️  MANUAL STEP REQUIRED:")
print("="*70)
print(f"1. Plots saved to Drive: {FIGURES_DIR}")
print(f"2. Copy to local repo: outputs/phase_2/figures/exp_01_temporal/")
print(f"3. Review plots before committing")
print(f"4. Commit to Git:")
print(f"   git add outputs/phase_2/figures/exp_01_temporal/")
print(f"   git commit -m 'feat: Add temporal analysis plots'")
print(f"   git push")
print("="*70)
```

---

**Last Updated:** 2026-01-28
