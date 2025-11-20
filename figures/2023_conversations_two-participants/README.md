# Temporal Clustering Analysis Figures

This directory contains two related analyses of the same dataset (6,964 tutor-student conversations):

---

## 📁 Root Directory: 5D Conversation-Level Analysis

**Files:** `figure1-4.png` in this directory

**Features:** 5 temporal metrics computed at the **conversation level** (aggregating both tutor and student):
- Burst Coefficient (BC)
- Cluster Density (CD)
- Response Acceleration (RA)
- Memory Coefficient (MC)
- Timing Consistency (TC)

**Result:** Identified outlier conversations using DBSCAN (989 outliers, 14.2%)

---

## 📁 [role_15d/](role_15d/) - 15D Role-Decomposed Analysis

**Files:** `figure1-4.png` in `role_15d/` subdirectory

**Features:** Same 5 temporal metrics but **decomposed by role** into 15 dimensions:
- 5 conversation-level features
- 5 tutor-only features
- 5 student-only features

**Key Finding:** Student temporal patterns drive clustering (η²=0.33-0.41), while tutor patterns have negligible effects (η²<0.04). This role decomposition reveals that **students**, not tutors, differentiate conversation types.

**Result:** Two clusters with hierarchical clustering (88.4% vs 11.6%)

---

## Which Analysis Should I Look At?

- **For conversation-level patterns**: Use root directory figures
- **For role-specific patterns** (who drives temporal behavior): Use `role_15d/` figures
- **For the main research finding** (student vs tutor effects): Use `role_15d/` figures

See [role_15d/README.md](role_15d/README.md) for detailed figure descriptions.

---

**Dataset:** 2023 tutor-student conversations (n=6,964)
**Generated:** 2025-11-20
