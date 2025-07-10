`markdown
#  Colourimetric Qualitative Detection using SVM (RBF Kernel)

This project performs qualitative detection of biochemical samples using colorimetric analysis. It leverages computer vision (OpenCV) to extract HSV values from well-plate images and applies a Support Vector Machine (SVM) with RBF kernel to classify samples as:
-  Positive (`+1`)
-  Negative (`-1`)
-  Uninterpretable (`*` - skipped during training)

---

##  Objective

Detect whether u
`

---

##  Tech Stack

| Layer              | Tech                                |
| ------------------ | ----------------------------------- |
|  ML Model        | SVM with RBF (scikit-learn)         |
|  Logic           | HSV Feature Extraction (OpenCV)     |
|  Visualization   | Seaborn, Matplotlib                 |
|  Data Handling   | Pandas, NumPy                       |
|  GUI (Optional) | Tkinter (Live Image Classification) |

---

##  Directory Structure
---

##  Data Processing

1. **Crop Wells** into a 5×4 or 6×5 grid.
2. **Compute Mean HSV** for each well using OpenCV.
3. Label wells manually using:

   * `+1`: Yellow (positive)
   * `-1`: Pink (negative)
   * `*`: Orange (excluded)
4. Store as CSV (`Row,Col,H,S,V,Label`).

---

##  ML Logic

* **Feature Vector**: `[H, S, V]` per well
* **Classifier**: `SVC(kernel='rbf', gamma='scale')`
* **Train/Test Split**: 80-20
* **Evaluation**: Accuracy, Confusion Matrix, Heatmaps

---

##  Heatmap Example
---

##  One Pager (Project Summary)

* **Title**: *Colourimetric Qualitative Detection using SVM*
* **Objective**: Classify test wells as positive or negative based on colorimetric image detection
* **Input**: Images of micro-well plates
* **Processing**:

  * Grid detection & cropping
  * HSV color extraction
  * Labeling from image map
* **Model**: Support Vector Machine (RBF Kernel)
* **Output**: CSV results, classification grid, heatmap
* **Tech Used**: OpenCV, Scikit-Learn, Pandas, Matplotlib, Seaborn
* **Future Work**: Add semi-supervised learning, deploy via Flask or Streamlit, auto-threshold HSV ranges

---

##  Contributions

Made with  by [j
Jhinuk Roy](https://github.com/jhinukroy)
