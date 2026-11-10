"""
Builds the technical report (Word .docx) from the real results in outputs/.
Run from the project folder:   python report/build_report.py
To use the official NED logo: save it as assets/ned_logo.png, then run this again
(or simply right-click the picture on page 1 in Word -> Change Picture).
Needs:  pip install python-docx
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "outputs" / "figures"
M = json.loads((BASE / "outputs" / "metrics.json").read_text())
OUT = BASE / "report" / "Rice_Variety_Classification_Report.docx"

res = {r["Model"]: r for r in M["results"]}
dtb, dtt = res["Decision Tree (Baseline)"], res["Decision Tree (Tuned)"]
knb, knt = res["KNN (Baseline)"], res["KNN (Tuned)"]
dum = res["Majority Class (Dummy)"]
bp = M["best_params"]
q = M["quality"]
cm = M["confusion_matrices"]
fi = M["feature_importance"]
pct = lambda v: f"{v * 100:.2f}%"

# ---------------------------------------------------------------------------
# Methodology diagram (drawn here so the report always matches the pipeline)
# ---------------------------------------------------------------------------
def draw_methodology(path):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=color, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.4, color="#333333"))

    c1, c2, c3, c4 = "#ffe9c7", "#d7ecff", "#dff3d9", "#f3dcf3"
    box(0.2, 4.2, 2.4, 1.1, "UCI Rice Dataset\n3810 grains, 7 features\n(Cammeo / Osmancik)", c1)
    box(3.1, 4.2, 2.4, 1.1, "Data Inspection & EDA\nmissing, duplicates,\noutliers, correlations", c2)
    box(6.0, 4.2, 2.4, 1.1, "Pre-processing\nlabel encoding,\nmedian imputer", c2)
    box(8.7, 4.2, 2.2, 1.1, "Stratified Split\n80% train (3048)\n20% test (762)", c2)
    arrow(2.6, 4.75, 3.1, 4.75); arrow(5.5, 4.75, 6.0, 4.75); arrow(8.4, 4.75, 8.7, 4.75)

    box(0.2, 2.2, 2.9, 1.2, "Baseline models\nMajority class, default\nDecision Tree, default KNN", c3)
    box(3.8, 2.2, 3.2, 1.2, "Improvement: GridSearchCV\n5-fold stratified CV\nDecision Tree and KNN (scaled)", c3)
    box(7.7, 2.2, 3.1, 1.2, "Final tuned models\nDecision Tree, KNN", c3)
    arrow(9.8, 4.2, 9.8, 3.7); ax.plot([9.8, 1.65], [3.7, 3.7], color="#333333", lw=1.4)
    arrow(1.65, 3.7, 1.65, 3.4); arrow(5.4, 3.7, 5.4, 3.4)
    arrow(7.0, 2.8, 7.7, 2.8)
    ax.text(9.9, 3.85, "training data only", fontsize=8, style="italic", ha="center")

    box(0.2, 0.2, 3.2, 1.2, "Evaluation on unseen test set\nAccuracy, Precision, Recall,\nF1, ROC-AUC, Confusion Matrix", c4)
    box(4.1, 0.2, 2.9, 1.2, "Compare baseline vs tuned\nfeature importance,\nerror analysis", c4)
    box(7.7, 0.2, 3.1, 1.2, "Save model (joblib)\nStreamlit interface:\nprediction + result charts", c4)
    arrow(9.25, 2.2, 9.25, 1.4); ax.plot([9.25, 1.8], [1.8, 1.8], color="#333333", lw=1.4)
    arrow(1.8, 1.8, 1.8, 1.4)
    ax.plot([1.8, 1.8], [2.2, 1.8], color="#333333", lw=1.4)
    arrow(3.4, 0.8, 4.1, 0.8); arrow(7.0, 0.8, 7.7, 0.8)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


METH = FIG / "00_methodology_diagram.png"
draw_methodology(METH)

# ---------------------------------------------------------------------------
# Docx helpers
# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.3)
sec.different_first_page_header_footer = True

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.font.size = Pt(12)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25

for name, size, color in [("Heading 1", 17, "1F3864"), ("Heading 2", 14, "2F5496"), ("Heading 3", 12, "2F5496")]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        st.element.rPr.rFonts.set(qn(att), "Times New Roman")
    for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if st.element.rPr.rFonts.get(qn(att)) is not None:
            del st.element.rPr.rFonts.attrib[qn(att)]
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)
    st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

# footer with page number
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run("Rice Variety Classification  |  CT-361 AIES  |  Page ")
run.font.size = Pt(9)
r2 = fp.add_run()
r2.font.size = Pt(9)
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if t:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), t)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = txt
    r2._r.append(el)


def H1(t): doc.add_heading(t, level=1)
def H2(t): doc.add_heading(t, level=2)
def H3(t): doc.add_heading(t, level=3)


def P(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    return p


def B(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(3)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def T(header, rows, widths=None, caption=None, font=10):
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = c.add_run(caption); rr.bold = True; rr.font.size = Pt(10.5)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(255, 255, 255)
        shade(cell, "2F5496")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(font)
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    n_rows = len(t.rows)
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.0
                if len(rows) <= 8 and ri < n_rows - 1:
                    p.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


fig_no = [0]


def FIGURE(path, caption, width=15.5):
    fig_no[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {fig_no[0]}: {caption}")
    r.italic = True; r.font.size = Pt(10.5)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def centered(text, size=12, bold=False, color=None, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.font.size = Pt(size); r.bold = bold
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.space_after = Pt(space_after)
    return p


# ===========================================================================
# PAGE 1 - COVER
# ===========================================================================
logo = BASE / "assets" / "ned_logo.png"
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(30)
p.add_run().add_picture(str(logo), width=Cm(5.2))
centered("NED University of Engineering & Technology", 20, True, "1F3864", 2)
centered("Department of Computer Science & Information Technology", 13, False, None, 40)
centered("AIES", 30, True, "1F3864", 2)
centered("Artificial Intelligence & Expert System (CT-361)", 16, True, None, 2)
centered("Complex Computing Problem (CCP) & Complex Computing Activity (CCA)", 12, False, None, 36)
centered("Project Title", 12, False, "666666", 2)
centered("Rice Variety Classification", 26, True, "2F5496", 2)
centered("Cammeo vs Osmancik using Decision Tree and K-Nearest Neighbours", 12, False, None, 44)
centered("Submitted by", 13, True, None, 8)
T(["Student Name", "Roll Number"],
  [["Muhammad Bilal Uddin", "CT-24036"],
   ["Muhammad Usman", "CT-24031"],
   ["Muhammad Mustafa", "CT-24038"]],
  widths=[9.0, 5.0], font=12)
page_break()

# ===========================================================================
# PAGE 2 - ABSTRACT + CONTENTS
# ===========================================================================
H1("Abstract")
P("Rice is one of the most important food crops in the world, and different varieties differ in price, "
  "cooking quality and market demand. Telling varieties apart by eye is slow and subjective. This project "
  "builds a Python-based system that classifies a rice grain as Cammeo or Osmancik from seven numerical "
  "morphological measurements (area, perimeter, major axis length, minor axis length, eccentricity, convex "
  "area and extent) taken from the UCI Rice (Cammeo and Osmancik) dataset of 3810 grains.")
P(f"The data were inspected and pre-processed, split 80/20 with stratification, and used to train a "
  f"Decision Tree and a K-Nearest Neighbours (KNN) model. A majority-class model, a default Decision Tree "
  f"and a default KNN form the baseline. Both algorithms were then improved with 5-fold cross-validated "
  f"grid-search hyper-parameter tuning. On the unseen test set the tuned Decision Tree reached "
  f"{pct(dtt['Accuracy'])} accuracy (ROC-AUC {dtt['ROC-AUC']:.3f}) and the tuned KNN reached "
  f"{pct(knt['Accuracy'])} accuracy (ROC-AUC {knt['ROC-AUC']:.3f}), compared with {pct(dum['Accuracy'])} for "
  f"the majority-class baseline. A Streamlit interface lets a user enter grain measurements, see predictions "
  f"with confidence from both models, and inspect the evaluation charts.")

H2("Contents")
for i, t in enumerate([
    "Introduction and Problem Statement", "Problem Description", "Dataset Description",
    "Assumptions and Constraints", "Methodology", "Data Pre-processing", "Models Used",
    "Results and Evaluation", "User Interface", "Ethics, Privacy and Fairness",
    "Challenges and Limitations", "Conclusion and Future Work", "References",
    "Appendix: Project Structure and How to Run"], start=1):
    p = doc.add_paragraph(f"{i}.  {t}")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(1)
page_break()

# ===========================================================================
# 1. INTRODUCTION
# ===========================================================================
H1("1. Introduction and Problem Statement")
H2("1.1 Background")
P("Rice grains of different varieties look very similar to the human eye, yet they differ in shape, size, "
  "cooking time, taste and market value. In rice processing and trading, correct identification of the "
  "variety matters: mixing varieties lowers product quality and can cause financial loss and disputes. "
  "Traditional identification relies on trained inspectors, which is slow, subjective and hard to scale.")
P("Computer vision systems can photograph grains and extract numerical shape measurements automatically. "
  "Once these measurements exist, machine learning can learn the difference between varieties from labelled "
  "examples and classify new grains in milliseconds.")
H2("1.2 Problem Statement")
B([("The issue being addressed: ", "manual, subjective and slow identification of rice variety. The project "
    "automates the decision of whether a grain is Cammeo or Osmancik using only its measured shape features."),
   ("Real-world impact: ", "reliable automatic variety identification supports quality control in rice mills, "
    "prevents mixing and fraud in the supply chain, helps fair pricing, and reduces dependence on human experts."),
   ("AI approach applied: ", "supervised predictive modelling (classification). Two classical machine learning "
    "algorithms, Decision Tree and K-Nearest Neighbours, are trained, compared with baselines, tuned and "
    "evaluated, and presented through a simple interface.")])
H2("1.3 Objectives")
B(["Define a clear real-world classification problem in the agriculture / smart-systems domain.",
   "Load, inspect and pre-process the UCI Rice dataset, including checks for noise, missing values and imbalance.",
   "Implement baseline models and improve them with hyper-parameter tuning.",
   "Evaluate the models with suitable performance metrics on unseen data.",
   "Provide a user-friendly interface that makes predictions and results interpretable.",
   "Discuss ethics, privacy, fairness, transparency and accountability."])

# ===========================================================================
# 2. PROBLEM DESCRIPTION
# ===========================================================================
H1("2. Problem Description")
H2("2.1 Domain and Use Case")
P("Domain: agriculture and food-quality monitoring (a smart-system application). Use case: a camera-based "
  "inspection station photographs rice grains; image processing extracts seven shape measurements per grain; "
  "the classifier receives these numbers and returns the variety (Cammeo or Osmancik) with a confidence score "
  "so that grains can be sorted, verified or flagged for human review.")
H2("2.2 Goals of the System")
B(["Accurately predict the variety of a grain from its measurements.",
   "Show that AI can turn raw measurements into automated decisions better than a naive baseline.",
   "Keep the model transparent: a Decision Tree gives readable rules and feature importance.",
   "Present results in an interpretable way through charts and a prediction interface."])
H2("2.3 Inputs and Outputs")
T(["Item", "Description"],
  [["Input", "Seven numeric features of one rice grain: Area, Perimeter, Major_Axis_Length, "
             "Minor_Axis_Length, Eccentricity, Convex_Area, Extent"],
   ["Output", "Predicted class: Cammeo or Osmancik, with the probability of each class"],
   ["Task type", "Binary classification (supervised learning)"],
   ["Learning data", f"{q['rows']} labelled grains ({q['class_counts']['Cammeo']} Cammeo, "
                     f"{q['class_counts']['Osmancik']} Osmancik)"]],
  widths=[3.5, 12.0], caption="Table 1: Inputs and outputs of the system")
H2("2.4 Alternative Solution Strategies Considered")
P("Several strategies can solve this problem. A rule-based expert system could encode expert thresholds, but "
  "such thresholds are hard to obtain manually and the two varieties overlap. Deep learning on raw images "
  "needs far more data and computing power. Because the features are already extracted and the dataset is of "
  "moderate size, classical machine learning is an effective and efficient choice. Decision Tree was selected "
  "because it is interpretable and produces if-then rules close to an expert system, and KNN was selected "
  "because it is a simple, distance-based method with a very different working principle, which makes the "
  "comparison between the two meaningful.")

# ===========================================================================
# 3. DATASET
# ===========================================================================
H1("3. Dataset Description")
H2("3.1 Source and Relevance")
P("The project uses the Rice (Cammeo and Osmancik) dataset from the UCI Machine Learning Repository, donated "
  "in 2019 by Cinar and Koklu of Selcuk University, Turkey. Images of 3810 rice grains of two certified varieties "
  "grown in Turkey were captured, processed, and seven morphological features were extracted for each grain. "
  "The dataset is released under the CC BY 4.0 licence. It is directly relevant because it contains exactly "
  "the measurements an automated inspection system would produce, together with expert-assigned labels. The "
  "dataset has 3810 samples, which is large enough for a reliable train/test split and cross-validation.")
H2("3.2 Features")
T(["Feature", "Type", "Meaning"],
  [["Area", "Integer", "Number of pixels inside the grain boundary"],
   ["Perimeter", "Continuous", "Length of the grain boundary in pixels"],
   ["Major_Axis_Length", "Continuous", "Longest line that can be drawn across the grain"],
   ["Minor_Axis_Length", "Continuous", "Shortest line that can be drawn across the grain"],
   ["Eccentricity", "Continuous", "How round the ellipse with the same moments as the grain is"],
   ["Convex_Area", "Integer", "Pixel count of the smallest convex shell around the grain"],
   ["Extent", "Continuous", "Ratio of the grain region to its bounding box"],
   ["Class (target)", "Binary", "Cammeo or Osmancik"]],
  widths=[4.2, 2.6, 8.7], caption="Table 2: Dataset features")
H2("3.3 Class Distribution")
P(f"The dataset contains {q['class_counts']['Osmancik']} Osmancik grains "
  f"({q['class_counts']['Osmancik'] / q['rows'] * 100:.1f}%) and {q['class_counts']['Cammeo']} Cammeo grains "
  f"({q['class_counts']['Cammeo'] / q['rows'] * 100:.1f}%). The classes are mildly imbalanced, so a model that "
  f"always predicts Osmancik would already be right {pct(dum['Accuracy'])} of the time. This is why the "
  "majority-class model is used as the lowest baseline and why precision, recall, F1 and ROC-AUC are "
  "reported in addition to accuracy.")
FIGURE(FIG / "01_class_distribution.png", "Class distribution of the dataset", width=9)
H2("3.4 Exploratory Data Analysis")
P("The feature histograms show that the two varieties overlap, but Cammeo grains are generally larger and "
  "longer (greater area, perimeter and major axis length), while Osmancik grains are smaller and rounder. "
  "Because of this overlap, perfect classification is not expected.")
FIGURE(FIG / "02_feature_distributions.png", "Distribution of each feature by variety")
P("The correlation matrix shows very strong correlation between Area, Perimeter, Convex Area and Major Axis "
  "Length, since all of them grow with grain size. This redundancy has no effect on a Decision Tree but is one "
  "reason feature scaling matters for the distance-based KNN model.")
FIGURE(FIG / "03_correlation_heatmap.png", "Correlation between features", width=11)
H2("3.5 Data Quality Findings")
o = q["iqr_outliers_per_feature"]
T(["Check", "Result"],
  [["Missing values", str(q["missing_values"])],
   ["Duplicate rows", str(q["duplicate_rows"])],
   ["IQR outliers: Area / Convex_Area", f"{o['Area']} / {o['Convex_Area']}"],
   ["IQR outliers: Minor_Axis_Length / Eccentricity", f"{o['Minor_Axis_Length']} / {o['Eccentricity']}"],
   ["IQR outliers: Perimeter / Major_Axis / Extent", f"{o['Perimeter']} / {o['Major_Axis_Length']} / {o['Extent']}"]],
  widths=[9.5, 6.0], caption="Table 3: Data quality summary")
P("The dataset has no missing values and no duplicates. A small number of statistical outliers exist "
  "(mainly in minor axis length and eccentricity). They were kept because they are genuine measurements of "
  "unusually shaped grains, not recording errors, and removing them would make the system fail on real "
  "unusual grains. Tree-based models are not sensitive to outliers, and KNN is protected by feature scaling.")
FIGURE(FIG / "04_boxplots.png", "Box plots of the features by variety (outlier check)")

# ===========================================================================
# 4. ASSUMPTIONS & CONSTRAINTS
# ===========================================================================
H1("4. Assumptions and Constraints")
P("The following table lists each project requirement and how it is handled in this work.")
T(["Requirement / constraint", "How it is addressed"],
  [["Completed in a group", "Developed by a group of three students (see cover page)."],
   ["Suitable dataset(s)", "One sufficiently large and suitable dataset is used: UCI Rice (Cammeo and "
                            "Osmancik), 3810 samples, as permitted by the activity description."],
   ["Noise, missing values, inconsistencies, imbalance",
    "Data checked for missing values, duplicates and outliers; median imputer included in every pipeline; "
    "class imbalance handled by stratified splitting, class-weight tuning and imbalance-aware metrics."],
   ["Diverse data types", "All inputs are numeric. The pipeline handles integer and continuous values and "
                          "encodes the text label into 0/1."],
   ["External factors", "Lighting, camera and grain placement can change measured values; the system "
                        "assumes measurements come from a consistent imaging setup (see Section 11)."],
   ["Data privacy", "The data contain only physical grain measurements; no personal or sensitive data "
                    "are collected or stored."],
   ["Ethics: fairness, transparency, accountability", "Discussed in Section 10; the Decision Tree is "
                                                      "interpretable and confidence is always shown."],
   ["Baseline model", "Majority-class model, default Decision Tree and default KNN are implemented first."],
   ["Improvement with multiple approaches", "Hyper-parameter tuning with GridSearchCV is applied to both "
                                           "Decision Tree and KNN and compared with the baselines."],
   ["Real-time monitoring depends on network",
    "The model runs locally on the user's computer, so predictions do not need a network connection."]],
  widths=[5.2, 10.3], caption="Table 4: Requirements and how they are addressed")
H2("4.1 Assumptions")
B(["The labels in the dataset are correct (assigned by the dataset authors).",
   "The seven features are measured in the same way for new grains as in the training data.",
   "Only the two varieties in the dataset are present; grains of any other variety are out of scope.",
   "The train and test samples come from the same distribution."])

# ===========================================================================
# 5. METHODOLOGY
# ===========================================================================
H1("5. Methodology")
P("The solution follows a standard supervised machine learning workflow. Figure 5 shows the methodology "
  "diagram of the complete system, from the raw dataset to the final interface.")
FIGURE(METH, "Methodology diagram of the Rice Variety Classifier", width=16)
P("The main stages are:")
B([("Data loading and inspection: ", "read the CSV file, check shape, types, missing values, duplicates, "
    "outliers and class balance."),
   ("Pre-processing: ", "encode the label, split into training and test sets, impute (if needed) and scale "
    "(for KNN)."),
   ("Baseline modelling: ", "train the majority-class model, a default Decision Tree and a default KNN."),
   ("Improvement: ", "tune the hyper-parameters of both algorithms with 5-fold stratified cross-validation, "
    "using the training data only."),
   ("Evaluation: ", "measure all models on the unseen test set with several metrics and plots."),
   ("Deployment: ", "save the trained models and present them through a Streamlit interface.")],
  style="List Number")

# ===========================================================================
# 6. PREPROCESSING
# ===========================================================================
H1("6. Data Pre-processing")
B([("Duplicate removal: ", f"duplicate rows would be dropped; {q['duplicate_rows']} were found."),
   ("Label encoding: ", "Cammeo = 0 and Osmancik = 1."),
   ("Stratified train/test split: ", f"80% training ({M['split']['train']} grains) and 20% test "
    f"({M['split']['test']} grains) with a fixed random seed ({M['split']['seed']}) for reproducibility. "
    "Stratification keeps the class ratio the same in both sets."),
   ("Missing-value handling: ", "a median imputer is the first step of every model pipeline. No values are "
    "missing in this dataset, but the pipeline remains robust if a measurement is missing."),
   ("Feature scaling: ", "StandardScaler (zero mean, unit variance) is applied inside the KNN pipeline because "
    "KNN uses distances and Area (about 12,000) would otherwise dominate Eccentricity (about 0.9). Decision "
    "Trees split on single features and do not need scaling."),
   ("Prevention of data leakage: ", "the imputer and scaler are fitted only on training folds because they are "
    "part of a scikit-learn Pipeline. The test set is never used for tuning.")])

# ===========================================================================
# 7. MODELS
# ===========================================================================
H1("7. Models Used")
H2("7.1 Decision Tree")
P("A Decision Tree repeatedly splits the data using a threshold on one feature, choosing the split that best "
  "separates the classes (Gini impurity or entropy). The result is a tree of if-then rules that a human can "
  "read, which makes the model transparent and close in spirit to a rule-based expert system. Its weakness is "
  "over-fitting: a very deep tree memorises the training data. Limiting depth and leaf size controls this.")
H2("7.2 K-Nearest Neighbours (KNN)")
P("KNN classifies a new grain by looking at the k most similar training grains (smallest distance in scaled "
  "feature space) and taking a majority vote. It makes no assumption about the shape of the decision boundary, "
  "but it is sensitive to feature scale and to the choice of k: a small k is noisy, a large k blurs the boundary.")
H2("7.3 Baseline Models")
B(["Majority-class (Dummy) model: always predicts the most common class. It shows the accuracy that can be "
   "obtained without any learning.",
   "Decision Tree with default settings (no depth limit).",
   "KNN with k = 5 and scaled features."])
H2("7.4 Performance Improvement: Hyper-parameter Tuning")
P(f"GridSearchCV with 5-fold stratified cross-validation (accuracy as the scoring metric) was run on the "
  f"training set. The Decision Tree grid had {M['grid_sizes']['decision_tree']} combinations and the KNN grid "
  f"had {M['grid_sizes']['knn']} combinations.")
T(["Model", "Hyper-parameters searched", "Best values found"],
  [["Decision Tree", "criterion {gini, entropy}; max_depth {3,4,5,6,8,10,None}; min_samples_split "
                     "{2,5,10,20}; min_samples_leaf {1,2,5,10}; class_weight {None, balanced}",
    f"criterion = {bp['decision_tree']['criterion']}; max_depth = {bp['decision_tree']['max_depth']}; "
    f"min_samples_split = {bp['decision_tree']['min_samples_split']}; "
    f"min_samples_leaf = {bp['decision_tree']['min_samples_leaf']}; "
    f"class_weight = {bp['decision_tree']['class_weight']}"],
   ["KNN", "n_neighbors {1,3,...,31}; weights {uniform, distance}; metric {euclidean, manhattan}",
    f"n_neighbors = {bp['knn']['n_neighbors']}; weights = {bp['knn']['weights']}; "
    f"metric = {bp['knn']['metric']}"]],
  widths=[2.8, 7.0, 5.7], caption="Table 5: Hyper-parameter search spaces and best values")
P(f"The tuned Decision Tree is shallow (maximum depth {bp['decision_tree']['max_depth']}) with a minimum of "
  f"{bp['decision_tree']['min_samples_leaf']} grains per leaf, which prevents over-fitting. The tuned KNN uses "
  f"a large neighbourhood (k = {bp['knn']['n_neighbors']}), which smooths out noise in the overlapping region "
  "between the two varieties.")

# ===========================================================================
# 8. RESULTS
# ===========================================================================
H1("8. Results and Evaluation")
H2("8.1 Evaluation Metrics")
B([("Accuracy: ", "share of grains classified correctly."),
   ("Precision, Recall, F1-score: ", "macro-averaged over both classes so that each variety counts equally."),
   ("ROC-AUC: ", "ability to separate the classes across all thresholds (1.0 is perfect, 0.5 is random)."),
   ("Confusion matrix: ", "shows exactly which grains are confused."),
   ("Cross-validated accuracy: ", "mean and standard deviation over 5 folds of the training data, a check on "
    "stability.")])
H2("8.2 Overall Comparison")
rows = []
for name in ["Majority Class (Dummy)", "Decision Tree (Baseline)", "Decision Tree (Tuned)",
             "KNN (Baseline)", "KNN (Tuned)"]:
    r = res[name]
    rows.append([name, f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}", f"{r['Recall']:.4f}",
                 f"{r['F1-Score']:.4f}", f"{r['ROC-AUC']:.4f}",
                 f"{r['CV Accuracy (mean)']:.4f} ± {r['CV Accuracy (std)']:.4f}"])
T(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV accuracy"], rows,
  widths=[4.1, 1.8, 1.9, 1.7, 1.5, 1.9, 2.6], caption="Table 6: Test-set performance of all models", font=9)
FIGURE(FIG / "05_model_comparison.png", "Comparison of all models on the test set")
P(f"All learning models are far better than the majority-class baseline ({pct(dum['Accuracy'])}). Tuning "
  f"improved the Decision Tree clearly: accuracy rose from {pct(dtb['Accuracy'])} to {pct(dtt['Accuracy'])}, "
  f"F1 from {dtb['F1-Score']:.4f} to {dtt['F1-Score']:.4f} and ROC-AUC from {dtb['ROC-AUC']:.4f} to "
  f"{dtt['ROC-AUC']:.4f}. For KNN, the test accuracy stayed at {pct(knt['Accuracy'])}, but ROC-AUC improved from "
  f"{knb['ROC-AUC']:.4f} to {knt['ROC-AUC']:.4f} and the cross-validated accuracy from "
  f"{knb['CV Accuracy (mean)']:.4f} to {knt['CV Accuracy (mean)']:.4f}. The baseline KNN was already strong, "
  "so tuning gave it better-calibrated probabilities rather than more correct labels on this particular test split.")
P(f"The test set holds {M['split']['test']} grains, so one percentage point equals roughly eight grains. The "
  "difference between the tuned Decision Tree and the tuned KNN is therefore small and should not be "
  "over-interpreted; both models perform at a similar level, in line with accuracies of roughly 92 to 94 "
  "percent reported by other studies on this dataset.")
H2("8.3 Confusion Matrices")
FIGURE(FIG / "06_confusion_matrices.png", "Confusion matrices of baseline and tuned models", width=16)
t = cm["Decision Tree (Tuned)"]; k = cm["KNN (Tuned)"]
P(f"For the tuned Decision Tree, {t[0][0]} Cammeo and {t[1][1]} Osmancik grains were classified correctly, "
  f"{t[0][1]} Cammeo grains were wrongly labelled Osmancik and {t[1][0]} Osmancik grains were wrongly labelled "
  f"Cammeo. The tuned KNN made {k[0][1]} and {k[1][0]} errors of the two kinds respectively. Errors occur "
  "mostly for grains whose size lies in the overlapping region of the two varieties.")
H2("8.4 ROC Curves")
FIGURE(FIG / "07_roc_curves.png", "ROC curves of the models", width=11)
P("The curves of the tuned models lie close to the top-left corner, confirming that the two varieties are "
  "well separable from the seven features. The tuned models have visibly higher AUC than the baselines.")
H2("8.5 Feature Importance")
top = list(fi.items())
FIGURE(FIG / "08_feature_importance.png", "Feature importance of the tuned Decision Tree", width=11)
P(f"The tuned Decision Tree relies mostly on {top[0][0].replace('_', ' ')} (importance {top[0][1]:.2f}), followed "
  f"by {top[1][0].replace('_', ' ')} ({top[1][1]:.2f}). Because Area, Perimeter and Convex Area are strongly "
  "correlated with each other, the tree uses one of them and the others receive a low score; a low importance "
  "does not mean that the feature is unrelated to the variety. The result agrees with the exploratory "
  "analysis: grain length is the most informative difference between Cammeo and Osmancik.")
H2("8.6 Effect of Hyper-parameters")
FIGURE(FIG / "09_knn_accuracy_vs_k.png", "KNN cross-validated accuracy for different values of k", width=11)
P("Accuracy rises as k grows from 1, because a single nearest neighbour is sensitive to noisy grains, and "
  "then levels off at moderate values of k.")
FIGURE(FIG / "10_tree_depth_curve.png", "Decision Tree accuracy for different depths", width=11)
P("The training accuracy of the Decision Tree keeps increasing with depth while the cross-validated accuracy "
  "stops improving after a shallow depth. The growing gap is over-fitting, which is why a shallow tuned tree "
  "generalises better than the unrestricted baseline tree.")
H2("8.7 Model Transparency")
FIGURE(FIG / "11_decision_tree_top.png", "First levels of the tuned Decision Tree", width=16)
P("The first levels of the tuned tree can be read as simple rules on grain measurements. This transparency "
  "allows a domain expert to check whether the model's reasoning is sensible.")

# ===========================================================================
# 9. INTERFACE
# ===========================================================================
H1("9. User Interface")
P("A web-based interface was built with Streamlit (file app.py) and runs locally in the browser with the "
  "command: streamlit run app.py. It has four tabs:")
B([("Predict: ", "sliders for the seven grain measurements (limited to the range seen in the dataset), "
    "starting values for a typical Cammeo grain, a typical Osmancik grain or the dataset average, and the "
    "choice of Decision Tree, KNN or both. The predicted variety is shown with a confidence percentage and a "
    "probability bar chart."),
   ("Model Results: ", "the table of baseline versus tuned metrics, best hyper-parameters, the model "
    "comparison chart and the confusion matrices."),
   ("Charts: ", "ROC curves, feature importance, KNN accuracy versus k, Decision Tree depth curve and the "
    "top of the tree."),
   ("Dataset: ", "a preview of the data, class-wise averages, feature distributions and the correlation "
    "heatmap.")])
P("Showing both models side by side with their confidence helps users judge how reliable a prediction is: if "
  "the two models disagree or the confidence is low, the grain can be sent for manual inspection.")

# ===========================================================================
# 10. ETHICS
# ===========================================================================
H1("10. Ethics, Privacy and Fairness")
B([("Data privacy: ", "the dataset contains only physical measurements of rice grains. It has no personal or "
    "sensitive information, so no anonymisation is necessary. If the system were extended with farmer or "
    "supplier data, such data would have to be anonymised."),
   ("Fairness: ", "the classes are imbalanced (about 57% and 43%). To avoid a model that favours the larger "
    "class, a stratified split, class-weight tuning and per-class metrics (macro precision, recall, F1) and "
    "confusion matrices were used. The model is not used to decide anything about people."),
   ("Transparency: ", "the Decision Tree is human-readable, feature importance is reported, and the interface "
    "displays a probability with every prediction instead of a bare label."),
   ("Accountability: ", "all results are reproducible (fixed random seed, saved code and model). Predictions "
    "are decision support; a human inspector remains responsible for final quality decisions, especially for "
    "low-confidence predictions."),
   ("Appropriate use: ", "the model should only be used for the two varieties and imaging conditions it was "
    "trained on, and it should not be presented as perfectly accurate.")])

# ===========================================================================
# 11. CHALLENGES
# ===========================================================================
H1("11. Challenges and Limitations")
B([("Overlapping classes: ", "the two varieties overlap in size, so some grains are misclassified by every "
    "model and accuracy saturates at about 91 to 93 percent."),
   ("Correlated features: ", "Area, Perimeter and Convex Area carry almost the same information, which makes "
    "feature-importance values harder to interpret and makes scaling essential for KNN."),
   ("Over-fitting: ", "the unrestricted Decision Tree memorises training data; this was controlled by limiting "
    "depth and leaf size with cross-validated tuning."),
   ("Class imbalance: ", "mild imbalance can hide errors in accuracy alone; extra metrics were used."),
   ("Small test set: ", "with 762 test grains, differences below about one percentage point are not reliable."),
   ("External factors: ", "measurements depend on camera, lighting, resolution and grain placement. A system "
    "deployed in another factory would need a similar imaging set-up or re-calibration."),
   ("Scope: ", "only two varieties are covered; a grain of any other variety would still be forced into one "
    "of the two classes."),
   ("Network dependence: ", "the model is run locally, so predictions do not depend on network availability; "
    "only a future cloud-based monitoring deployment would.")])

# ===========================================================================
# 12. CONCLUSION
# ===========================================================================
H1("12. Conclusion and Future Work")
P(f"This project built a complete AI solution for classifying rice grains as Cammeo or Osmancik. After "
  f"inspecting and pre-processing the 3810-grain UCI dataset, a Decision Tree and a KNN model were trained, "
  f"compared with baselines and tuned with cross-validation. The tuned Decision Tree achieved "
  f"{pct(dtt['Accuracy'])} accuracy and the tuned KNN {pct(knt['Accuracy'])}, both far above the "
  f"{pct(dum['Accuracy'])} majority-class baseline, with ROC-AUC values near 0.97. Grain length (major axis "
  "length) proved the most informative feature. A Streamlit interface makes the predictions and evaluation "
  "results easy to use and interpret.")
P("Possible future work includes testing further algorithms and ensembles, adding more rice varieties, "
  "collecting images from different cameras to test robustness, and connecting the classifier directly to a "
  "camera-based sorting line.")

H1("References")
for i, ref in enumerate([
    "Cinar, I. and Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. "
    "International Journal of Intelligent Systems and Applications in Engineering, 7(3), 188-194. "
    "doi:10.18201/ijisae.2019355381",
    "Cinar, I. and Koklu, M. Rice (Cammeo and Osmancik) [Dataset]. UCI Machine Learning Repository (2019). "
    "doi:10.24432/C5MW4Z",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning "
    "Research, 12, 2825-2830.",
    "Streamlit documentation: https://docs.streamlit.io",
    "Course material: CT-361 Artificial Intelligence & Expert System, NED University of Engineering and "
    "Technology."], start=1):
    rp = P(f"[{i}]  {ref}", align=WD_ALIGN_PARAGRAPH.LEFT)
    rp.paragraph_format.left_indent = Cm(1.0)
    rp.paragraph_format.first_line_indent = Cm(-1.0)

H1("Appendix: Project Structure and How to Run")
T(["File / folder", "Purpose"],
  [["train_model.py", "Full pipeline: load, pre-process, baselines, tuning, evaluation, saves model and figures"],
   ["app.py", "Streamlit interface"],
   ["data/Rice_Cammeo_Osmancik.csv", "UCI Rice dataset (3810 rows)"],
   ["models/rice_models.joblib", "Saved tuned Decision Tree and KNN pipelines"],
   ["outputs/", "metrics.json, results_table.csv and all figures"],
   ["report/", "This report and its build script"],
   ["requirements.txt", "Python packages needed"],
   ["README_Roman_Urdu.txt", "Roman Urdu explanation of the whole project"]],
  widths=[5.5, 10.0], caption="Table 7: Project files")
P("Steps: (1) pip install -r requirements.txt   (2) python train_model.py   (3) streamlit run app.py", bold=True,
  align=WD_ALIGN_PARAGRAPH.LEFT)

doc.save(OUT)
print("Saved", OUT)
