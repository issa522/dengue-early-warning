"""Build the written report (PDF) from the notebook's saved results and figures.

Run after executing the notebook:   python reports/build_report.py
Edit the text below and re-run to regenerate reports/Dengue_Early_Warning_Report.pdf.
"""
import ast
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).parent
FIG = HERE / "figures"
OUT = HERE / "Dengue_Early_Warning_Report.pdf"

STUDENT_NAME = "Isuru ____________"   # <-- fill in
STUDENT_ID = "____________"           # <-- fill in
MODULE = "IT5514 – Applied Machine Learning (MSc in Artificial Intelligence)"
GITHUB = "https://github.com/issa522/dengue-early-warning"

# ------------------------------------------------------------------ fonts & styles
FONT, BOLD, ITALIC = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
win_fonts = Path("C:/Windows/Fonts")
if (win_fonts / "arial.ttf").exists():  # Arial has ≈ ≥ → glyphs that Helvetica lacks
    pdfmetrics.registerFont(TTFont("Arial", str(win_fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(win_fonts / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(win_fonts / "ariali.ttf")))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    FONT, BOLD, ITALIC = "Arial", "Arial-Bold", "Arial-Italic"

ACCENT = colors.HexColor("#9b2c2c")
LIGHT = colors.HexColor("#f6eeee")
GRID = colors.HexColor("#c9c9c9")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName=FONT, fontSize=10, leading=14.2,
                      alignment=TA_JUSTIFY, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=BOLD, fontSize=15, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=7, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=11.5, spaceBefore=9, spaceAfter=4,
                    keepWithNext=1)
caption = ParagraphStyle("caption", parent=body, fontName=ITALIC, fontSize=8.8, leading=11.5,
                         alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11, alignment=0, spaceAfter=0)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName=BOLD)
title_s = ParagraphStyle("title", parent=body, fontName=BOLD, fontSize=21, leading=26, alignment=TA_CENTER,
                         textColor=ACCENT, spaceAfter=8)
sub_s = ParagraphStyle("sub", parent=body, fontSize=11.5, leading=16, alignment=TA_CENTER, spaceAfter=4)

WIDTH = A4[0] - 4.4 * cm
story = []
fig_no = [0]
tab_no = [0]


def P(text, style=body):
    story.append(Paragraph(text, style))


def H1(text):
    story.append(Paragraph(text, h1))


def H2(text):
    story.append(Paragraph(text, h2))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 4))


def table(rows, widths, caption_text, header=True, highlight_rows=(), heading=None):
    tab_no[0] += 1
    data = [[Paragraph(str(c), cell_b if (header and r == 0) else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4d6")))
    t.setStyle(TableStyle(style))
    head = [Paragraph(heading, h2)] if heading else []
    story.append(KeepTogether(head + [Paragraph(f"<b>Table {tab_no[0]}.</b> {caption_text}", caption), t,
                                      Spacer(1, 10)]))


def figure(name, caption_text, width=1.0):
    fig_no[0] += 1
    path = FIG / f"{name}.png"
    img = Image(str(path))
    scale = WIDTH * width / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    story.append(KeepTogether([img, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption_text}", caption)]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2.2 * cm, 1.2 * cm, "Dengue Outbreak Early Warning – Colombo, Sri Lanka")
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------ results from the notebook
cv = pd.read_csv(HERE / "cv_results.csv").set_index("model")
test = pd.read_csv(HERE / "test_results.csv", index_col=0)
et = test.loc["Extra Trees"]
et_params = ast.literal_eval(cv.loc["Extra Trees", "best params"])

# ================================================================== TITLE
story += [Spacer(1, 3.2 * cm)]
P("Dengue Outbreak Early Warning for Colombo District, Sri Lanka", title_s)
P("Predicting high-risk dengue months four weeks ahead with Extremely Randomized Trees", sub_s)
story.append(Spacer(1, 1.2 * cm))
P(MODULE, sub_s)
P("Machine Learning Assignment – Written Report", sub_s)
story.append(Spacer(1, 1.2 * cm))
P(f"<b>Student:</b> {STUDENT_NAME}", sub_s)
P(f"<b>Student ID:</b> {STUDENT_ID}", sub_s)
P(f"<b>Source code:</b> {GITHUB}", sub_s)
story.append(Spacer(1, 1.6 * cm))

abstract = (
    "Dengue is a major public-health burden in Sri Lanka, and Colombo district reports the largest number of "
    "cases in the country. Vector-control actions take weeks to organise, so an advance warning is valuable. "
    "This project builds a binary early-warning classifier that uses the last eight weeks of notified cases and "
    "the time of year to predict whether the <b>average weekly cases over the next four weeks</b> will exceed a "
    "high-risk threshold. The data are real weekly surveillance counts from the Epidemiology Unit of the "
    "Ministry of Health (2009–2013) with weekly weather. Several data-quality problems were found and corrected. "
    "The proposed algorithm, <b>Extra Trees</b> (Extremely Randomized Trees), which was not covered in the "
    "lectures, was compared with ten taught algorithms and a naive rule under time-aware validation. On the "
    f"unseen 2013 test year it achieved ROC-AUC {et['ROC-AUC']:.2f}, recall {et['Recall']:.2f} (13 of 15 "
    f"high-risk weeks detected) and balanced accuracy {et['Balanced acc.']:.2f}, against a recall of 0.40 for "
    "the naive rule. SHAP, partial dependence plots and LIME show that season, case momentum and current case "
    "levels drive predictions, in line with dengue epidemiology. A Streamlit application lets users enter recent "
    "case counts and view the prediction with its explanation."
)
story.append(Table([[Paragraph("<b>Abstract.</b> " + abstract, body)]], colWidths=[WIDTH],
                   style=[("BOX", (0, 0), (-1, -1), 0.6, ACCENT), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                          ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                          ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
story.append(PageBreak())

# ================================================================== 1. PROBLEM & DATA
H1("1. Problem Definition and Dataset")
H2("1.1 Problem and relevance")
P("Dengue fever, transmitted by <i>Aedes aegypti</i> and <i>Aedes albopictus</i> mosquitoes, is one of Sri "
  "Lanka's most serious public-health problems. Colombo district, the most densely populated and urbanised "
  "district, consistently reports the highest number of notified cases. The main prevention tools – fogging, "
  "breeding-site clean-up campaigns, school awareness programmes and preparing hospital capacity – take one to "
  "several weeks to organise. In practice they are often triggered only <i>after</i> cases have risen, when "
  "transmission is already widespread.")
P("The goal of this project is to give the Medical Officer of Health (MOH) and the Regional Epidemiologist "
  "an <b>advance warning</b>. The machine-learning task is defined as binary classification:")
P("<i>Using only information available at the end of the current week, predict whether the average weekly "
  "number of notified dengue cases over the next four weeks will exceed the high-risk threshold (the 75th "
  "percentile of the training period, 146 cases per week).</i>")
P("A four-week average was chosen instead of next week's exact count for two reasons. First, weekly "
  "notifications are very noisy because of reporting delays and batching (for example 41 cases one week and 231 "
  "the next in 2013). Initial regression experiments on single weeks could not beat a simple persistence "
  "forecast. Second, a four-week window matches the planning horizon of a vector-control campaign.")

H2("1.2 Data source and collection")
P("Weekly notified dengue cases for Colombo district come from the <b>Weekly Epidemiological Reports of the "
  "Epidemiology Unit, Ministry of Health, Sri Lanka</b>. They were compiled together with weekly Colombo weather "
  "observations by Talagala in the open-source dataset <i>colmozzie</i> (\"Dengue Cases and Climate Variables in "
  "Colombo, Sri Lanka\", published on CRAN). The dataset file (<i>colmozzie.rda</i>) was downloaded from the "
  "package's public GitHub repository, read in Python with <i>pyreadr</i> and saved as CSV. It covers 2009 "
  "week 1 to 2014 week 21: <b>279 weekly records and 12 columns</b> (Table 1).")
table([["Column", "Meaning", "Unit / type"],
       ["Cases", "Notified dengue cases in Colombo district during the week", "count"],
       ["Year, Week", "ISO year and week number", "integer"],
       ["TEM, TMAX, Tm", "Average, maximum and minimum temperature", "°C"],
       ["SLP", "Atmospheric pressure at sea level", "hPa"],
       ["H", "Average relative humidity", "%"],
       ["PP", "Total rainfall", "mm"],
       ["VV", "Average visibility", "km"],
       ["V, VM", "Average and maximum sustained wind speed", "km/h"]],
      [0.2, 0.6, 0.2], "Raw dataset columns.")

H2("1.3 Data-quality problems found during exploration")
P("Every value was checked for plausibility, not only for missingness. Four problems were found:")
bullets([
    "<b>Excel error:</b> sea-level pressure (SLP) contained the text <i>#DIV/0!</i> in 2009 week 50, so the "
    "column had been stored as text.",
    "<b>False zeros:</b> five weeks in 2012 report 0 cases. 2012 was a severe dengue year and the neighbouring "
    "weeks have 60–200 cases, so these are missing reports, not real zeros. Using them as zeros would teach the "
    "model something false.",
    "<b>Missing weeks:</b> weeks 37 and 38 of 2013 are absent from the series.",
    "<b>Synthetic weather:</b> the 2014 weather values have a weekly standard deviation about 25 times smaller "
    "than other years (temperature 0.035 °C against about 0.85 °C) and long decimal tails. They were evidently "
    "imputed by the dataset author, not observed. All data from 2013 week 52 onwards were therefore excluded.",
])
figure("01_weekly_cases", "Weekly notified dengue cases in Colombo (raw data). Two seasonal peaks per year, a "
       "large outbreak in mid-2011 and strong week-to-week noise are visible.")
figure("02_seasonality", "Left: cases by month, showing the main peak in June–July after the south-west monsoon. "
       "Right: average weekly cases per year.")

table([["Step", "What was done", "Why"],
       ["Type correction", "SLP converted to numeric; #DIV/0! becomes missing", "Excel error in source data"],
       ["Missing values", "False zeros and missing weeks set to missing, then linear interpolation in time; "
        "a median imputer is also part of the model pipeline", "Keeps the weekly series continuous; the imputer "
        "protects the app against blank inputs"],
       ["Unreliable data", "2013-W52 and all 2014 records excluded", "Weather was imputed, not observed"],
       ["Transformation", "log(1 + x) of case counts and rainfall; sine/cosine encoding of week number",
        "Reduces strong right skew (outbreaks, heavy-rain weeks); week 52 is next to week 1"],
       ["Feature engineering", "Current-week cases; 2-, 4- and 8-week case means; 4-week trend (last 4 vs "
        "previous 4 weeks); rainfall, humidity, min and mean temperature averaged over lags 0–3 and 4–7 weeks",
        "Uses only past information; the lagged weather reflects the 5–10 week transmission delay"],
       ["Outlier handling", "Winsorisation of every feature to its 1st–99th training percentile",
        "Limits the influence of extreme weeks without deleting real outbreaks"],
       ["Scaling", "StandardScaler", "Required by logistic regression and k-NN; harmless for trees"]],
      [0.17, 0.48, 0.35], "Pre-processing steps. All model-side steps are inside one scikit-learn pipeline, "
      "fitted on training data only.", heading="1.4 Pre-processing and feature engineering")
P("After cleaning, the modelling period is 2009–2013 (260 weeks). Building the 8-week history and the 4-week-"
  "ahead target leaves <b>249 labelled weeks</b>. No categorical encoding was needed because all variables are "
  "numeric.")
figure("03_distributions", "Skewness of cases and rainfall, and the effect of the log transformation.")

H2("1.5 Ethical data use")
P("The dataset contains only <b>aggregated weekly counts for the whole district</b> and public weather "
  "observations. There are no names, addresses, ages or any other personal or sensitive information, so "
  "individual consent is not required. The data are published openly for research use, and the source is "
  "credited in this report and in the code repository.")

# ================================================================== 2. ALGORITHM
H1("2. Selection of a New Machine Learning Algorithm")
H2("2.1 Algorithms covered in lectures")
P("The lectures covered linear, ridge and lasso regression, logistic regression, SVM, decision trees, bagging, "
  "Random Forest, k-NN, Gradient Boosting, voting ensembles, XGBoost, CatBoost, AdaBoost, LightGBM and stacking "
  "(plus clustering methods and a brief mention of Explainable Boosting Machines). The proposed algorithm had to "
  "be outside this list and not a deep-learning model.")
H2("2.2 Proposed algorithm: Extra Trees (Extremely Randomized Trees)")
P("Extra Trees was introduced by Geurts, Ernst and Wehenkel (2006). It builds an ensemble of decision trees "
  "and averages their predicted class probabilities. Its distinctive idea is <b>additional randomisation when "
  "choosing splits</b>:")
bullets([
    "At each node, a random subset of features is considered (parameter <i>max_features</i>).",
    "For <b>each</b> of these features, <b>one split threshold is drawn at random</b> between the feature's "
    "minimum and maximum in that node. A decision tree or Random Forest instead searches all possible thresholds "
    "for the best one.",
    "The best of these random candidate splits (by Gini impurity) is kept.",
    "By default every tree is trained on the <b>whole training set</b> rather than a bootstrap sample.",
])
P("Random thresholds make each individual tree slightly weaker (higher bias), but the trees become much less "
  "correlated with each other. Averaging many very different trees therefore removes more variance. On small, "
  "noisy datasets such as 198 training weeks of dengue notifications, this trade-off is usually favourable. "
  "Because no threshold search is needed, training is also faster than Random Forest.")
H2("2.3 Why this algorithm was selected")
bullets([
    "<b>Small, noisy data:</b> the main risk with about 200 training rows is overfitting (high variance). "
    "Extra Trees is specifically designed to reduce variance.",
    "<b>Non-linear relationships and interactions:</b> outbreak risk depends on combinations, such as a rising "
    "trend <i>during</i> the monsoon season, which linear models cannot capture.",
    "<b>Explainability:</b> as a tree ensemble it supports exact TreeSHAP explanations and partial dependence, "
    "which are essential in a public-health setting.",
    "<b>Not covered in lectures</b>, and not a deep-learning model.",
])
H2("2.4 How it differs from the standard models")
table([["Compared with", "Key difference", "Consequence for this problem"],
       ["Decision tree", "Hundreds of randomised trees are averaged instead of one optimised tree",
        "A single tree overfitted (lowest CV AUC, 0.76)"],
       ["Logistic regression", "Non-linear; captures interactions; no assumption of a linear log-odds",
        "Dengue risk is not a straight-line function of cases"],
       ["k-NN", "Learns which features matter; not based on distances", "k-NN weighs all features equally and "
        "depends on scaling"],
       ["Random Forest / bagging", "Random split thresholds; no bootstrap by default; less correlated trees",
        "Lower variance; faster training"],
       ["Boosting (Gradient Boosting, XGBoost, CatBoost, LightGBM, AdaBoost)", "Trees are built independently "
        "and averaged, not added sequentially to correct previous errors", "Boosting chases remaining errors, "
        "which in noisy weekly counts are partly noise"]],
      [0.22, 0.43, 0.35], "How Extra Trees differs from the standard models.")

# ================================================================== 3. TRAINING & EVALUATION
H1("3. Model Training and Evaluation")
H2("3.1 Train / validation / test split")
P("Neighbouring weeks are almost identical, so a random split would leak information from the future into "
  "training. A <b>chronological split</b> was used instead:")
bullets([
    "<b>Training:</b> 2009–2012 (198 weeks, 25 % high-risk).",
    "<b>Test (hold-out):</b> 2013 (47 weeks, 15 high-risk = 32 %). It was used only once, at the end.",
    "<b>Purging:</b> the target of the last four training weeks of 2012 uses January 2013 cases. These four rows "
    "were removed so that no test-period information is used in training.",
    "<b>Validation:</b> walk-forward (expanding-window) cross-validation inside the training period. Fold 1 "
    "trains on 2009–2010 and validates on 2011 (an outbreak year, 40 % high-risk). Fold 2 trains on 2009–2011 "
    "and validates on 2012 (25 % high-risk). Each fold also purges four weeks.",
    "The high-risk threshold (146 cases/week) was computed from the training targets only.",
])
H2("3.2 Feature-set selection")
P("Weather is the obvious biological driver of dengue, so its value was tested on the validation folds before "
  "tuning (Table 4). Adding eight lagged weather features slightly <b>reduced</b> validation AUC, and weather "
  "alone was much weaker. Recent case counts already reflect past weather, because rainfall affects cases 5–10 "
  "weeks later (Figure 4). With about 200 training rows, extra weakly informative features add noise. The final "
  "model therefore uses 7 features: five case-history features and the sine/cosine of the week.")
table([["Feature set", "Mean CV AUC (Random Forest and XGBoost probes)"],
       ["Cases + season (7 features)", "0.876"],
       ["Cases + season + weather (15 features)", "0.868"],
       ["Season + weather only (10 features)", "0.639"]],
      [0.55, 0.45], "Feature-set comparison on the walk-forward validation folds.", highlight_rows=(1,))
figure("05_lagged_correlation", "Correlation between weather k weeks earlier and dengue cases now. Effects "
       "appear after 5–10 weeks and remain weak (|r| &lt; 0.32).", width=0.85)

H2("3.3 Compared models and hyper-parameter tuning")
P("Extra Trees was compared with a naive rule (\"alert if the current 4-week average is already above the "
  "threshold\") and ten taught algorithms: logistic regression, k-NN, decision tree, Random Forest, AdaBoost, "
  "Gradient Boosting, XGBoost, LightGBM, CatBoost and a stacking ensemble (Random Forest + XGBoost + CatBoost "
  "with a logistic-regression meta-learner). No model used default hyper-parameters: each was tuned with "
  "<b>RandomizedSearchCV</b> (25–40 sampled configurations) over the walk-forward folds, maximising ROC-AUC. The "
  "search spaces favour shallow trees, small learning rates and regularisation because of the small sample size.")
P("A classifier outputs a probability, but an officer needs a yes/no alert. For every model, the <b>alert "
  "cut-off</b> was chosen as the probability that maximises F1 on its validation predictions, so all models were "
  "treated the same way and the test set was never used for this choice.")
table([["Hyper-parameter", "Search range", "Selected", "Justification"],
       ["n_estimators", "200–800", str(et_params["n_estimators"]), "More trees give a more stable average; "
        "no overfitting risk"],
       ["max_depth", "3, 4, 5, 6, 8, None", str(et_params["max_depth"]), "Shallow trees cannot memorise noise in "
        "about 200 weeks"],
       ["min_samples_leaf", "1–15", str(et_params["min_samples_leaf"]), "Each leaf must contain at least 12 weeks "
        "(about 3 months)"],
       ["max_features", "sqrt, 0.3, 0.5, 0.8, 1.0", str(et_params["max_features"]), "Only 30 % of features (2 of "
        "7) per split: more diverse trees, lower variance"],
       ["class_weight", "None, balanced", str(et_params["class_weight"]), "Re-weighting did not help on "
        "validation; the imbalance is handled by the cut-off"],
       ["Alert cut-off", "0.10–0.80", f"{cv.loc['Extra Trees', 'cut-off']:.2f}", "Maximises validation F1; lower "
        "than 0.5 because averaged shallow trees give cautious probabilities and missed outbreaks are costly"]],
      [0.19, 0.2, 0.11, 0.5], "Tuned hyper-parameters of the proposed Extra Trees model.")

H2("3.4 Performance metrics")
bullets([
    "<b>ROC-AUC:</b> how well the model ranks high-risk weeks above normal weeks, independent of the cut-off. "
    "Used for tuning.",
    "<b>PR-AUC:</b> similar, but focused on the minority (high-risk) class.",
    "<b>Recall:</b> share of high-risk weeks that were detected. This is the most important metric for an "
    "early-warning system, since a missed outbreak means no preparation.",
    "<b>Precision:</b> share of alerts that were correct (it measures the false-alarm burden).",
    "<b>F1</b> and <b>balanced accuracy:</b> single-number summaries robust to class imbalance. Plain accuracy is "
    "not reported because it rewards always predicting \"normal\".",
])

order =["Extra Trees", "Random Forest", "XGBoost", "CatBoost", "Gradient Boosting", "LightGBM", "AdaBoost",
         "Stacking", "KNN", "Logistic Regression", "Decision Tree"]
rows = [["Model", "Taught", "CV AUC (± std)", "CV F1", "Test AUC", "Test F1", "Recall", "Precision",
         "Bal. acc."]]
for m in order:
    t, c = test.loc[m], cv.loc[m]
    rows.append([("<b>Extra Trees (proposed)</b>" if m == "Extra Trees" else m), "No" if m == "Extra Trees" else
                 "Yes", f"{c['CV AUC']:.3f} ± {c['CV AUC std']:.3f}", f"{c['CV F1']:.2f}", f"{t['ROC-AUC']:.2f}",
                 f"{t['F1']:.2f}", f"{t['Recall']:.2f}", f"{t['Precision']:.2f}", f"{t['Balanced acc.']:.3f}"])
nv = test.loc["Naive rule (current level)"]
rows.append(["Naive rule", "–", "–", "–", f"{nv['ROC-AUC']:.2f}", f"{nv['F1']:.2f}", f"{nv['Recall']:.2f}",
             f"{nv['Precision']:.2f}", f"{nv['Balanced acc.']:.3f}"])
table(rows, [0.2, 0.09, 0.16, 0.07, 0.09, 0.08, 0.08, 0.11, 0.1],
      "Validation (walk-forward CV, 2011–2012) and test (2013) results. Each model uses its own cut-off "
      "chosen on validation.", highlight_rows=(1,), heading="3.5 Results")
P("<b>What the results indicate:</b>")
bullets([
    "Every machine-learning model clearly beats the naive rule (recall 0.40), so learning from case dynamics and "
    "season adds real value over reacting when cases are already high.",
    f"Extra Trees is <b>competitive on validation</b>: its CV AUC ({cv.loc['Extra Trees', 'CV AUC']:.3f}) is "
    "within 0.01 of the best taught algorithms, well inside its fold-to-fold spread, and it has the best CV F1 of "
    "all ensembles.",
    f"On the unseen 2013 year it achieved the <b>joint-highest recall ({et['Recall']:.2f})</b>, detecting 13 of "
    f"15 high-risk weeks, and the <b>highest balanced accuracy ({et['Balanced acc.']:.3f})</b> of all models, "
    f"narrowly ahead of Random Forest ({test.loc['Random Forest', 'Balanced acc.']:.3f}). "
    "Its ROC-AUC (0.89) equals Random Forest, XGBoost and CatBoost.",
    "Compared with Random Forest, Extra Trees raised two more false alarms (6 against 4) but caught one more "
    "outbreak week. For early warning, a missed outbreak costs more than an unnecessary clean-up campaign.",
    "k-NN has the highest test AUC (0.96), but at its validation-chosen cut-off it detected only 8 of 15 "
    "high-risk weeks, and it cannot explain which features matter.",
    "The test year has only 47 weeks, so one week more or less changes recall by about 7 points. Differences of "
    "a few AUC points between ensembles are within this uncertainty. Scores near 100 % would have indicated "
    "leakage. These realistic scores come from a strictly time-ordered evaluation.",
])
figure("06_roc_confusion", "Left: ROC curves on the 2013 test set. Right: confusion matrix of Extra Trees at "
       "its cut-off (13 true alerts, 6 false alarms, 2 missed high-risk weeks, 26 correct normal weeks).")
figure("07_test_timeline", "Extra Trees risk probability (blue) against the actual next-4-week average cases "
       "(red) during 2013. The signal rose with the mid-year outbreak and fell afterwards. It reacted late to the "
       "November rise, which started from a low base.")

# ================================================================== 4. EXPLAINABILITY
H1("4. Explainability and Interpretation")
P("Three complementary explainability methods were applied to the final Extra Trees model: <b>SHAP</b> "
  "(global and local), <b>Partial Dependence Plots</b> (global shape of each effect) and <b>LIME</b> (local "
  "check with a different method).")
H2("4.1 Which features are most influential (SHAP)")
P("SHAP splits every prediction into additive contributions from each feature, based on Shapley values from "
  "cooperative game theory. TreeSHAP computes them exactly for tree ensembles. The most influential feature is "
  "<b>season</b> (week_cos), followed by the <b>4-week trend</b>, <b>current-week cases</b> and the <b>2-week "
  "average</b>. The 8-week average contributes least.")
figure("09_shap_bar", "Global importance: mean absolute SHAP value per feature on the 2013 test weeks.", width=0.72)
figure("08_shap_beeswarm", "SHAP summary. Each dot is a test week, and colour shows the feature value. Low "
       "week_cos (blue, the middle of the year) and high trend or case values (red) push risk up.", width=0.8)
H2("4.2 Shape of the learned relationships (PDP)")
figure("14_pdp", "Partial dependence (red: average effect; blue: individual weeks). Risk falls as week_cos "
       "increases (away from mid-year) and rises smoothly with trend, current cases and the 2-week average.")
P("All curves are smooth and monotonic, with no implausible jumps. Risk increases steadily with case levels, "
  "and the trend effect levels off above about +0.6 (about 80 % growth over the previous month).")
H2("4.3 Individual predictions (SHAP waterfall and LIME)")
figure("11_shap_waterfall_high", "Why the week of 15 July 2013 was flagged high-risk: mid-year season, a "
       "strong rising trend and high current cases.", width=0.8)
figure("12_shap_waterfall_low", "The lowest-risk prediction of 2013 (14 October). Every case feature pushed the "
       "risk down, but a late-year outbreak followed: an instructive missed onset.", width=0.8)
P("LIME fits a simple local linear model to perturbed copies of one week. For the high-risk week its strongest "
  "conditions were mid-year season, more than about 147 cases this week, a high 2-week average and a positive "
  "trend. These are the same drivers SHAP identified, so the two methods agree.")
figure("13_lime_high", "LIME explanation for the high-risk week of 15 July 2013 (case features on log scale: "
       "5.0 ≈ 147 cases).", width=0.75)
H2("4.4 What the model has learned and alignment with domain knowledge")
table([["Model behaviour", "Domain knowledge", "Agrees?"],
       ["Highest risk around weeks 20–32 (mid-May to early August)", "Colombo's main dengue peak follows the "
        "south-west monsoon (May–September): rain creates breeding sites and cases rise 4–8 weeks later", "Yes"],
       ["A rising 4-week trend increases risk", "Epidemics grow exponentially at the start; a rising trend means "
        "transmission is accelerating", "Yes"],
       ["Higher current and 2-week case levels increase risk", "More infected people infect more mosquitoes "
        "(person–mosquito–person cycle)", "Yes"],
       ["The 8-week average matters little", "The transmission cycle is about 2–3 weeks; old cases say little "
        "about current transmission", "Yes"],
       ["Weather adds no extra predictive power", "Weather acts with a 5–10 week delay, so its effect is already "
        "visible in recent case counts", "Consistent"],
       ["Misses onsets from a low base (October 2013)", "The north-east monsoon peak is less regular; serotype "
        "changes can start outbreaks without warning in case counts", "Known limitation"]],
      [0.36, 0.5, 0.14], "Model behaviour compared with dengue epidemiology.")
P("The model has learned an epidemiologically sensible rule: <i>raise the alarm when cases are rising during "
  "the monsoon-driven season</i>. Agreement between SHAP, PDP and LIME increases confidence that the explanations "
  "reflect the model's real behaviour.")

# ================================================================== 5. CRITICAL DISCUSSION
H1("5. Critical Discussion")
H2("5.1 Limitations of the model")
bullets([
    "<b>Small data:</b> 249 labelled weeks from one district. Two validation folds and one test year give "
    "metrics with wide uncertainty.",
    "<b>Reactive more than predictive at onset:</b> the model detects outbreaks once momentum begins, but it "
    "cannot anticipate a sudden rise from a quiet period (October 2013).",
    "<b>Ageing data (2009–2013):</b> dengue in Sri Lanka has changed since then, for example the record 2017 "
    "epidemic linked to a shift in the dominant serotype, and continued urbanisation. The model must be retrained "
    "on recent data before operational use.",
    "<b>Fixed threshold:</b> \"high risk\" (146 cases/week) is relative to 2009–2012. Population growth or "
    "improved reporting would change it.",
])
H2("5.2 Data-quality issues")
bullets([
    "Corrected: an Excel error, five false zero-case weeks, two missing weeks and a year of synthetic weather. "
    "Interpolated weeks are estimates, not observations.",
    "<b>Notified cases are not all infections.</b> Many mild or asymptomatic infections are never reported, and "
    "notification depends on hospital workload and holidays. The model predicts notifications, not true "
    "incidence.",
    "Weather comes from a single station, which may not represent conditions across the district.",
])
H2("5.3 Risks of bias or unfairness")
bullets([
    "<b>Reporting bias:</b> communities with better access to hospitals report more completely. Risk may be "
    "under-estimated in under-served areas, such as low-income urban settlements where dengue is often worst. If "
    "resources follow the model, these communities could receive less help.",
    "<b>Geographic bias:</b> the model learned Colombo's seasonality and should not be applied to other "
    "districts without retraining and validation.",
    "<b>Asymmetric errors:</b> the cut-off deliberately favours recall over precision. This is a value judgement "
    "that should be agreed with public-health officials.",
])
H2("5.4 Real-world impact and ethical considerations")
bullets([
    "<b>Positive impact:</b> a four-week warning allows fogging, clean-up drives, school programmes and hospital "
    "preparation to be scheduled before the peak, potentially reducing illness and deaths.",
    "<b>Privacy:</b> only aggregated, anonymous district-level counts are used.",
    "<b>Human in the loop:</b> the system is decision support, not an automatic decision maker. Alerts should be "
    "reviewed by epidemiologists together with field reports. The explanation shown with every prediction "
    "supports transparency and accountability.",
    "<b>Alert fatigue:</b> about one in three alerts in 2013 was a false alarm. Too many false alarms could make "
    "staff ignore warnings, so the cut-off should be reviewed with users.",
])

# ================================================================== 6. FRONT END
H1("6. Front-End Integration")
P("The complete pipeline (column selection, imputation, winsorisation, scaling and Extra Trees), together with "
  "the threshold and alert cut-off, was saved as one <i>joblib</i> file. A <b>Streamlit</b> web application "
  "(<i>app.py</i>) loads it and provides:")
bullets([
    "<b>Input:</b> the notified cases for the last eight weeks and the date of the latest week. Users can type "
    "their own values or load real historical examples (a rising outbreak, a quiet period, the 2011 peak).",
    "<b>Prediction:</b> the probability of a high-risk month, a progress bar and a clear alert (high risk with "
    "recommended actions, or normal risk).",
    "<b>Explanation:</b> a SHAP waterfall chart for that specific prediction, labelled in everyday units (for "
    "example \"170 = cases this week\"), plus a bar chart of the eight weeks against the threshold.",
    "<b>Model information:</b> the algorithm, training period, threshold and test performance.",
    "Input validation, for example a warning when a week has zero cases, which usually indicates a missing "
    "report.",
])
P("The same feature-engineering code (<i>src/data_prep.py</i>) is used in training and in the app, so the "
  "transformations are identical. The demonstration video shows the application in use.")

# ================================================================== 7. CONCLUSION
H1("7. Conclusion")
P("This project developed a dengue early-warning classifier for Colombo using real Sri Lankan surveillance "
  "data. Careful data exploration revealed and corrected several quality problems that would otherwise have "
  "biased the model. The proposed algorithm, Extremely Randomized Trees, matched the best taught algorithms "
  "on ranking ability and, on an unseen year, detected 13 of 15 high-risk weeks four weeks ahead with the highest "
  "balanced accuracy of all models. Explainability analysis with SHAP, PDP and LIME showed that the model relies "
  "on season, case momentum and current case levels, consistent with dengue epidemiology. Future work should use "
  "more recent data and more districts, add rainfall forecasts to anticipate outbreak onsets, and retrain the "
  "model regularly.")

H1("References")
for ref in [
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5–32.",
    "Epidemiology Unit, Ministry of Health, Sri Lanka. <i>Weekly Epidemiological Report</i>. https://www.epid.gov.lk",
    "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. <i>Annals of "
    "Statistics</i>, 29(5), 1189–1232.",
    "Geurts, P., Ernst, D., &amp; Wehenkel, L. (2006). Extremely randomized trees. <i>Machine Learning</i>, "
    "63(1), 3–42.",
    "Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. "
    "<i>Advances in Neural Information Processing Systems 30</i>.",
    "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. <i>Journal of Machine Learning "
    "Research</i>, 12, 2825–2830.",
    "Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). \"Why should I trust you?\" Explaining the "
    "predictions of any classifier. <i>Proceedings of the 22nd ACM SIGKDD Conference</i>, 1135–1144.",
    "Talagala, T. S. <i>colmozzie: Dengue Cases and Climate Variables in Colombo, Sri Lanka</i> (R package "
    "version 1.1.1). https://github.com/thiyangt/colmozzie",
]:
    story.append(Paragraph(ref, ParagraphStyle("ref", parent=body, leftIndent=14, firstLineIndent=-14,
                                               alignment=0, fontSize=9.2, leading=12.5)))

H1("Appendix: Reproducibility")
P("The source code repository contains the dataset (<i>data/raw</i>), the full notebook with all outputs "
  "(<i>notebooks/dengue_outbreak_early_warning.ipynb</i>), the shared pre-processing module "
  "(<i>src/data_prep.py</i>), the saved model (<i>models/</i>), all figures and result tables (<i>reports/</i>) "
  "and the Streamlit application (<i>app.py</i>). A fixed random seed (42) is used throughout. To reproduce: "
  "install <i>requirements.txt</i>, run the notebook, then start the app with <i>streamlit run app.py</i>.")

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                        bottomMargin=2 * cm, title="Dengue Outbreak Early Warning – Colombo",
                        author=STUDENT_NAME)
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Saved", OUT)
