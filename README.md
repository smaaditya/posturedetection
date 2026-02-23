**🦴 SpinalSense**

IMU-Based Posture & Activity Detection

_Project Documentation & Usage Guide_

| Random ForestModel Type | Dual-ModelArchitecture | 19 FeaturesInput Dimensions | 5 SubjectsTraining Subjects |
| --- | --- | --- | --- |

Posture Model Accuracy

**99.94%**

Activity Model Accuracy

**85.40%**

# 1\. Project Overview

SpinalSense is a machine learning pipeline that uses **IMU (Inertial Measurement Unit) sensor data** from five spinal landmarks to simultaneously detect a person's **posture type** and **physical activity** in real time. Two independent Random Forest classifiers run in parallel — one for each prediction task.

## What It Detects

| 🦴 Posture Classes | 🏃 Activity Classes |
| --- | --- |
| Normal | Sitting |
| Cervical Lordosis | Standing |
| Lumbar Lordosis | Walking |
| Thoracic Kyphosis |  |
| Scoliosis |  |

# 2\. Project Files

All files should live in the same folder. Here is what each one does:

| Filename | Type | Purpose |
| --- | --- | --- |
| IMU_Spinal_Balanced.csv | Dataset | Cleaned and balanced training dataset. 10,226 rows, 23 columns. |
| posture_pipeline.py | Script | Main pipeline. Handles training, batch prediction, and evaluation. |
| row_predict.py | Script | Interactive row predictor. Enter any Excel row number to get predictions. |
| rf_posture.pkl | Model | Trained Random Forest model for posture detection (5 classes). |
| rf_activity.pkl | Model | Trained Random Forest model for activity detection (3 classes). |
| le_posture.pkl | Encoder | Label encoder for posture class names. |
| le_activity.pkl | Encoder | Label encoder for activity class names. |
| model_meta.json | Config | Feature column names, class lists, and top feature importances. |
| SpinalSense_Demo.html | App | Interactive browser demo. Sliders for all 19 features, live predictions. |
| SpinalSense_Metrics_Report.html | Report | Full performance report. Confusion matrices, F1 scores, LOPO results. |

# 3\. Prerequisites

## Python Packages

Install all required packages with:

| pip install pandas numpy scikit-learn |
| --- |

The following packages are used:

*   pandas — loading and manipulating the CSV dataset
*   numpy — numerical operations and array handling
*   scikit-learn — Random Forest classifiers and evaluation metrics
*   pickle — saving and loading trained model files (built into Python)

# 4\. Quick Start

Follow these steps in order the first time you set up the project.

## Step 1 — Verify your folder

Make sure all files listed in Section 2 are in the same folder. Open a terminal in that folder:

| cd C:\Users\YourName\posturedetection |
| --- |

## Step 2 — Retrain the models (optional)

The .pkl model files are already trained and ready to use. You only need to retrain if you want to update the models with new data:

| python posture_pipeline.py --mode train --data IMU_Spinal_Balanced.csv |
| --- |

This will overwrite rf\_posture.pkl, rf\_activity.pkl, le\_posture.pkl, le\_activity.pkl, and model\_meta.json.

## Step 3 — Open the demo app

No Python needed. Just run this in your terminal:

| start SpinalSense_Demo.html |
| --- |

Or double-click the file in Windows Explorer. It opens in your browser with 14 preloaded scenarios and adjustable sliders.

## Step 4 — Predict from a row in the dataset

Run the interactive row predictor:

| python row_predict.py |
| --- |

Type any Excel row number between **2** and **10227** and press Enter.

# 5\. posture\_pipeline.py — Main Pipeline

This is the core script. It has three modes controlled by the --mode flag.

## Mode: train

Trains both models from scratch using the CSV dataset and saves all model files.

| python posture_pipeline.py --mode train --data IMU_Spinal_Balanced.csv |
| --- |

Output files created:

*   rf\_posture.pkl — posture Random Forest model
*   rf\_activity.pkl — activity Random Forest model
*   le\_posture.pkl — posture label encoder
*   le\_activity.pkl — activity label encoder
*   model\_meta.json — feature names and class lists

## Mode: predict

Runs predictions on a .json or .csv input file and saves results.

| python posture_pipeline.py --mode predict --input input.jsonpython posture_pipeline.py --mode predict --input input.csv |
| --- |

Your input file must have all **19 feature columns** as flat keys. See Section 7 for the exact format.

Output is saved as input\_predictions.json in the same folder.

## Mode: evaluate

Runs a full model evaluation — 5-fold cross-validation and Leave-One-Person-Out testing.

| python posture_pipeline.py --mode evaluate --data IMU_Spinal_Balanced.csv |
| --- |

Prints per-class precision, recall, F1 scores, and LOPO accuracy for each subject.

## Calling predict\_single() from your own code

Import the predict\_single() function directly into any Python script:

| from posture_pipeline import predict_singlefeatures = {"C1x": 0.664406, "C1y": 0.053927, "C1z": -0.116920,"C7x": 0.425013, "C7y": -0.156756, "C7z": 0.077966,"T5x": 0.289437, "T5y": 0.005209, "T5z": -0.074793,"T12x": 0.604223, "T12y": 0.255447, "T12z": -0.110010,"L5x": 0.422400, "L5y": -0.087597, "L5z": -0.168612,"cervical_angle": 41.64, "thoracic_angle": 25.15,"lumbar_angle": 37.17, "scoliosis_angle": 3.09}result = predict_single(features)print(result["posture"]) # e.g. "normal"print(result["posture_confidence_pct"]) # e.g. 100.0print(result["activity"]) # e.g. "standing" |
| --- |

# 6\. row\_predict.py — Row Predictor

Lets you enter any **Excel row number** (rows 2–10227, where row 1 is the header) and instantly see the posture and activity prediction for that row's sensor data.

## Interactive Mode

Keeps running until you type q. Enter as many rows as you like:

| python row_predict.py# Then type any row number:Enter Excel row (2-10227) > 42Enter Excel row (2-10227) > 1500Enter Excel row (2-10227) > q # to quit |
| --- |

## Single Row Mode

Pass the row number directly as a command-line argument:

| python row_predict.py --row 42python row_predict.py --row 4947 |
| --- |

## Custom Dataset Path

By default it looks for IMU\_Spinal\_Balanced.csv in the same folder. Use --data to point to a different file:

| python row_predict.py --data C:\path\to\MyDataset.csv |
| --- |

## Output Explained

For every row, the predictor shows:

*   **Predicted posture** — with confidence % and a visual bar
*   **✓ / ✗** — whether the prediction matches the actual label in the dataset
*   **All class probabilities** — for all 5 posture classes
*   **Predicted activity** — with confidence % and colour-coded bar
*   **All activity probabilities** — for all 3 activity classes
*   **Key sensor values** — the 8 most important features used for that prediction

Confidence is colour-coded: green ≥ 90%, yellow 70–89%, red below 70%.

# 7\. Input Format for Batch Prediction

When using --mode predict you must provide an input file with all 19 features as flat keys. There are no nested objects.

## Single Row (JSON)

| {"C1x": 0.664406, "C1y": 0.053927, "C1z": -0.116920,"C7x": 0.425013, "C7y": -0.156756, "C7z": 0.077966,"T5x": 0.289437, "T5y": 0.005209, "T5z": -0.074793,"T12x": 0.604223, "T12y": 0.255447, "T12z": -0.110010,"L5x": 0.422400, "L5y": -0.087597, "L5z": -0.168612,"cervical_angle": 41.64,"thoracic_angle": 25.15,"lumbar_angle": 37.17,"scoliosis_angle": 3.09} |
| --- |

## Multiple Rows (JSON Array)

| [{ "C1x": 0.664, "C1y": 0.053, ... },{ "C1x": 0.880, "C1y": 0.143, ... }] |
| --- |

## CSV Format

Column headers must exactly match the 19 feature names. Case-sensitive — C1x not c1x.

| C1x,C1y,C1z,C7x,C7y,C7z,T5x,T5y,T5z,T12x,T12y,T12z,L5x,L5y,L5z,cervical_angle,thoracic_angle,lumbar_angle,scoliosis_angle0.664,0.053,-0.116,0.425,-0.156,0.077,0.289,0.005,-0.074,0.604,0.255,-0.110,0.422,-0.087,-0.168,41.64,25.15,37.17,3.09 |
| --- |
| ⚠️ Common ErrorsWrong key names: Keys are case-sensitive. "C1x" ≠ "c1x" ≠ "C1X"Nested objects: All 19 features must be at the top level, not inside {"sensors": {...}}Missing angles: You must include all 4 angle columns — not just the 15 raw sensor columns.Extra wrapper: {"data": {...}} will fail — remove any wrapper key. |
| --- |

# 8\. Feature Reference

All 19 input features and their valid ranges:

| Feature | Spinal Location | Min | Max | Unit |
| --- | --- | --- | --- | --- |
| C1x / C1y / C1z | Atlas (cervical top) | -1.0 / -0.38 / -0.96 | 1.0 / 0.54 / 0.85 | normalised |
| C7x / C7y / C7z | Cervicothoracic junction | -0.01 / -0.88 / -0.83 | 1.06 / 0.87 / 0.78 | normalised |
| T5x / T5y / T5z | Mid-thoracic | -0.80 / -0.86 / -0.77 | 0.83 / 0.77 / 0.91 | normalised |
| T12x / T12y / T12z | Thoracolumbar junction | 0.21 / -0.85 / -0.81 | 1.18 / 0.81 / 0.79 | normalised |
| L5x / L5y / L5z | Lumbar / sacral junction | 0.27 / -0.39 / -0.86 | 1.09 / 0.41 / 0.82 | normalised |
| cervical_angle | C1 angle from vertical | 0 | 70 | ° |
| thoracic_angle | T5 angle from vertical | 0 | 60 | ° |
| lumbar_angle | T12–L5 angle from vertical | 10 | 80 | ° |
| scoliosis_angle | Lateral spinal deviation | -20 | +20 | ° |

# 9\. Model Performance Summary

## Posture Model

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Cervical Lordosis | 99.95% | 100.00% | 99.97% | 1,994 |
| Lumbar Lordosis | 99.85% | 99.93% | 99.89% | 1,357 |
| Normal | 99.97% | 100.00% | 99.99% | 3,732 |
| Scoliosis | 100.00% | 99.86% | 99.93% | 1,454 |
| Thoracic Kyphosis | 99.78% | 99.57% | 99.68% | 463 |
| Overall / Macro Avg | 99.91% | 99.87% | 99.89% | 9,000 |

## Activity Model

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Sitting | 96.41% | 96.57% | 96.49% | 3,000 |
| Standing | 79.64% | 81.10% | 80.36% | 3,000 |
| Walking | 80.14% | 78.53% | 79.33% | 3,000 |
| Overall / Macro Avg | 85.39% | 85.40% | 85.39% | 9,000 |
| 📊 Activity Model NoteThe activity model achieves 85% in cross-validation but drops to ~56% in Leave-One-Person-Out testing. This means it partially learns person-specific movement patterns. The main confusion is between standing and walking — both produce similar static spinal angles. Adding temporal motion features (e.g. acceleration variance, jerk) in a future version would significantly improve generalisation to new users. |
| --- |

# 10\. Opening the HTML Files

Both HTML files are fully self-contained — no server required. Open them directly in any browser.

## SpinalSense\_Demo.html

*   Interactive prediction demo with sliders for all 19 features
*   14 real dataset presets covering every posture and activity combination
*   Live spinal diagram that highlights affected vertebrae in colour
*   Probability bars for all prediction classes

Open it:

| start SpinalSense_Demo.html |
| --- |

## SpinalSense\_Metrics\_Report.html

*   Full model evaluation report with all performance metrics
*   Colour-coded confusion matrices for both models
*   Per-class F1, precision, and recall tables with inline bar charts
*   Feature importance rankings
*   Leave-One-Person-Out results per subject

Open it:

| start SpinalSense_Metrics_Report.html |
| --- |

# 11\. Troubleshooting

| Error | Fix |
| --- | --- |
| ValueError: Input is missing required columns | Your input JSON is missing one or more of the 19 feature keys. Check all keys are spelled correctly (case-sensitive) and at the top level. |
| FileNotFoundError: rf_posture.pkl not found | Run training first: python posture_pipeline.py --mode train --data IMU_Spinal_Balanced.csv |
| ModuleNotFoundError: No module named sklearn | Install dependencies: pip install pandas numpy scikit-learn |
| Row out of range error in row_predict.py | Valid Excel rows are 2 to 10227. Row 1 is the header and is not valid. |
| HTML file shows blank or broken layout | Use a modern browser (Chrome, Edge, Firefox). The file needs internet access for Google Fonts on first load. |
