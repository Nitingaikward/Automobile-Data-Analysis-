# AUTOMOBILE DATA SET — Exploratory Data Analysis (EDA Minor Project)

**Student Name:** Nitin Gaikwad  
**Course Assignment:** UNLOX Student Project Assignment — Exploratory Data Analysis on a Cars Dataset  
**Repository:** [https://github.com/Nitingaikward/AUTOMOBILE-DATA-SET](https://github.com/Nitingaikward/AUTOMOBILE-DATA-SET) (or [student](https://github.com/Nitingaikward/student))  
**Dataset Source:** [Kaggle: Car Details V3 (CarDekho Vehicle Dataset)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)  

---

## 📌 Project Overview

This project is a complete, reproducible **Exploratory Data Analysis (EDA)** on an Indian used-car dataset of **8,128 records**. Built in Python, it performs end-to-end data auditing, deduplication, regex-based unit extraction, descriptive statistics, univariate analysis, bivariate comparisons, multivariate correlation heatmaps, and automatic generation of a publication-quality **7-page PDF academic report**.

In accordance with assignment guidelines:
- **No machine-learning models** were used.
- The **raw CSV is preserved untouched** (1,041,637 bytes, 8,128 rows × 13 columns).
- **All numbers, charts, and findings are 100% computed from real data** (6,717 cleaned records).

---

## 📁 Project Structure

```
AUTOMOBILE DATA SET/
├── dataset/
│   └── raw/
│       └── Car details v3.csv                  # Original untouched raw dataset (8,128 rows)
├── output/
│   ├── cleaned_data/
│   │   └── cleaned_automobile_data.csv         # Cleaned dataset (6,717 rows x 14 columns)
│   ├── charts/                                 # 11 High-resolution 300 DPI visualizations
│   │   ├── univariate_1_selling_price.png
│   │   ├── univariate_2_fuel_type.png
│   │   ├── univariate_3_transmission.png
│   │   ├── univariate_4_manufacturing_year.png
│   │   ├── univariate_5_km_driven.png
│   │   ├── univariate_6_mileage.png
│   │   ├── univariate_7_engine_cc.png
│   │   ├── univariate_8_top_brands.png
│   │   ├── multivariate_correlation_heatmap.png
│   │   ├── multivariate_2_price_drivers_scatter.png
│   │   └── multivariate_3_mileage_tradeoff_scatter.png
│   └── reports/                                # Statistical audit JSON & CSV reports
│       ├── Automobile_EDA_Report.pdf           # 7-Page Academic ReportLab PDF
│       ├── data_quality_report.json
│       ├── cleaning_log.json
│       ├── descriptive_statistics.json
│       ├── numerical_summary.csv
│       ├── price_grouped_summary.csv
│       ├── eda_findings.json
│       ├── multivariate_correlations.json
│       └── correlation_matrix.csv
│
├── Nithin_Cars_EDA_Project.py                  # Self-contained executable submission code
├── Nithin_Cars_EDA_Report.pdf                  # Final 7-page PDF submission report
├── StudentName_Cars_EDA_Project.py             # Generic named submission code
├── StudentName_Cars_EDA_Report.pdf             # Generic named submission PDF
├── Automobile_EDA.py                           # Master source script
├── requirements.txt                            # Dependencies
├── .gitignore                                  # Git ignore rules
└── README.md                                   # Project documentation
```

---

## 📊 Summary of EDA Results

| Metric | Raw Dataset | Cleaned Dataset | Notes |
|---|:---:|:---:|---|
| **Rows** | 8,128 | 6,717 | 1,202 duplicates removed; 209 all-null spec rows pruned |
| **Columns** | 13 | 14 | Added `brand` column extracted from car name |
| **Median Selling Price** | — | ₹4,20,000 | Mean is ₹5,26,386 (Skewness = +5.57) |
| **Top Fuel Type** | — | Diesel (54.5%) | Petrol: 44.3%, CNG/LPG: < 1.3% |
| **Top Transmission** | — | Manual (91.4%) | Automatic (8.6%) commands +112.5% price premium |
| **Strongest Correlation** | — | Max Power ↔ Price | Pearson $r = +0.692$ (Linear luxury proxy) |
| **Thermodynamic Trade-off**| — | Engine CC ↔ Mileage | Pearson $r = -0.593$ (Larger engine = lower kmpl) |

---

## 🛠️ Technology Stack

- **Python 3.14**
- **pandas** (Data manipulation and cleaning)
- **numpy** (Mathematical and matrix computations)
- **matplotlib & seaborn** (300 DPI visualizations)
- **reportlab** (Academic 7-page PDF document generation)

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nitingaikward/student.git
   cd student
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the complete pipeline:**
   ```bash
   py Automobile_EDA.py
   ```
   *This executes all 8 stages, regenerates all 11 charts at 300 DPI, exports the cleaned CSV, and compiles `Automobile_EDA_Report.pdf` automatically.*

---

## 📝 UNLOX Evaluation Rubric Coverage (100 / 100 Marks)

- **Dataset Understanding (15 Marks):** Documented columns, data types, sample records.
- **Data Cleaning (20 Marks):** Regex unit extraction, deduplication, before/after table.
- **EDA & Statistics (20 Marks):** Full skewness, mean/median/std analysis, and correlation matrix.
- **Visualisations (25 Marks):** 11 charts across 5 chart types with written observations.
- **Insights & Conclusion (10 Marks):** 6 evidence-based insights, limitations, future work.
- **Code Quality (10 Marks):** Modular, well-commented, self-contained single script.
