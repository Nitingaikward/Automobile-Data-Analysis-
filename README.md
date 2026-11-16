# Automobile Data Analysis (EDA Minor Project)

A Python-based Exploratory Data Analysis (EDA) and Data Preprocessing minor project on a used car sales dataset containing 8,128 vehicle listings.

---

## 📁 Project Folder Structure

```
Automobile_Data_Analysis/
│
├── dataset/
│   └── raw/
│       └── Car details v3.csv       # Original untouched raw dataset (8,128 records)
│
├── output/
│   ├── cleaned_data/
│   │   └── cleaned_automobile_data.csv  # Cleaned dataset (6,717 records x 14 features)
│   ├── charts/                      # Visualizations output directory
│   └── reports/                     # Statistical & data quality reports
│       ├── data_quality_report.json
│       ├── cleaning_log.json
│       ├── descriptive_statistics.json
│       ├── numerical_summary.csv
│       └── price_grouped_summary.csv
│
├── Automobile_EDA.py                # Main executable Python pipeline
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## 🛠️ Tech Stack & Dependencies

The project relies on standard Python scientific data science packages:

| Library | Version | Purpose |
| :--- | :--- | :--- |
| **`pandas`** | 3.1.1 | Tabular DataFrame manipulation, loading, and cleaning |
| **`numpy`** | 2.2.3 | Fast mathematical operations and array computations |
| **`matplotlib`** | 3.11.2 | Foundational charting and plotting |
| **`seaborn`** | 0.13.2 | High-level statistical visualization graphics |
| **`scipy`** | 1.18.1 | Statistical skewness, kurtosis, and correlation tests |
| **`reportlab`** | 5.1.0 | Formatted PDF summary report generation |

---

## 🚀 How to Run the Project

1. Open your terminal in the `Automobile_Data_Analysis` folder:
   ```powershell
   cd Automobile_Data_Analysis
   ```

2. Execute the main script using the official Python launcher:
   ```powershell
   py Automobile_EDA.py
   ```

*(Note: On Windows, use `py` to invoke the configured Python 3.14 environment).*

---

## 📊 Pipeline Stages Implemented

### Stage 1: Safe CSV Dataset Loading
- Dynamically resolves paths using `pathlib.Path` relative to script location.
- Reads raw dataset into memory using `pd.read_csv()` in read-only mode.
- Validates baseline shape: **8,128 rows $\times$ 13 columns**.

### Stage 2: Dataset Understanding & Structural Inspection
- Automatically separates variables into numerical and categorical types.
- Analyzes domain column meanings (`name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`).
- Uncovers units embedded in text strings (`kmpl`, `km/kg`, `CC`, `bhp`).

### Stage 3: Data Quality Assessment
- Preserves raw baseline copy (`df_original`).
- Identifies **1,202 exact duplicate records (14.79%)**.
- Audits missing values across 5 mechanical columns (~2.7% each).
- Detects hidden anomalies: 17 records with `'0.0 kmpl'` (invalid zero mileage) and 1 record with `' bhp'` (blank power).
- Calculates IQR outlier bounds for all numeric variables.
- Exports audit findings to `output/reports/data_quality_report.json`.

### Stage 4: Data Cleaning & Preprocessing
- **Deduplication**: Drops 1,202 identical listings, leaving 6,926 unique cars.
- **Brand Extraction**: Extracts clean automotive manufacturer names across 32 brands (Maruti, Hyundai, Toyota, etc.).
- **Unit Conversion**:
  - `mileage_kmpl`: Converted to float; zero mileage marked as `NaN`.
  - `engine_cc`: Stripped `" CC"` and converted to float.
  - `max_power_bhp`: Stripped `" bhp"` and converted to float.
  - `torque_nm`: Standardized into Newton-meters (converted `kgm` to `Nm` via $1 \text{ kgm} = 9.80665 \text{ Nm}$).
- **Justified Missing Value Treatments**:
  - Dropped 209 incomplete listings lacking all recorded mechanical specifications (avoids fabricating data).
  - Imputed 15 zero-mileage records using `(brand, fuel)` group medians.
  - Imputed missing seating capacity using the mode (`5.0`).
- **Outlier Preservation**: Retained 282 luxury vehicles (up to ₹1,00,00,000) and 162 fleet vehicles (> 192,500 km) as genuine real-world variance.
- **Export**: Saved clean, 0-null dataset to `output/cleaned_data/cleaned_automobile_data.csv` (6,717 rows $\times$ 14 columns).
- **Audit**: Logged all operations to `output/reports/cleaning_log.json`.

### Stage 5: Descriptive Statistical Analysis
- **Numerical Summary**: Calculates Count, Mean, Median, Std Dev, Min, Max, and Skewness.
- **Skewness Insights**:
  - `selling_price`: Strongly right-skewed (+5.57). Mean (₹5.26 Lakhs) is ₹1.06 Lakhs higher than Median (₹4.20 Lakhs) due to luxury cars. Median is the most representative metric of market price.
  - `km_driven`: Extreme right skew (+11.92) caused by commercial fleet vehicles.
  - `year`: Moderately left-skewed (-0.93) with peak inventory between 2011 and 2018.
  - `mileage_kmpl`: Perfectly symmetric (+0.06) with a balanced bell curve around ~19.5 kmpl.
- **Categorical Distributions**:
  - Fuel: Diesel (54.46%), Petrol (44.26%), CNG (0.76%), LPG (0.52%).
  - Transmission: Manual (91.44%), Automatic (8.56%).
  - Seller: Individual (89.68%), Dealer (9.92%), Trustmark Dealer (0.40%).
  - Ownership: First Owner (62.17%), Second Owner (28.11%), Third Owner (7.34%).
- **Market Price Summaries**:
  - Automatic cars command **+112.5% higher median price** (₹8.5 Lakhs vs ₹4.0 Lakhs).
  - Diesel cars command **+64.1% higher median price** over Petrol (₹5.25 Lakhs vs ₹3.2 Lakhs).
  - Resale price drops by **36%** from 1st owner to 2nd owner.
  - Toyota leads volume brand resale value (₹7.5 Lakhs median), while Maruti dominates market volume (2,089 cars, ₹3.5 Lakhs median).
- **Exports**: Saved statistical reports to `output/reports/descriptive_statistics.json`, `numerical_summary.csv`, and `price_grouped_summary.csv`.

---

## 🔒 Data Integrity Guarantees
- Original raw CSV in `dataset/raw/Car details v3.csv` is **read-only and never modified**.
- No synthetic or fabricated data is introduced.
- All transformations are logged and reproducible.
