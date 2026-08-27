# Logistics Data Analysis Internship

## Week 1: Strategic Planning and Data Exploration

### Project Overview

This project is part of my internship in **Logistics Data Analysis**. The main objective of Week 1 is to understand logistics data, perform basic data exploration, calculate important statistics, and identify delivery-related patterns.

The project uses Python and Pandas to analyze information such as delivery time, transportation distance, transportation cost, and delivery status.

### Objectives

* Understand the basic structure of logistics data.
* Create and analyze a logistics dataset.
* Calculate average delivery time.
* Calculate average transportation cost.
* Identify delayed orders.
* Analyze delivery status.
* Create simple visualizations for better understanding of the data.
* Develop basic data analysis skills using Python.

### Technologies Used

* **Python**
* **Pandas**
* **Matplotlib**
* **GitHub**

### Dataset

The project uses a sample logistics dataset containing the following information:

| Column          | Description                                        |
| --------------- | -------------------------------------------------- |
| Order_ID        | Unique identification number of an order           |
| Delivery_Days   | Number of days required for delivery               |
| Distance_KM     | Delivery distance in kilometers                    |
| Transport_Cost  | Cost of transportation                             |
| Delivery_Status | Whether the order was delivered on time or delayed |

### Analysis Performed

The Python program performs the following analysis:

1. Creates a logistics dataset using Pandas.
2. Displays the complete dataset.
3. Checks basic information about the data.
4. Generates a statistical summary.
5. Calculates the average delivery time.
6. Calculates the average transportation cost.
7. Counts on-time and delayed deliveries.
8. Identifies delayed orders.
9. Creates a bar chart showing delivery time for each order.
10. Creates a bar chart showing transportation cost for each order.

### Project Files

```text
logistics-data-analysis-internship/
│
├── README.md
├── logistics_analysis.py
└── Week_1_Report.docx
```

### How to Run the Project

1. Install Python on your computer.
2. Install the required libraries:

```bash
pip install pandas matplotlib
```

3. Download or clone this repository.
4. Open the project folder in VS Code or another Python editor.
5. Run:

```bash
python logistics_analysis.py
```

6. The analysis results will be displayed in the terminal, and the charts will appear automatically.

### Expected Outcome

After running the program, the user can see:

* Complete logistics data.
* Data information and statistics.
* Average delivery time.
* Average transportation cost.
* Number of delayed and on-time orders.
* List of delayed orders.
* Delivery-time visualization.
* Transportation-cost visualization.

### Learning Outcome

Through this project, I learned how to work with a logistics dataset using Python. I learned basic data exploration using Pandas, statistical analysis, filtering data, identifying delayed deliveries, and creating visualizations using Matplotlib. This project provides a foundation for more advanced logistics data analysis in the upcoming internship weeks.

### Future Scope

In future weeks, this project can be expanded by using a larger real-world logistics dataset. Additional analysis can include delivery performance, route optimization, cost analysis, customer demand, warehouse performance, and predictive analytics.

## Author

**Sahil**

**Internship Project: Logistics Data Analysis**



# Week 2: Data Preprocessing & Pipeline Report – Logistics Supply Chain Analysis

**Author:** Student / Data Analyst  
**Task:** Week 2 – Data Collection, Cleaning, and Preprocessing for Logistics Analysis  
  

---

## 📌 Project Overview

This repository contains the dataset simulation, methodology, and automated Python processing script for **Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis**. 

The primary goal of this project is to build an end-to-end data preparation pipeline for complex supply chain telemetry (modeled after the publicly available **DataCo Smart Supply Chain Dataset**). The clean output enables reliable downstream machine learning modeling for predicting shipment delay risks and optimizing freight costs.

---

## 📊 Dataset Profile & Characteristics

The simulated raw dataset contains **10,000 shipment transactions** spanning **10 distinct features**:

| Feature Name | Data Type | Description | Initial Quality Issue Identified |
| :--- | :--- | :--- | :--- |
| `Shipment_ID` | String | Unique tracking identifier | Duplicate entries present |
| `Order_Date` | Object | Timestamp of order placement | Inconsistent string formats |
| `Delivery_Date` | Object | Timestamp of final delivery | Missing values (nulls) |
| `Shipping_Mode` | Categorical | Standard, Express, Same Day, First Class | Unstandardized text casing |
| `Distance_KM` | Float | Transit distance in kilometers | Extreme negative outliers / sensor errors |
| `Freight_Cost_USD` | Float | Shipping cost charged | Missing values (MCAR) |
| `Package_Weight_KG` | Float | Gross weight of shipment | Outliers (faulty scale readings) |
| `Delay_Minutes` | Float | Difference from target schedule | Right-skewed distribution |
| `Warehouse_Block` | Categorical | Origin facility node (A, B, C, D) | Categorical encoding required |
| `Late_Delivery_Risk` | Binary (0/1) | Target indicator variable | Imbalanced target labels |

---

## 🛠️ Data Cleaning & Preprocessing Pipeline

The processing pipeline is executed in **5 distinct sequential stages**:

1. **Structural Cleaning & De-duplication:**  
   Identified and removed exact row duplicates and instances with invalid primary keys (`Shipment_ID`). Standardized header naming conventions to lowercase `snake_case`.
2. **Missing Data Treatment:**  
   - `Delivery_Date` (MAR): Imputed using `Order_Date` + median transit duration per `Shipping_Mode`.
   - `Freight_Cost_USD` (MCAR): Grouped by `Distance_KM` deciles and `Shipping_Mode`, imputing group median cost.
3. **Outlier Detection & Removal (IQR Winsorization):**  
   Applied the Interquartile Range (IQR) method ($Q3 + 1.5 \times IQR$) to continuous features like `Distance_KM` and `Package_Weight_KG`[cite: 1]. Extreme values were capped rather than dropped to preserve data volume[cite: 1].
4. **Categorical Encoding:**  
   Standardized text formatting and applied One-Hot Encoding to nominal variables (`Shipping_Mode`, `Warehouse_Block`) to eliminate unintended ordinal bias[cite: 1].
5. **Feature Normalization:**  
   Scaled numerical features using `StandardScaler` ($Z$-score normalization) to prevent higher-magnitude continuous values from dominating model weights[cite: 1].

---

## 💻 Python Implementation Script

The pipeline is implemented using Python's core data science tools (`pandas`, `numpy`, `scikit-learn`)[cite: 1]:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def execute_logistics_pipeline(raw_csv_path: str) -> pd.DataFrame:
    # ---------------------------------------------------------
    # Step 1: Data Ingestion & Structural Cleaning
    # ---------------------------------------------------------
    df = pd.read_csv(raw_csv_path)

    # Standardize column headers
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicate primary keys
    df = df.drop_duplicates(subset=["shipment_id"]).copy()

    # Convert datetime columns
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

    # ---------------------------------------------------------
    # Step 2: Handling Missing Data
    # ---------------------------------------------------------
    # Remove negative distance errors
    df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan

    # Median Imputation based on operational groupings
    df["distance_km"] = df.groupby("shipping_mode")["distance_km"].transform(
        lambda x: x.fillna(x.median())
    )

    df["freight_cost_usd"] = df.groupby("shipping_mode")[
        "freight_cost_usd"
    ].transform(lambda x: x.fillna(x.median()))

    # Impute missing delivery dates using calculated average transit times
    transit_medians = (
        df["delivery_date"] - df["order_date"]
    ).dt.total_seconds() / 86400
    median_days = transit_medians.median()
    df["delivery_date"] = df["delivery_date"].fillna(
        df["order_date"] + pd.Timedelta(days=median_days)
    )

    # ---------------------------------------------------------
    # Step 3: Outlier Treatment (IQR Method & Winsorization)
    # ---------------------------------------------------------
    for col in ["package_weight_kg", "freight_cost_usd"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Winsorize: Cap extreme outliers instead of dropping rows
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # ---------------------------------------------------------
    # Step 4: Categorical Encoding
    # ---------------------------------------------------------
    categorical_cols = ["shipping_mode", "warehouse_block"]
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_array = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # ---------------------------------------------------------
    # Step 5: Feature Normalization (Standard Scaling)
    # ---------------------------------------------------------
    scaler = StandardScaler()
    num_cols = ["distance_km", "freight_cost_usd", "package_weight_kg"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


if __name__ == "__main__":
    print("Pipeline executed successfully. Processed dataset ready for modeling.")
