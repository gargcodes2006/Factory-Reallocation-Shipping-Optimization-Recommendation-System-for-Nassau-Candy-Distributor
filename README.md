# 🏭 Factory Reallocation & Shipping Optimization Recommendation System

> **For Nassau Candy Distributor — Internship Demonstration Project**

A professional data-science application that uses **linear programming** to find the minimum-cost shipment allocation from factories to distribution centers while respecting production capacity and demand constraints.

> **⚠️ Disclaimer:** All data, results, and savings figures in this project are **illustrative sample data** created for demonstration purposes. They do not represent actual Nassau Candy business data or verified operational savings.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives & Key Features](#objectives--key-features)
- [Technologies Used](#technologies-used)
- [Dataset Descriptions](#dataset-descriptions)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Uploading Custom Data](#uploading-custom-data)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## Project Overview

Nassau Candy is a large confectionery distributor that ships products from multiple factory locations to distribution centers across the country. This project models that supply chain as a **transportation problem** and solves it with LP optimisation to:

- **Minimise total shipping costs** while meeting every distribution center's demand.
- **Compare** the optimal solution against a feasible baseline to quantify savings.
- **Visualise** the results in an interactive Streamlit dashboard.
- **Generate data-driven recommendations** for decision-makers.

### Business Value

Even modest percentage reductions in transportation costs translate to significant savings at scale. This tool gives supply-chain managers an immediate, data-driven answer to *"Which factory should ship how much to which DC?"*

---

## Objectives & Key Features

| Feature | Description |
|---|---|
| **LP Optimisation** | PuLP-based solver minimises total transportation (and optionally production) cost |
| **Capacity & Demand Constraints** | Every DC's demand is met; no factory exceeds its capacity |
| **Baseline Comparison** | Greedy heuristic baseline provides a realistic "before" scenario |
| **CSV Upload** | Users can load their own factories, destinations, and cost data |
| **Data Validation** | Missing columns, duplicates, negatives, and cross-reference errors are caught |
| **Interactive Dashboard** | Six-page Streamlit app with KPI cards, charts, and recommendations |
| **Plotly Visualisations** | Bar charts, stacked charts, Sankey network diagram, cost comparisons |
| **Auto Recommendations** | Dynamic insights generated from the current data — nothing is hardcoded |
| **Export** | Download shipment plan CSV, cost comparison CSV, and a summary report |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Data processing |
| **PuLP** | Linear programming solver |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts |
| **Pytest** | Automated testing |

---

## Dataset Descriptions

All sample datasets are in the `data/` directory.

### `factories.csv`

| Column | Type | Description |
|---|---|---|
| `factory_id` | str | Unique factory identifier (e.g. F1) |
| `factory_name` | str | Human-readable name |
| `location` | str | City / state |
| `production_capacity` | int | Max units the factory can produce |
| `production_cost_per_unit` | float | *(Optional)* Cost to produce one unit |

### `destinations.csv`

| Column | Type | Description |
|---|---|---|
| `destination_id` | str | Unique DC identifier (e.g. D1) |
| `destination_name` | str | Human-readable name |
| `location` | str | City / state |
| `demand` | int | Units required by this DC |

### `shipping_costs.csv`

| Column | Type | Description |
|---|---|---|
| `factory_id` | str | References `factories.csv` |
| `destination_id` | str | References `destinations.csv` |
| `shipping_cost_per_unit` | float | Cost to ship one unit on this route |
| `distance_miles` | float | *(Optional)* Route distance |

> Every `(factory_id, destination_id)` pair must appear exactly once.

---

## Mathematical Formulation

### Decision Variable

$$x_{ij} = \text{quantity shipped from factory } i \text{ to distribution center } j$$

### Objective — Minimise Total Transportation Cost

$$\min \; Z = \sum_{i} \sum_{j} c_{ij} \cdot x_{ij}$$

where $c_{ij}$ is the shipping cost per unit from factory $i$ to DC $j$.

### Extended Objective (optional, includes production cost)

$$\min \; Z = \sum_{i} \sum_{j} (c_{ij} + p_i) \cdot x_{ij}$$

where $p_i$ is the production cost per unit at factory $i$.

### Constraints

1. **Demand satisfaction** (equality):
$$\sum_{i} x_{ij} = d_j \quad \forall \; j$$

2. **Factory capacity** (inequality):
$$\sum_{j} x_{ij} \le C_i \quad \forall \; i$$

3. **Non-negativity**:
$$x_{ij} \ge 0 \quad \forall \; i, j$$

---

## Installation

```bash
# 1. Clone or download the project
cd "Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor"

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### Quick Start

1. Open the **📊 Overview** page to see KPIs and capacity charts.
2. Go to **📁 Data Management** to inspect or upload your own CSV files.
3. Click **🚀 Optimize Shipping** on the **⚙️ Optimization** page.
4. Explore charts on **📈 Visual Analytics**.
5. Read auto-generated insights on **💡 Recommendations**.
6. Download results from **📥 Export Results**.

---

## Uploading Custom Data

1. Navigate to **📁 Data Management → Upload CSV**.
2. Upload one, two, or all three CSV files (unmodified files keep the current data).
3. Click **✅ Validate & Load** — the system checks column names, data types, non-negativity, uniqueness, and cross-references.
4. Fix any reported errors and re-upload.
5. Run the optimiser again with your new data.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite validates:
- Demand is satisfied for every DC.
- Factory capacity is never exceeded.
- All shipments are non-negative.
- Total cost equals the sum of route costs.
- Optimal cost ≤ baseline cost.
- Extended (production + shipping) model is feasible.
- Infeasible inputs are detected and explained.
- Missing columns, negative values, duplicates, and null values are rejected.
- End-to-end pipeline with sample CSV files.

---

## Project Structure

```
nassau-candy-optimization/
├── app.py                 # Streamlit dashboard (6 pages)
├── optimizer.py           # LP solver + baseline heuristic
├── data_loader.py         # CSV loading & validation
├── recommendations.py     # Auto recommendation generator
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   ├── factories.csv      # Sample factory data
│   ├── destinations.csv   # Sample DC data
│   └── shipping_costs.csv # Sample shipping cost matrix
└── tests/
    └── test_optimizer.py  # Pytest test suite
```

---

## Limitations & Future Improvements

### Current Limitations
- Single-period, static model (no time-series or seasonal demand).
- Transportation problem only — no multi-echelon or inventory considerations.
- Solver uses default CBC; very large problems may benefit from commercial solvers.
- No geographic map visualisation (Sankey diagram is used instead).

### Future Improvements
- **Multi-period optimisation** with inventory carry-over.
- **Scenario analysis** — what-if sliders for demand/capacity changes.
- **Geographic mapping** with Folium or Mapbox.
- **Integer programming** for truckload / pallet constraints.
- **API integration** for real freight-rate data.
- **Authentication and multi-user support** for enterprise deployment.
- **Database backend** (PostgreSQL, BigQuery) instead of CSV files.

---

## License

This project is created for educational and internship demonstration purposes.

---

*Built with ❤️ using Python, PuLP, Streamlit, and Plotly.*
