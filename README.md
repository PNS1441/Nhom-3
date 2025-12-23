# 🛒 Khai phá luật kết hợp sử dụng thuật toán FP-Growth và so sánh với Apriori

Phân tích dữ liệu bán lẻ để tìm ra mối quan hệ giữa các sản phẩm thường được mua cùng nhau bằng các kỹ thuật **Association Rule Mining** (Apriori & FP-Growth). Project triển khai pipeline đầy đủ từ xử lý dữ liệu → phân tích → khai thác luật → so sánh thuật toán → sinh báo cáo.

---

## Features

- Làm sạch dữ liệu & xử lý giá trị lỗi
- Xây dựng basket matrix (transaction × product)
- Khai phá tập mục phổ biến (Frequent itemsets) với **Apriori** & **FP-Growth**
- Sinh luật kết hợp (Association Rules)
- Các chỉ số:
  - Support
  - Confidence
  - Lift
- So sánh hiệu suất Apriori vs FP-Growth
- Visualization với:
  - bar chart
  - scatter plot
  - network graph
  - interactive Plotly
- Tự động hóa pipeline bằng **Papermill**

---

## Project Structure

```text
shopping_cart_analysis/
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   └── processed/
│       ├── cleaned_uk_data.csv
│       ├── basket_bool.parquet
│       ├── rules_apriori_filtered.csv
│       └── rules_fpgrowth_filtered.csv
│
├── notebooks/
│   ├── preprocessing_and_eda.ipynb
│   ├── basket_preparation.ipynb
│   ├── apriori_modelling.ipynb
│   ├── fp_growth_modelling.ipynb
│   ├── compare_apriori_fpgrowth.ipynb
│   └── runs/
│       ├── preprocessing_and_eda_run.ipynb
│       ├── basket_preparation_run.ipynb
│       ├── apriori_modelling_run.ipynb
│       ├── fp_growth_modelling_run.ipynb
│       └── compare_apriori_fpgrowth_run.ipynb
│
├── src/
│   └── apriori_library.py
│
├── run_papermill.py
├── weighted_analysis.py
├── BLOG_LAB2.md
└── README.md
```
├── requirements.txt
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone <your_repo_url>
cd shopping_cart_analysis
pip install -r requirements.txt
```

### Data Preparation
Đặt file gốc vào:
```bash
data/raw/online_retail.csv
```

### Run Pipeline (Recommended)
Chạy toàn bộ phân tích chỉ với 1 lệnh:

```bash
python run_papermill.py
```

Kết quả sinh ra:
```bash
data/processed/cleaned_uk_data.csv
data/processed/basket_bool.parquet
data/processed/rules_apriori_filtered.csv
data/processed/rules_fpgrowth_filtered.csv
notebooks/runs/*.ipynb
```

### Changing Parameters
Các tham số có thể chỉnh trong `run_papermill.py`:

```python
MIN_SUPPORT=0.01
MAX_LEN=3
FILTER_MIN_CONF=0.3
FILTER_MIN_LIFT=1.2
```

---

## Results & Insights

### Lab 1: Apriori Analysis
- **Dataset**: 20,907 transactions × 4,070 products
- **Frequent Itemsets**: 1,247 itemsets (min_support=0.01)
- **Association Rules**: 245 rules filtered (min_lift≥1.2)

**Top Rules**:
1. {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER} (lift=3.2)
2. {PINK REGENCY TEACUP AND SAUCER} → {GREEN REGENCY TEACUP AND SAUCER} (lift=2.9)
3. {JUMBO BAG PINK POLKADOT} → {JUMBO BAG RED RETROSPOT} (lift=2.6)

### Lab 2: FP-Growth vs Apriori + Weighted Association
- **Performance**: FP-Growth 2-3x faster than Apriori
- **Weighted Analysis**: Focus on high-value patterns vs high-frequency
- **Revenue Concentration**: Top 10% weighted rules account for 65% of value
- **Accuracy**: Both algorithms produce identical results

**Business Insights**:
- Premium product combinations (Regency Tea Set) drive higher revenue
- Revenue concentration in few high-value rules
- Value-based product hubs differ from frequency-based hubs
- Optimal parameters: min_weighted_support=0.008, min_weighted_lift≥2.0

---

## Visualization & Analysis

### Available Charts
- Top rules by Lift/Confidence (bar charts)
- Support vs Confidence scatter plots
- Product association network graphs
- Interactive Plotly dashboards

### Export Results
```bash
# Export notebook to HTML
jupyter nbconvert notebooks/runs/apriori_modelling_run.ipynb --to html

# Export to PDF
jupyter nbconvert notebooks/runs/apriori_modelling_run.ipynb --to pdf
```

---

## Business Applications

- **Product Recommendation**: Suggest complementary items
- **Cross-selling Strategy**: Create product bundles
- **Store Layout**: Optimize product placement
- **Inventory Management**: Stock related products together
- **Marketing Campaigns**: Target customers with bundle offers

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9 | Main language |
| Pandas | Transaction data processing |
| MLxtend | Apriori & FP-Growth algorithms |
| Papermill | Automated notebook execution |
| Matplotlib/Seaborn | Static visualizations |
| Plotly | Interactive dashboards |
| Jupyter | Notebook environment |
| PyArrow | Parquet file handling |

---

## Project Status

- ✅ **Lab 1**: Apriori implementation complete
- ✅ **Lab 2**: FP-Growth implementation & comparison complete
- 🔄 **Future**: Weighted rules, sequential patterns, Streamlit dashboard

---

*Data Mining Course - Group 3*


### Author
Project được thực hiện bởi:
Trang Le

📄 License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.
