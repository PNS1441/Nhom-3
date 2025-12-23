# 📚 HƯỚNG DẪN THỰC HIỆN LAB 2: FP-GROWTH

## 📋 TÓM TẮT YÊU CẦU

### Mục tiêu chính:
1. **Triển khai thuật toán FP-Growth** để khai phá luật kết hợp
2. **So sánh Apriori vs FP-Growth** về:
   - Thời gian chạy
   - Số lượng tập phổ biến/luật
   - Độ nhạy với tham số (min_support)
3. **Trực quan hóa kết quả** (tối thiểu 2 biểu đồ)
4. **Phân tích insight kinh doanh** (tối thiểu 5 insights)
5. **Viết blog/report** và **trình bày 5-7 phút**

---

## 🎯 CHECKLIST THỰC HIỆN

### ✅ PHẦN 1: CÀI ĐẶT MÔI TRƯỜNG (5 phút)
- [ ] Kích hoạt môi trường: `conda activate shopping_env`
- [ ] Kiểm tra thư viện đã cài: `pip list | grep mlxtend`
- [ ] Đảm bảo có đủ dữ liệu từ Lab 1

### ✅ PHẦN 2: TRIỂN KHAI FP-GROWTH (Q1)

#### Bước 1: Cập nhật `src/apriori_library.py`
Thêm class `FPGrowthMiner` mới:

```python
class FPGrowthMiner:
    """
    A class for mining association rules using the FP-Growth algorithm.
    """
    
    def __init__(self, basket_bool: pd.DataFrame):
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None
    
    def mine_frequent_itemsets(
        self,
        min_support: float = 0.01,
        max_len: int = None,
        use_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using FP-Growth algorithm.
        """
        from mlxtend.frequent_patterns import fpgrowth
        
        fi = fpgrowth(
            self.basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )
        fi.sort_values(by="support", ascending=False, inplace=True)
        self.frequent_itemsets = fi
        return self.frequent_itemsets
    
    def generate_rules(
        self,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Generate association rules from frequent itemsets."""
        if self.frequent_itemsets is None:
            raise ValueError("Frequent itemsets not mined.")
        
        rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules
        return self.rules
    
    # Sử dụng lại các method từ AssociationRulesMiner:
    # - add_readable_rule_str()
    # - filter_rules()
    # - save_rules()
```

**Gợi ý:** Copy các method `add_readable_rule_str()`, `filter_rules()`, `save_rules()` từ `AssociationRulesMiner` sang `FPGrowthMiner` để tái sử dụng.

#### Bước 2: Tạo notebook `fp_growth_modelling.ipynb`

Tạo file mới `notebooks/fp_growth_modelling.ipynb` với cấu trúc tương tự `apriori_modelling.ipynb`:

**Cell 1 - Markdown: Tiêu đề và giới thiệu**
```markdown
# Bước 4: FP-Growth Modeling for Association Rules

Notebook này sử dụng thuật toán **FP-Growth** để:
- Khai thác tập mục phổ biến nhanh hơn Apriori
- Sinh luật kết hợp với các chỉ số: support, confidence, lift
- So sánh hiệu suất với Apriori
```

**Cell 2 - Parameters:**
```python
# PARAMETERS (for papermill)
BASKET_BOOL_PATH = "data/processed/basket_bool.parquet"
RULES_OUTPUT_PATH = "data/processed/rules_fpgrowth_filtered.csv"

MIN_SUPPORT = 0.01
MAX_LEN = 3
METRIC = "lift"
MIN_THRESHOLD = 1.0

FILTER_MIN_SUPPORT = 0.01
FILTER_MIN_CONF = 0.3
FILTER_MIN_LIFT = 1.2
FILTER_MAX_ANTECEDENTS = 2
FILTER_MAX_CONSEQUENTS = 1

TOP_N_RULES = 20
PLOT_TOP_LIFT = True
PLOT_TOP_CONF = True
PLOT_SCATTER = True
PLOT_NETWORK = True
PLOT_PLOTLY_SCATTER = True
```

**Cell 3 - Setup:**
```python
%load_ext autoreload
%autoreload 2

import os
import sys
import time
import pandas as pd

# Setup project path
cwd = os.getcwd()
if os.path.basename(cwd) == "notebooks":
    project_root = os.path.abspath("..")
else:
    project_root = cwd

src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from apriori_library import FPGrowthMiner, DataVisualizer
```

**Cell 4 - Load data:**
```python
# Tải basket_bool
basket_bool = pd.read_parquet(BASKET_BOOL_PATH)

print("=== Thông tin basket_bool ===")
print(f"- Số hoá đơn (rows): {basket_bool.shape[0]:,}")
print(f"- Số sản phẩm (columns): {basket_bool.shape[1]:,}")
print(f"- Tỷ lệ ô = 1 (có mua): {basket_bool.values.mean():.4f}")

basket_bool.head()
```

**Cell 5 - Mine frequent itemsets:**
```python
# Khởi tạo FP-Growth miner
miner = FPGrowthMiner(basket_bool=basket_bool)

start_time = time.time()
frequent_itemsets_fp = miner.mine_frequent_itemsets(
    min_support=MIN_SUPPORT,
    max_len=MAX_LEN,
    use_colnames=True,
)
elapsed_time = time.time() - start_time

print("=== Kết quả khai thác tập mục phổ biến (FP-Growth) ===")
print(f"- Thời gian chạy: {elapsed_time:.2f} giây")
print(f"- Số tập mục phổ biến thu được: {len(frequent_itemsets_fp):,}")

frequent_itemsets_fp.head(10)
```

**Cell 6-10:** Tương tự apriori_modelling.ipynb:
- Generate rules
- Filter rules
- Visualizations (top lift, top confidence, scatter, network)
- Save results

### ✅ PHẦN 3: SO SÁNH APRIORI VS FP-GROWTH (Q2)

Tạo notebook `notebooks/compare_apriori_fpgrowth.ipynb`:

```python
# Cell 1: Load cả 2 bộ rules
rules_apriori = pd.read_csv("data/processed/rules_apriori_filtered.csv")
rules_fpgrowth = pd.read_csv("data/processed/rules_fpgrowth_filtered.csv")

print(f"Apriori: {len(rules_apriori)} rules")
print(f"FP-Growth: {len(rules_fpgrowth)} rules")
```

```python
# Cell 2: So sánh thời gian chạy với nhiều giá trị min_support
import time
import matplotlib.pyplot as plt

support_values = [0.05, 0.03, 0.01, 0.008, 0.005]
time_apriori = []
time_fpgrowth = []
count_apriori = []
count_fpgrowth = []

for sup in support_values:
    # Test Apriori
    start = time.time()
    miner_ap = AssociationRulesMiner(basket_bool)
    fi_ap = miner_ap.mine_frequent_itemsets(min_support=sup, max_len=3)
    time_apriori.append(time.time() - start)
    count_apriori.append(len(fi_ap))
    
    # Test FP-Growth
    start = time.time()
    miner_fp = FPGrowthMiner(basket_bool)
    fi_fp = miner_fp.mine_frequent_itemsets(min_support=sup, max_len=3)
    time_fpgrowth.append(time.time() - start)
    count_fpgrowth.append(len(fi_fp))

# Vẽ biểu đồ so sánh
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ thời gian
ax1.plot(support_values, time_apriori, marker='o', label='Apriori')
ax1.plot(support_values, time_fpgrowth, marker='s', label='FP-Growth')
ax1.set_xlabel('Min Support')
ax1.set_ylabel('Thời gian (giây)')
ax1.set_title('So sánh thời gian chạy')
ax1.legend()
ax1.invert_xaxis()

# Biểu đồ số lượng itemsets
ax2.plot(support_values, count_apriori, marker='o', label='Apriori')
ax2.plot(support_values, count_fpgrowth, marker='s', label='FP-Growth')
ax2.set_xlabel('Min Support')
ax2.set_ylabel('Số frequent itemsets')
ax2.set_title('So sánh số lượng tập phổ biến')
ax2.legend()
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
```

```python
# Cell 3: Bảng tổng hợp so sánh
comparison_df = pd.DataFrame({
    'Min Support': support_values,
    'Apriori Time (s)': time_apriori,
    'FP-Growth Time (s)': time_fpgrowth,
    'Apriori Count': count_apriori,
    'FP-Growth Count': count_fpgrowth,
    'Speedup': [a/f if f > 0 else 0 for a, f in zip(time_apriori, time_fpgrowth)]
})

comparison_df
```

### ✅ PHẦN 4: TRỰC QUAN HÓA (Tối thiểu 2 biểu đồ)

**Biểu đồ 1: Scatter plot so sánh Support vs Confidence**
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Apriori
scatter1 = ax1.scatter(
    rules_apriori['support'],
    rules_apriori['confidence'],
    c=rules_apriori['lift'],
    s=50, alpha=0.6, cmap='viridis'
)
ax1.set_xlabel('Support')
ax1.set_ylabel('Confidence')
ax1.set_title('Apriori: Support vs Confidence (màu = Lift)')
plt.colorbar(scatter1, ax=ax1, label='Lift')

# FP-Growth
scatter2 = ax2.scatter(
    rules_fpgrowth['support'],
    rules_fpgrowth['confidence'],
    c=rules_fpgrowth['lift'],
    s=50, alpha=0.6, cmap='plasma'
)
ax2.set_xlabel('Support')
ax2.set_ylabel('Confidence')
ax2.set_title('FP-Growth: Support vs Confidence (màu = Lift)')
plt.colorbar(scatter2, ax=ax2, label='Lift')

plt.tight_layout()
plt.show()
```

**Biểu đồ 2: Network Graph**
- Sử dụng hàm `plot_rules_network()` đã có sẵn trong `DataVisualizer`

### ✅ PHẦN 5: PHÂN TÍCH INSIGHT KINH DOANH (Tối thiểu 5 insights)

Trong notebook phân tích, thêm cell markdown:

```markdown
## 📊 Insights Kinh Doanh

### Insight 1: Combo sản phẩm có Lift cao
**Luật:** {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER}
- Support: 0.015 | Confidence: 0.45 | Lift: 3.2
- **Hành động:** Tạo combo "Bộ trà chiều Regency" với giá ưu đãi 10%

### Insight 2: Sản phẩm bán chạy (Hub)
{WHITE HANGING HEART T-LIGHT HOLDER} xuất hiện trong 65 luật
- **Hành động:** Đặt ở vị trí dễ thấy, dùng làm sản phẩm "mồi" cho cross-selling

### Insight 3: So sánh Apriori vs FP-Growth
- FP-Growth nhanh hơn 5-10 lần khi min_support < 0.01
- Cùng tham số → cùng kết quả luật → chọn FP-Growth cho dữ liệu lớn

### Insight 4: Độ nhạy tham số
- Giảm min_support từ 0.01 → 0.005: số luật tăng gấp 3 lần
- Nhiều luật "nhiễu" → nên giữ min_support ≥ 0.01, tập trung vào min_lift ≥ 1.5

### Insight 5: Mùa vụ và giá trị
- Tháng 11-12: luật liên quan "Christmas", "Gift" có support cao
- **Hành động:** Chuẩn bị stock sớm, tạo banner gợi ý theo mùa
```

### ✅ PHẦN 6: CẬP NHẬT PIPELINE

Cập nhật `run_papermill.py`:

```python
# Thêm vào danh sách notebooks
notebooks = [
    {
        "input": "notebooks/preprocessing_and_eda.ipynb",
        "output": "notebooks/runs/preprocessing_and_eda_run.ipynb",
    },
    {
        "input": "notebooks/basket_preparation.ipynb",
        "output": "notebooks/runs/basket_preparation_run.ipynb",
    },
    {
        "input": "notebooks/apriori_modelling.ipynb",
        "output": "notebooks/runs/apriori_modelling_run.ipynb",
    },
    # MỚI: FP-Growth
    {
        "input": "notebooks/fp_growth_modelling.ipynb",
        "output": "notebooks/runs/fp_growth_modelling_run.ipynb",
    },
    # MỚI: So sánh
    {
        "input": "notebooks/compare_apriori_fpgrowth.ipynb",
        "output": "notebooks/runs/compare_apriori_fpgrowth_run.ipynb",
    },
]
```

### ✅ PHẦN 7: VIẾT BLOG/REPORT

Tạo file `BLOG_LAB2.md` với cấu trúc:

```markdown
# 🛒 Khai phá luật kết hợp: So sánh Apriori vs FP-Growth

## 1. Giới thiệu bài toán
- Phân tích giỏ hàng bán lẻ
- Tìm sản phẩm thường mua cùng nhau
- Mục tiêu: tối ưu doanh thu qua cross-selling

## 2. Phương pháp
### 2.1 Pipeline
[Hình ảnh pipeline]

### 2.2 Apriori vs FP-Growth
- Apriori: bottom-up, sinh ứng viên
- FP-Growth: tree-based, nén dữ liệu

## 3. Kết quả
### 3.1 So sánh hiệu suất
[Biểu đồ thời gian chạy]

### 3.2 Luật nổi bật
[Bảng top 10 rules]

## 4. Insights kinh doanh
[5 insights đã phân tích]

## 5. Kết luận
- FP-Growth ưu việt về tốc độ
- Apriori đơn giản hơn để hiểu
- Đề xuất: dùng FP-Growth cho production

## 6. Demo & Source Code
- GitHub: [link]
- Live demo: [link nếu có]
```

### ✅ PHẦN 8: CHUẨN BỊ TRÌNH BÀY (5-7 phút)

**Slide outline:**
1. **Slide 1:** Tiêu đề + Thành viên nhóm
2. **Slide 2:** Bài toán và dữ liệu
3. **Slide 3:** Pipeline Lab 2 (hình minh họa)
4. **Slide 4:** Apriori vs FP-Growth (bảng so sánh)
5. **Slide 5:** Kết quả - Biểu đồ thời gian chạy
6. **Slide 6:** Top luật có Lift cao
7. **Slide 7:** 3 insights kinh doanh quan trọng nhất
8. **Slide 8:** Kết luận + Q&A

**Lưu ý khi trình bày:**
- ❌ KHÔNG đọc code
- ❌ KHÔNG giải thích thuật toán chi tiết
- ✅ TẬP TRUNG vào insight và giá trị thực tế
- ✅ Dùng Feynman style: giải thích đơn giản như dạy bạn bè

---

## 🚀 THỨ TỰ THỰC HIỆN ĐỀ NGHỊ

### Tuần 1 (Cơ bản - Hoàn thành Q1 & Q2):
1. ✅ Ngày 1-2: Cập nhật `apriori_library.py` với `FPGrowthMiner`
2. ✅ Ngày 3-4: Tạo `fp_growth_modelling.ipynb` và chạy thử
3. ✅ Ngày 5-6: Tạo `compare_apriori_fpgrowth.ipynb` 
4. ✅ Ngày 7: Tạo 2+ biểu đồ và viết 5+ insights

### Tuần 2 (Nâng cao - Tùy chọn):
5. ⭐ Ngày 8-10: Triển khai Weighted Association Rules (nếu muốn điểm cao)
6. ⭐ Ngày 11-12: Chọn 1 trong 7 chủ đề mở rộng
7. 📝 Ngày 13-14: Viết blog/report hoàn chỉnh
8. 🎤 Ngày 15: Chuẩn bị slide và tập trình bày

---

## 📚 TÀI LIỆU THAM KHẢO

1. **mlxtend documentation:** https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
2. **Apriori vs FP-Growth:** https://towardsdatascience.com/apriori-vs-fp-growth-6f3e9c9b6eaf
3. **Weighted Association Rules:** Tham khảo paper gốc hoặc hỏi giảng viên

---

## 🆘 TROUBLESHOOTING

### Lỗi: ModuleNotFoundError: No module named 'mlxtend'
```bash
pip install mlxtend
```

### Lỗi: fpgrowth() takes too long
- Tăng min_support (0.01 → 0.02)
- Giảm max_len (3 → 2)
- Giảm số sản phẩm trong basket_bool

### Lỗi: FP-Growth và Apriori cho kết quả khác nhau
- Kiểm tra lại tham số min_support phải giống nhau
- Đảm bảo use_colnames=True cho cả 2
- Kiểm tra phiên bản mlxtend: `pip show mlxtend`

---

## ✨ TIPS ĐỂ LẤY ĐIỂM CAO

1. **So sánh chi tiết:** Không chỉ thời gian, mà còn bộ nhớ, độ dài itemset trung bình
2. **Visualizations đẹp:** Dùng Plotly interactive thay vì matplotlib tĩnh
3. **Insights sâu:** Kết hợp với RFM (nếu có từ Lab 1) để phân khúc khách hàng
4. **Code sạch:** Có docstring, comments rõ ràng
5. **Blog chuyên nghiệp:** Có mục lục, hình ảnh đẹp, link GitHub
6. **Trình bày tự tin:** Luyện tập trước, không đọc slide

---

**Chúc bạn thành công với Lab 2! 🎉**

Nếu cần hỗ trợ thêm về code chi tiết, hãy hỏi tôi.
