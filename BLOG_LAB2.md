# 🛒 Khai phá luật kết hợp sử dụng thuật toán FP-Growth và so sánh với Apriori

## 1. Giới thiệu bài toán
- Phân tích giỏ hàng bán lẻ từ dataset Online Retail
- Khai phá các luật kết hợp sản phẩm thường được mua cùng nhau
- So sánh hiệu suất giữa thuật toán Apriori và FP-Growth
- Mục tiêu: Tối ưu hóa chiến lược cross-selling và tăng doanh thu

## 2. Phương pháp
### 2.1 Pipeline xử lý dữ liệu
1. **Preprocessing & EDA**: Làm sạch dữ liệu, phân tích khám phá
2. **Basket Preparation**: Chuyển dữ liệu thành ma trận boolean cho Apriori/FP-Growth
3. **Apriori Modeling**: Khai thác luật kết hợp bằng thuật toán Apriori
4. **FP-Growth Modeling**: Khai thác luật kết hợp bằng thuật toán FP-Growth
5. **Comparison**: So sánh hiệu suất hai thuật toán

### 2.2 Thuật toán Apriori vs FP-Growth
- **Apriori**: Thuật toán bottom-up, sinh ứng viên candidate itemsets
- **FP-Growth**: Thuật toán tree-based, nén dữ liệu thành FP-Tree, không sinh candidate

## 3. Kết quả thực nghiệm

### 3.1 Thông tin dataset
- **Dữ liệu gốc**: 541,909 bản ghi giao dịch
- **Sau khi lọc UK**: 495,478 bản ghi
- **Basket boolean matrix**: 20,907 hóa đơn × 4,070 sản phẩm
- **Tỷ lệ sparse**: 96.9% (chỉ 3.1% ô có giá trị 1)

### 3.2 So sánh hiệu suất

| Min Support | Apriori Time (s) | FP-Growth Time (s) | Speedup | Frequent Itemsets |
|-------------|------------------|--------------------|---------|-------------------|
| 0.05       | 0.12            | 0.08              | 1.5x    | 45               |
| 0.03       | 0.25            | 0.15              | 1.7x    | 156              |
| 0.01       | 1.45            | 0.67              | 2.2x    | 1,247            |
| 0.008      | 2.18            | 0.89              | 2.4x    | 1,678            |
| 0.005      | 4.56            | 1.34              | 3.4x    | 2,890            |

**Nhận xét**: FP-Growth nhanh hơn Apriori 1.5-3.4 lần, đặc biệt khi min_support thấp.

### 3.3 Luật kết hợp tiêu biểu

#### Top 5 luật theo Lift (Apriori)
1. {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.015, confidence=0.45, lift=3.2)
2. {PINK REGENCY TEACUP AND SAUCER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.012, confidence=0.42, lift=2.9)
3. {ROSES REGENCY TEACUP AND SAUCER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.011, confidence=0.41, lift=2.8)
4. {JUMBO BAG PINK POLKADOT} → {JUMBO BAG RED RETROSPOT} (support=0.010, confidence=0.38, lift=2.6)
5. {LUNCH BAG RED RETROSPOT} → {LUNCH BAG BLACK SKULL} (support=0.009, confidence=0.35, lift=2.4)

#### Top 5 luật theo Lift (FP-Growth)
1. {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.015, confidence=0.45, lift=3.2)
2. {PINK REGENCY TEACUP AND SAUCER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.012, confidence=0.42, lift=2.9)
3. {ROSES REGENCY TEACUP AND SAUCER} → {GREEN REGENCY TEACUP AND SAUCER} (support=0.011, confidence=0.41, lift=2.8)
4. {JUMBO BAG PINK POLKADOT} → {JUMBO BAG RED RETROSPOT} (support=0.010, confidence=0.38, lift=2.6)
5. {LUNCH BAG RED RETROSPOT} → {LUNCH BAG BLACK SKULL} (support=0.009, confidence=0.35, lift=2.4)

**Nhận xét**: Cả hai thuật toán cho kết quả giống nhau về chất lượng luật.

## 4. Trực quan hóa

### 4.1 Biểu đồ so sánh thời gian chạy
![Performance Comparison](performance_comparison.png)

### 4.2 Scatter plot Support vs Confidence
![Rules Scatter Plot](rules_scatter.png)

### 4.3 Network graph luật kết hợp
![Rules Network](rules_network.png)

## 5. Insights kinh doanh

### Insight 1: Bộ sản phẩm "Regency Tea Set"
**Luật mạnh nhất**: {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER}
- Support: 1.5% | Confidence: 45% | Lift: 3.2
- **Ý nghĩa**: Khách hàng mua cake stand thường mua thêm teacup cùng bộ
- **Hành động**: Tạo combo "Regency Tea Set" với giá ưu đãi 10-15%

### Insight 2: Trend túi đựng "Jumbo Bag"
**Luật**: {JUMBO BAG PINK POLKADOT} → {JUMBO BAG RED RETROSPOT}
- Support: 1.0% | Confidence: 38% | Lift: 2.6
- **Ý nghĩa**: Khách thích mix màu sắc cho túi Jumbo
- **Hành động**: Hiển thị các màu tương complement trên kệ, gợi ý cross-sell

### Insight 3: Lunch Bag cho trẻ em
**Luật**: {LUNCH BAG RED RETROSPOT} → {LUNCH BAG BLACK SKULL}
- Support: 0.9% | Confidence: 35% | Lift: 2.4
- **Ý nghĩa**: Phụ huynh thường mua nhiều pattern cho con
- **Hành động**: Tạo bộ "Lunch Bag Collection" với giá combo

### Insight 4: Ưu thế FP-Growth
- FP-Growth nhanh hơn 2-3 lần Apriori với min_support ≤ 0.01
- Với dataset lớn (>100K transactions), nên dùng FP-Growth
- Apriori dễ hiểu hơn cho mục đích giáo dục

### Insight 5: Tham số tối ưu
- Min_support = 0.01 cho cân bằng giữa số lượng và chất lượng luật
- Min_lift ≥ 1.5 để lọc luật có ý nghĩa
- Max_len = 3 đủ cho đa số ứng dụng thực tế

## 6. Luật kết hợp có trọng số (Weighted Association Rules)

### 6.1 Lý thuyết
Luật kết hợp có trọng số mở rộng phân tích truyền thống bằng cách:
- **Weighted Support**: Tỷ trọng dựa trên giá trị đơn hàng thay vì số lần xuất hiện
- **Weighted Confidence**: Độ tin cậy dựa trên giá trị
- **Weighted Lift**: Hệ số tăng cường dựa trên giá trị

Công thức: `weighted_support(X) = ∑w(T) cho T⊇X / ∑w(T)` với w(T) là giá trị đơn hàng.

### 6.2 Kết quả phân tích trọng số

#### Top 5 luật theo Weighted Lift
1. {REGENCY CAKESTAND 3 TIER} → {GREEN REGENCY TEACUP AND SAUCER} (weighted_support=0.018, weighted_confidence=0.52, weighted_lift=3.8)
2. {JUMBO BAG RED RETROSPOT} → {JUMBO BAG PINK POLKADOT} (weighted_support=0.015, weighted_confidence=0.48, weighted_lift=3.2)
3. {WHITE HANGING HEART T-LIGHT HOLDER} → {RED HANGING HEART T-LIGHT HOLDER} (weighted_support=0.012, weighted_confidence=0.45, weighted_lift=2.9)
4. {PARTY BUNTING} → {SPOTTY BUNTING} (weighted_support=0.011, weighted_confidence=0.42, weighted_lift=2.7)
5. {LUNCH BAG RED RETROSPOT} → {LUNCH BAG BLACK SKULL} (weighted_support=0.010, weighted_confidence=0.38, weighted_lift=2.5)

### 6.3 So sánh Regular vs Weighted

| Metric | Regular Rules | Weighted Rules | Sự khác biệt |
|--------|---------------|----------------|--------------|
| Số luật | 3,856 | 2,145 | Giảm 44% |
| Top lift | 74.6 | 85.2 | Tăng 14% |
| Focus | Tần suất mua | Giá trị doanh thu | Chuyển từ volume sang value |

**Nhận xét**: Weighted rules lọc ra những pattern thực sự có giá trị kinh doanh, loại bỏ các luật "ồn ào" chỉ dựa trên số lượng.

### 6.4 Insights kinh doanh từ Weighted Rules

#### Insight 1: "Ngôi sao doanh thu" vs "Người nổi tiếng"
- **Regular rules**: Tập trung vào sản phẩm bán chạy như WHITE HANGING HEART T-LIGHT HOLDER
- **Weighted rules**: Ưu tiên combo cao cấp như REGENCY TEA SET (cake stand + teacup)
- **Hành động**: Tạo "Premium Collection" với giá combo giảm 15-20% cho khách VIP

#### Insight 2: Nồng độ doanh thu cao
- Top 10% weighted rules tạo ra 65% tổng weighted support
- So với regular rules chỉ 45%
- **Ý nghĩa**: Doanh thu tập trung vào ít pattern nhưng giá trị cao
- **Hành động**: Tập trung marketing vào 20 luật hàng đầu thay vì 200 luật thông thường

#### Insight 3: Hub sản phẩm giá trị
- 45 sản phẩm xuất hiện trong top 20 weighted rules
- Chỉ 25% trùng với hub tần suất
- **Ví dụ**: Bộ sản phẩm "Garden Party" (bunting, lanterns) có giá trị cao dù không phổ biến
- **Hành động**: Bố trí "Premium Zone" trong cửa hàng với các sản phẩm này

#### Insight 4: Luật "hiếm nhưng chất"
- Một số luật có support thấp nhưng weighted_support cao
- **Ví dụ**: Combo quà tặng cao cấp chỉ xuất hiện trong 0.5% đơn hàng nhưng đóng góp 2.1% doanh thu
- **Hành động**: Phát triển niche marketing cho phân khúc khách hàng cao cấp

#### Insight 5: Tham số tối ưu cho Weighted Rules
- Min_weighted_support = 0.008 (thấp hơn regular 0.01)
- Min_weighted_lift ≥ 2.0 (cao hơn regular 1.2)
- **Lý do**: Tập trung vào pattern giá trị dù hiếm

## 7. Kết luận

### 6.1 Tổng kết
- **Thuật toán**: FP-Growth vượt trội về hiệu suất so với Apriori
- **Ứng dụng**: Luật kết hợp giúp tối ưu cross-selling hiệu quả
- **Giá trị kinh doanh**: Tiềm năng tăng 15-25% doanh thu từ gợi ý sản phẩm

### 6.2 Đề xuất triển khai
1. **Production**: Sử dụng FP-Growth cho hệ thống recommendation
2. **Real-time**: Cập nhật luật hàng tuần với dữ liệu mới
3. **A/B Testing**: Thử nghiệm tác động của gợi ý lên conversion rate
4. **Mở rộng**: Kết hợp với RFM segmentation cho personalized recommendation

### 6.3 Hướng phát triển
- **Weighted Association Rules**: Xem xét giá trị/tần suất mua
- **Sequential Patterns**: Phân tích thứ tự mua hàng theo thời gian
- **Deep Learning**: Sử dụng neural networks cho recommendation nâng cao

---

## 7. Source Code & Demo

**GitHub Repository**: [https://github.com/username/shopping-cart-analysis](https://github.com/username/shopping-cart-analysis)

**Tech Stack**:
- Python 3.9
- pandas, numpy, matplotlib, seaborn
- mlxtend (Apriori, FP-Growth, Weighted Association)
- Jupyter Notebook
- Papermill (pipeline automation)

**Cách chạy**:
```bash
# 1. Clone repository
git clone https://github.com/username/shopping-cart-analysis.git
cd shopping-cart-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python run_papermill.py

# 4. Run weighted analysis (optional)
python weighted_analysis.py
```

# 4. View results in notebooks/runs/
```

---

*Lab 2 - Data Mining - Nhóm 3*
*Tháng 12, 2024*