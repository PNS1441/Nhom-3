#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Association Rule Analysis Script
Phân tích luật kết hợp có trọng số dựa trên giá trị đơn hàng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
project_root = Path.cwd()
src_path = project_root / "src"
sys.path.append(str(src_path))

from apriori_library import WeightedAssociationMiner

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Parameters
MIN_WEIGHTED_SUPPORT = 0.01
MIN_WEIGHTED_CONFIDENCE = 0.3
MIN_WEIGHTED_LIFT = 1.2
TOP_N_RULES = 20

# Setup paths
data_dir = project_root / "data"
processed_dir = data_dir / "processed"
basket_path = processed_dir / "basket_bool.parquet"
cleaned_data_path = processed_dir / "cleaned_uk_data.csv"

print("=== Weighted Association Rule Analysis ===")
print(f"Project root: {project_root}")

# Load cleaned data to get invoice values
print("\n=== Loading cleaned data ===")
df_cleaned = pd.read_csv(cleaned_data_path)
print(f"Loaded {len(df_cleaned):,} transactions")

# Calculate invoice values (weights)
print("\n=== Calculating invoice values ===")
df_cleaned['InvoiceValue'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
invoice_weights = df_cleaned.groupby('InvoiceNo')['InvoiceValue'].sum()
print(f"Calculated weights for {len(invoice_weights):,} invoices")
print(f"Total value: ${invoice_weights.sum():,.2f}")
print(f"Average invoice value: ${invoice_weights.mean():.2f}")

# Load basket matrix
print("\n=== Loading basket matrix ===")
basket_bool = pd.read_parquet(basket_path)
print(f"Basket matrix: {basket_bool.shape[0]:,} transactions × {basket_bool.shape[1]:,} products")

# Align weights with basket transactions
common_invoices = basket_bool.index.intersection(invoice_weights.index)
basket_bool = basket_bool.loc[common_invoices]
transaction_weights = invoice_weights.loc[common_invoices]

print(f"After alignment: {len(basket_bool):,} transactions with weights")

# Initialize WeightedAssociationMiner
print("\n=== Initializing Weighted Association Miner ===")
weighted_miner = WeightedAssociationMiner(basket_bool, transaction_weights)

# Mine weighted frequent itemsets
print("\n=== Mining Weighted Frequent Itemsets ===")
weighted_itemsets = weighted_miner.mine_frequent_itemsets_weighted(
    min_weighted_support=MIN_WEIGHTED_SUPPORT,
    max_len=3
)
print(f"Found {len(weighted_itemsets):,} weighted frequent itemsets")
print("\nTop 10 weighted frequent itemsets:")
print(weighted_itemsets.head(10))

# Generate weighted rules
print("\n=== Generating Weighted Association Rules ===")
weighted_rules = weighted_miner.generate_weighted_rules(
    frequent_itemsets=weighted_itemsets,
    min_weighted_confidence=MIN_WEIGHTED_CONFIDENCE,
    min_weighted_lift=MIN_WEIGHTED_LIFT
)
print(f"Generated {len(weighted_rules):,} weighted rules")

# Add readable rule strings
weighted_rules = weighted_miner.add_readable_rule_str(weighted_rules)

print("\nTop 10 weighted rules by lift:")
print(weighted_rules[['rule_str', 'weighted_support', 'weighted_confidence', 'weighted_lift']].head(10))

# Compare with regular rules
print("\n=== Comparing with Regular Rules ===")
regular_rules_path = processed_dir / "rules_fpgrowth_filtered.csv"
if regular_rules_path.exists():
    regular_rules = pd.read_csv(regular_rules_path)
    print(f"Loaded {len(regular_rules):,} regular rules")

    # Compare top rules
    print("\nTop 5 regular rules:")
    print(regular_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

    print("\nTop 5 weighted rules:")
    print(weighted_rules[['antecedents', 'consequents', 'weighted_support', 'weighted_confidence', 'weighted_lift']].head())
else:
    print("Regular rules file not found")

# Visualizations
print("\n=== Creating Visualizations ===")

# Plot top weighted rules
plt.figure(figsize=(12, 8))
top_rules = weighted_rules.head(TOP_N_RULES)
bars = plt.barh(range(len(top_rules)), top_rules['weighted_lift'])
plt.yticks(range(len(top_rules)), [str(rule)[:50] + '...' if len(str(rule)) > 50 else str(rule) for rule in top_rules['rule_str']])
plt.xlabel('Weighted Lift')
plt.title('Top Rules by Weighted Lift')
plt.tight_layout()
plt.savefig(processed_dir / "weighted_rules_top_lift.png", dpi=150, bbox_inches='tight')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    weighted_rules['weighted_support'],
    weighted_rules['weighted_confidence'],
    c=weighted_rules['weighted_lift'],
    alpha=0.6,
    cmap='viridis'
)
plt.colorbar(scatter, label='Weighted Lift')
plt.xlabel('Weighted Support')
plt.ylabel('Weighted Confidence')
plt.title('Weighted Rules: Support vs Confidence (Color = Lift)')
plt.tight_layout()
plt.savefig(processed_dir / "weighted_rules_scatter.png", dpi=150, bbox_inches='tight')
plt.show()

# Comparison plot
if 'regular_rules' in locals() and len(regular_rules) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Support comparison
    min_len = min(len(regular_rules), len(weighted_rules))
    axes[0].scatter(regular_rules['support'][:min_len], weighted_rules['weighted_support'][:min_len], alpha=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.7)
    axes[0].set_xlabel('Regular Support')
    axes[0].set_ylabel('Weighted Support')
    axes[0].set_title('Support: Regular vs Weighted')
    axes[0].grid(True, alpha=0.3)

    # Lift comparison
    axes[1].scatter(regular_rules['lift'][:min_len], weighted_rules['weighted_lift'][:min_len], alpha=0.5)
    max_val = max(regular_rules['lift'][:min_len].max(), weighted_rules['weighted_lift'][:min_len].max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    axes[1].set_xlabel('Regular Lift')
    axes[1].set_ylabel('Weighted Lift')
    axes[1].set_title('Lift: Regular vs Weighted')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(processed_dir / "weighted_vs_regular_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

# Business Insights
print("\n=== Business Insights from Weighted Rules ===")

# Insight 1: High-value vs High-frequency products
high_value_rules = weighted_rules[weighted_rules['weighted_support'] > weighted_rules['weighted_support'].quantile(0.75)]
print(f"1. High-value rules (weighted_support > Q3): {len(high_value_rules)} rules")
print("   These represent premium product combinations that drive revenue despite lower frequency")

# Insight 2: Revenue concentration
total_weighted_support = weighted_rules['weighted_support'].sum()
top_10_percent = weighted_rules.head(int(len(weighted_rules)*0.1))['weighted_support'].sum()
concentration_ratio = top_10_percent / total_weighted_support
print(f"2. Revenue concentration: Top 10% rules account for {concentration_ratio:.1%} of weighted support")
print("   Indicates high concentration of value in few key rules")
premium_products = set()
for _, rule in weighted_rules.head(20).iterrows():
    premium_products.update(rule['antecedents'])
    premium_products.update(rule['consequents'])
print(f"3. Premium product hub: {len(premium_products)} products appear in top 20 weighted rules")
print("   Focus marketing efforts on these high-value items")

# Save results
output_path = processed_dir / "weighted_rules_fpgrowth.csv"
weighted_rules.to_csv(output_path, index=False)
print(f"\nSaved weighted rules to: {output_path}")

print("\n=== Weighted Association Analysis Complete ===")