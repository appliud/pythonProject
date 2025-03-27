# -*- coding: utf-8 -*-
"""波士顿房价预测完整代码"""
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

matplotlib.use('TkAgg')

# ----------------------------------
# 1. 数据加载（解决SSL验证问题）
# ----------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# 从OpenML加载数据集
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target.astype(float)  # 目标变量转换为float

print("特征维度:", X.shape)
print("\n前5行数据:\n", X.head())

# ----------------------------------
# 2. 探索性数据分析（EDA）
# ----------------------------------
plt.figure(figsize=(10, 6))

# 房价分布
plt.subplot(2, 2, 1)
sns.histplot(y, kde=True, color='blue')
plt.title('MEDV Distribution')

# 关键特征关系
plt.subplot(2, 2, 2)
sns.scatterplot(x=X['LSTAT'], y=y, alpha=0.6, color='green')
plt.title('LSTAT vs MEDV')

plt.subplot(2, 2, 3)
sns.scatterplot(x=X['RM'], y=y, alpha=0.6, color='red')
plt.title('RM vs MEDV')

# 相关性热力图
plt.subplot(2, 2, 4)
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

# ----------------------------------
# 3. 数据预处理
# ----------------------------------
# 处理异常值（以RM为例）
Q1 = X['RM'].quantile(0.25)
Q3 = X['RM'].quantile(0.75)
IQR = Q3 - Q1
X['RM'] = X['RM'].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 4. 建模与评估
# ----------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mse, r2])

    print(f"\n{name}:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")

# 结果对比
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])
print("\n模型对比:\n", results_df)

# ----------------------------------
# 5. 模型优化（随机森林调参）
# ----------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\n最佳参数:", grid_search.best_params_)
print("最佳R²:", grid_search.best_score_)

# ----------------------------------
# 6. 特征重要性分析
# ----------------------------------
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ----------------------------------
# 7. 模型保存与加载
# ----------------------------------
joblib.dump(best_rf, 'best_boston_model.pkl')
# loaded_model = joblib.load('best_boston_model.pkl')

# ----------------------------------
# 8. 预测示例
# ----------------------------------
sample_idx = 10
sample_data = X_scaled[sample_idx].reshape(1, -1)
true_price = y.iloc[sample_idx]
pred_price = best_rf.predict(sample_data)

print(f"\n预测示例:")
print(f"特征值: {X.iloc[sample_idx].values}")
print(f"真实价格: {true_price:.1f}千美元")
print(f"预测价格: {pred_price[0]:.1f}千美元")