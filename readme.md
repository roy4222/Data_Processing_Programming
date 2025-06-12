# 🚢 Kaggle-Titanic: 鐵達尼號生存預測完整實作

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一個完整的機器學習專案，實作經典的鐵達尼號生存預測問題。結合所有里程碑的綜合分析腳本，展示從資料探索到模型部署的完整 ML Pipeline。

## 📊 專案概述

本專案基於 [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) 資料集，利用乘客資訊（年齡、性別、艙等、票價等）預測鐵達尼號事件中的生存情況。專案採用完整的機器學習工作流程，包含資料探索、特徵工程、模型訓練和評估。

### 🎯 專案目標
- 實作完整的資料科學流程（5個里程碑）
- 比較 7 種機器學習演算法的效果
- 進行深度特徵工程和資料視覺化
- 達到具競爭力的預測準確率 (82%+)

### 📚 參考資料
本專案參考以下優質資源：
- [HackMD 完整實作教學](https://hackmd.io/@Go3PyC86QhypSl7kh5nA2Q/Hk4nXFYkK)
- [Medium 機器學習實戰](https://medium.com/@elvennote/kaggle-titanic-machine-learning-from-disaster)
- CodeSignal 和 GitHub 最佳實踐

## 🛠️ 技術棧

### 核心套件
- **Python 3.8+**: 主要程式語言
- **Pandas**: 資料處理和分析
- **NumPy**: 數值計算
- **Scikit-learn**: 機器學習框架
- **Matplotlib/Seaborn**: 資料視覺化
- **Pickle**: 模型序列化
- **JSON**: 報告格式化

### 機器學習演算法（完整比較）
1. **Random Forest** (最佳模型) - n_estimators=100
2. **Logistic Regression** 
3. **Support Vector Machine (Linear)**
4. **Support Vector Machine (RBF)**
5. **K-Nearest Neighbors** (k=5)
6. **Gaussian Naive Bayes**
7. **Decision Tree** (criterion='entropy')

## 🚀 快速開始

### 1. 環境設置
```bash
# 安裝依賴套件
pip install pandas numpy matplotlib seaborn scikit-learn

# 必要套件清單
- pandas: 資料處理
- numpy: 數值計算  
- matplotlib: 視覺化
- seaborn: 進階圖表
- scikit-learn: 機器學習
- pickle: 模型儲存 (內建)
- json: 報告輸出 (內建)
- re: 正則表達式 (內建)
```

### 2. 資料準備
確保 `csv/` 目錄包含以下檔案：
- `train.csv` - 訓練資料集 (891 筆記錄)
- `test.csv` - 測試資料集 (418 筆記錄)
- `gender_submission.csv` - 範例提交格式

### 3. 執行分析
```bash
# 一鍵執行完整分析 (包含所有 5 個里程碑)
python titanic_complete_analysis.py

# 腳本會自動：
# 1. 載入並探索資料
# 2. 生成視覺化圖表
# 3. 進行特徵工程
# 4. 訓練並比較 7 種模型
# 5. 生成預測結果和分析報告
```

## 📋 專案結構

```
titanic-survival-prediction/
├── 📁 csv/                                    # 資料集
│   ├── train.csv                              # 訓練資料 (891 rows)
│   ├── test.csv                               # 測試資料 (418 rows)
│   └── gender_submission.csv                  # 提交範例
├── 🐍 titanic_complete_analysis.py            # 主要分析腳本
├── 📊 生成檔案/
│   ├── titanic_visualization_analysis.png     # 綜合視覺化圖表
│   ├── titanic_detailed_survival_analysis.png # 詳細生存率分析
│   ├── titanic_correlation_analysis.png       # 特徵相關性分析
│   ├── titanic_complete_submission.csv        # Kaggle 提交檔案
│   ├── titanic_best_model_random_forest.pkl   # 最佳模型
│   └── titanic_complete_analysis_report.json  # 分析報告
└── 📖 README.md                               # 專案說明
```

## 🔬 分析流程 (5個里程碑)

### 里程碑 1: 資料載入與探索
```python
# 核心功能實作
- 載入 csv/train.csv (891行) 和 csv/test.csv (418行)
- 顯示資料集基本資訊和特徵欄位清單
- 統計缺失值: Age (19.9%), Cabin (77.1%), Embarked (0.2%)
- 計算整體生存率: 38.4% (342/891 人存活)
- 錯誤處理: FileNotFoundError 自動退出
```

### 里程碑 2: 資料分析與視覺化
```python
# 深度分析實作
- 合併訓練和測試集 (1309 筆記錄) 進行統一處理
- 多維度生存率分析：
  * 艙等: 1等艙 63% > 2等艙 47% > 3等艱 24%
  * 性別: 女性 74.2% vs 男性 18.9%
  * 年齡: 兒童(0-12) > 青年(19-30) > 中年(31-50)
  * 港口: 瑟堡 55.4% > 皇后鎮 39% > 南安普敦 33.7%

# 自動生成 3 組專業圖表 (PNG格式, DPI=300)
1. titanic_visualization_analysis.png - 綜合分析 (2×3 子圖)
2. titanic_detailed_survival_analysis.png - 詳細生存率 (2×2 子圖)  
3. titanic_correlation_analysis.png - 特徵相關性 (1×2 子圖)
```

### 里程碑 3: 特徵工程
```python
# 智能特徵工程實作
1. 稱謂提取 (extract_title 函數):
   - 正則表達式: re.search(' ([A-Za-z]+)\.', name)
   - 合併稀有稱謂: Dr/Rev/Col/Major → 'Rare'
   - 標準化: Mlle→Miss, Mme→Mrs

2. 缺失值智能處理:
   - Embarked: 用眾數 'S' 填補
   - Fare: 用中位數填補
   - Age: RandomForestRegressor 預測 (使用8個特徵)

3. 新特徵創建:
   - 票號前綴處理 (extract_ticket_prefix)
   - 客艙等級提取 (extract_cabin_class)  
   - 家庭大小分類: Alone/Small/Large
   - 年齡分組: Child/Teen/Young_Adult/Middle_Age/Senior
   - 票價四分位分組: Low/Medium_Low/Medium_High/High

4. 特徵編碼和最終選擇:
   - 原始特徵 12 個 → 最終特徵 14 個
   - 所有類別變數進行 LabelEncoder 編碼
```

### 里程碑 4: 模型訓練與預測
```python
# 完整模型比較 (train_multiple_models 函數)
7 種演算法同時訓練：
1. LogisticRegression(random_state=42)
2. KNeighborsClassifier(n_neighbors=5)  
3. SVC(kernel='linear', random_state=42)
4. SVC(kernel='rbf', random_state=42)
5. GaussianNB()
6. DecisionTreeClassifier(criterion='entropy', random_state=42)
7. RandomForestClassifier(n_estimators=100, random_state=42)

# 評估方法
- StratifiedKFold 5折交叉驗證
- 顯示每個模型的訓練準確率和 CV 分數 ± 標準差
- 自動選擇 CV 分數最高的模型作為最佳模型
- Random Forest 特徵重要性分析 (前5名)
```

### 里程碑 5: 結果分析與優化
```python
# 完整輸出流程
1. 檔案生成:
   - titanic_complete_submission.csv (Kaggle 提交格式)
   - titanic_best_model_*.pkl (最佳模型序列化)
   - titanic_complete_analysis_report.json (詳細分析報告)

2. 性能診斷:
   - 過擬合檢測: |訓練分數 - CV分數| 分析
   - 自動建議: 正則化/超參數調優/模型融合

3. 最終總結:
   - 處理資料筆數、特徵工程統計
   - 最佳模型性能和預測存活率  
   - 下一步優化方向建議

4. 可選互動評估:
   - 用戶選擇是否查看詳細模型評估
   - 混淆矩陣和分類報告輸出
```

## 📈 專案成果

### 🏆 模型性能
| 模型 | 交叉驗證準確率 | 標準差 |
|------|:-------------:|:------:|
| **Random Forest** | **82.15%** | **±1.11%** |
| Logistic Regression | 81.37% | ±0.96% |
| SVM Linear | 80.70% | ±1.62% |
| Gaussian NB | 78.68% | ±1.51% |
| Decision Tree | 78.56% | ±2.11% |
| K-Neighbors | 74.52% | ±1.66% |
| SVM RBF | 69.25% | ±0.77% |

### 🔍 關鍵洞察
1. **性別因素**: 女性存活率 (74.2%) 遠高於男性 (18.9%)
2. **社會階層**: 1等艙 (63.0%) vs 3等艙 (24.2%) 存活率差異顯著
3. **年齡影響**: 兒童 (0-12歲) 存活率最高 (58.0%)
4. **港口差異**: 瑟堡港 (55.4%) > 皇后鎮 (39.0%) > 南安普敦 (33.7%)

### 📊 特徵重要性排序
1. **Age** (19.4%) - 年齡是最重要的預測因子
2. **Fare** (17.2%) - 票價反映社會地位
3. **Title** (16.0%) - 稱謂包含性別和社會資訊
4. **Sex** (14.4%) - 性別對生存有決定性影響
5. **Pclass** (6.9%) - 艙等反映經濟狀況

## 🎨 視覺化展示

專案自動生成 3 組高品質圖表：

1. **綜合分析圖表** (`titanic_visualization_analysis.png`)
   - 整體生存率分布
   - 性別、艙等、港口生存率比較
   - 年齡分布和票價箱線圖

2. **詳細生存率分析** (`titanic_detailed_survival_analysis.png`)
   - 性別×艙等交叉分析熱力圖
   - 年齡組、家庭大小、票價區間生存率

3. **特徵相關性分析** (`titanic_correlation_analysis.png`)
   - 特徵相關性矩陣
   - 各特徵與生存率的相關係數

## 🔧 技術亮點

### 智能特徵工程
```python
# 實際程式碼實作
def extract_title(name):
    """從姓名中提取稱謂"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else ""

def categorize_family_size(size):
    """家庭大小分類"""
    if size == 1: return 'Alone'
    elif size <= 4: return 'Small'
    else: return 'Large'

# 隨機森林年齡預測
rf_age = RandomForestRegressor(n_estimators=100, random_state=42)
rf_age.fit(known_age[age_features], known_age['Age'])
predicted_ages = rf_age.predict(unknown_age[age_features])
```

### 穩健模型評估
```python
# 分層交叉驗證實作
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

# 自動最佳模型選擇
best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
best_model = all_models[best_model_name]

# 過擬合檢測
overfitting = train_score - best_cv_score
if overfitting > 0.1: print("嚴重過擬合")
elif overfitting > 0.05: print("中度過擬合") 
else: print("模型泛化良好")
```

### 專業輸出格式
```python
# Kaggle 提交檔案
submission = pd.DataFrame({
    'PassengerId': original_test['PassengerId'],
    'Survived': test_predictions.astype(int)
})
submission.to_csv('titanic_complete_submission.csv', index=False)

# 模型序列化
model_filename = f'titanic_best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

# JSON 結構化報告
analysis_report = {
    "專案名稱": "Kaggle-Titanic: 鐵達尼號生存預測完整實作",
    "完成時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "模型比較結果": {name: f"{result['mean']:.4f} ± {result['std']:.4f}" 
                    for name, result in cv_results.items()}
}
```

## 🎯 使用案例

### 學習目的
- 機器學習完整流程學習
- 特徵工程技巧實作
- 資料視覺化最佳實踐

### 實際應用
- Kaggle 競賽提交 (預期分數: 79-80%)
- 機器學習課程教學範例
- 面試作品集展示

### 進階擴展
- 超參數調優 (GridSearchCV)
- 模型融合 (Ensemble Methods)
- 深度學習方法 (Neural Networks)

## 📚 學習資源

### 參考文章
- [HackMD 完整實作教學](https://hackmd.io/@Go3PyC86QhypSl7kh5nA2Q/Hk4nXFYkK)
- [Medium 機器學習實戰](https://medium.com/@elvennote/kaggle-titanic-machine-learning-from-disaster)
- [CodeSignal 資料前處理](https://codesignal.com/learn/courses/data-preprocessing-for-machine-learning)

### 相關競賽
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Titanic 討論區](https://www.kaggle.com/c/titanic/discussion)

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

### 改進方向 (基於程式碼實作)
1. **模型優化**: 
   ```python
   # 腳本建議的優化方向
   print("🚀 進階優化方向:")
   print("   1. 超參數調優 (GridSearchCV)")
   print("   2. 特徵選擇和降維")
   print("   3. 模型融合 (Ensemble)")
   print("   4. 深度學習方法")
   ```

2. **視覺化增強**:
   - 目前生成 3 組靜態 PNG 圖表 (DPI=300)
   - 可擴展為互動式 Plotly 圖表
   - 添加更多統計分析視圖

3. **程式碼優化**:
   - 當前 685 行單一腳本實作
   - 可模組化分離各個里程碑
   - 加入單元測試和錯誤處理

## 📄 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- Kaggle 提供優質資料集和競賽平台
- Scikit-learn 社群的卓越機器學習工具
- 開源社群的無私貢獻

---

## 📋 程式碼核心結構

```python
# titanic_complete_analysis.py (685 行)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 里程碑 1: 資料載入與探索 (約 50-100 行)
# =============================================================================
train_df = pd.read_csv('csv/train.csv')  # 891 rows
test_df = pd.read_csv('csv/test.csv')    # 418 rows
# 缺失值統計、生存率計算

# =============================================================================  
# 里程碑 2: 資料分析與視覺化 (約 100-300 行)
# =============================================================================
# 3 組圖表生成: plt.subplots(), seaborn heatmap
# 分析: 艙等、性別、年齡、港口 vs 生存率

# =============================================================================
# 里程碑 3: 特徵工程 (約 300-450 行)
# =============================================================================
def extract_title(name): ...          # 稱謂提取
def categorize_family_size(size): ... # 家庭大小分類
# RandomForestRegressor 預測缺失年齡
# 最終 14 個特徵選擇

# =============================================================================
# 里程碑 4: 模型訓練與預測 (約 450-550 行)
# =============================================================================
def train_multiple_models(X_train, y_train): # 7 種演算法
# StratifiedKFold 5折交叉驗證
# 自動選擇最佳模型

# =============================================================================
# 里程碑 5: 結果分析與文件生成 (約 550-685 行)
# =============================================================================
# CSV 提交檔案、PKL 模型儲存、JSON 報告
# 過擬合檢測、互動式詳細評估
```

**🏆 專案特色**: 
- ✅ **完整的 685 行實作** - 從資料載入到模型部署
- ✅ **5 個里程碑整合** - 系統性機器學習流程
- ✅ **7 種演算法比較** - 自動選擇最佳模型  
- ✅ **智能特徵工程** - 正則表達式+隨機森林預測
- ✅ **專業視覺化** - 3 組高解析度圖表
- ✅ **Kaggle 就緒** - 標準提交格式
- ✅ **完整文檔** - 程式碼註解+分析報告

**⭐ 如果這個專案對您有幫助，請給個星星支持！**
