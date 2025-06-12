# 🚢 Kaggle-Titanic: 鐵達尼號生存預測完整實作

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一個完整的機器學習專案，實作經典的鐵達尼號生存預測問題。透過系統性的資料科學流程，從資料探索到模型部署，展示完整的 ML Pipeline。

## 📊 專案概述

本專案基於 [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) 資料集，利用乘客資訊（年齡、性別、艙等、票價等）預測鐵達尼號事件中的生存情況。專案採用完整的機器學習工作流程，包含資料探索、特徵工程、模型訓練和評估。

### 🎯 專案目標
- 實作完整的資料科學流程（5個里程碑）
- 比較多種機器學習演算法的效果
- 進行深度特徵工程和資料視覺化
- 達到具競爭力的預測準確率

## 🛠️ 技術棧

### 核心套件
- **Python 3.8+**: 主要程式語言
- **Pandas**: 資料處理和分析
- **NumPy**: 數值計算
- **Scikit-learn**: 機器學習框架
- **Matplotlib/Seaborn**: 資料視覺化

### 機器學習演算法
- Random Forest (最佳模型)
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree

## 🚀 快速開始

### 1. 環境設置
```bash
# 安裝依賴套件
pip install pandas numpy matplotlib seaborn scikit-learn

# 或使用 requirements.txt
pip install -r requirements.txt
```

### 2. 資料準備
確保 `csv/` 目錄包含以下檔案：
- `train.csv` - 訓練資料集 (891 筆)
- `test.csv` - 測試資料集 (418 筆)
- `gender_submission.csv` - 範例提交格式

### 3. 執行分析
```bash
# 一鍵執行完整分析
python titanic_complete_analysis.py
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
- 載入訓練集 (891行) 和測試集 (418行)
- 初步資料探索和缺失值分析
- 整體生存率統計: **38.4%**

### 里程碑 2: 資料分析與視覺化
- 多維度生存率分析 (性別、艙等、年齡、港口)
- 生成 3 組專業視覺化圖表
- 關鍵發現：女性存活率 74.2% vs 男性 18.9%

### 里程碑 3: 特徵工程
- **稱謂提取**: 從姓名提取 Mr/Miss/Mrs/Master/Rare
- **缺失值智能處理**: 隨機森林預測 263 個缺失年齡
- **衍生特徵**: 家庭大小、年齡分組、票價分組
- **特徵擴展**: 12 → 14 個最終特徵

### 里程碑 4: 模型訓練與預測
- **多模型比較**: 7 種演算法性能對比
- **交叉驗證**: 5折分層交叉驗證
- **最佳模型**: Random Forest (82.2% ± 1.1%)
- **特徵重要性**: Age (19.4%), Fare (17.2%), Title (16.0%)

### 里程碑 5: 結果分析與優化
- 生成 Kaggle 提交檔案
- 過擬合檢測和改進建議
- 完整分析報告輸出

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
- **正則表達式稱謂提取**: 精確識別 Mr/Miss/Mrs/Master
- **隨機森林缺失值預測**: 用 8 個特徵預測缺失年齡
- **多層次特徵創建**: 原始→衍生→編碼→最終特徵

### 穩健模型評估
- **分層交叉驗證**: 保持類別比例的 5折驗證
- **多模型集成比較**: 7 種演算法全面對比
- **過擬合檢測**: 自動診斷和改進建議

### 專業輸出格式
- **Kaggle 標準提交檔案**: 直接可用於競賽
- **模型持久化**: pickle 格式儲存最佳模型
- **JSON 分析報告**: 結構化成果記錄

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

### 改進方向
1. **模型優化**: 
   - 超參數調優
   - 特徵選擇算法
   - 模型融合技術

2. **視覺化增強**:
   - 互動式圖表 (Plotly)
   - 更多統計圖表
   - 動態儀表板

3. **程式碼優化**:
   - 模組化重構
   - 單元測試
   - 效能優化

## 📄 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- Kaggle 提供優質資料集和競賽平台
- Scikit-learn 社群的卓越機器學習工具
- 開源社群的無私貢獻

---

**🏆 專案特色**: 完整的 ML Pipeline | 豐富的視覺化 | 專業的文檔 | 即用的程式碼

**⭐ 如果這個專案對您有幫助，請給個星星支持！**
