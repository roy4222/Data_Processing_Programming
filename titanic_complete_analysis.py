#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle-Titanic: 鐵達尼號生存預測完整實作
結合所有里程碑的綜合分析腳本

本腳本包含完整的機器學習流程：
1. 資料載入與探索
2. 資料分析與視覺化
3. 特徵工程
4. 模型訓練與預測
5. 結果分析與優化建議

參考資料：
- https://hackmd.io/@Go3PyC86QhypSl7kh5nA2Q/Hk4nXFYkK
- https://medium.com/@elvennote/kaggle-titanic-machine-learning-from-disaster
- CodeSignal 和 GitHub 最佳實踐
"""

# =============================================================================
# 導入所有必要的套件
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import re
import pickle
from datetime import datetime

# 機器學習相關套件
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# 設定圖表參數和警告過濾
plt.rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚢 Kaggle-Titanic: 鐵達尼號生存預測完整實作")
print("📊 結合所有里程碑的綜合分析")
print("=" * 80)

# =============================================================================
# 里程碑 1: 資料載入與探索
# =============================================================================
print("\n" + "=" * 60)
print("📊 里程碑 1: 資料載入與探索")
print("=" * 60)

# 載入資料集
try:
    train_df = pd.read_csv('csv/train.csv')
    test_df = pd.read_csv('csv/test.csv')
    print("✅ 資料集載入成功")
    print(f"   訓練集: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    print(f"   測試集: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
except FileNotFoundError:
    print("❌ 找不到資料檔案，請確認 csv/train.csv 和 csv/test.csv 存在")
    exit(1)

# 基本資料探索
print(f"\n📋 資料集基本資訊:")
print(f"   訓練集大小: {train_df.shape}")
print(f"   測試集大小: {test_df.shape}")
print(f"   特徵欄位: {list(train_df.columns)}")

# 缺失值統計
print(f"\n📊 訓練集缺失值統計:")
missing_train = train_df.isnull().sum()
for col, missing in missing_train[missing_train > 0].items():
    percentage = (missing / len(train_df)) * 100
    print(f"   {col}: {missing} ({percentage:.1f}%)")

# 生存率統計
survival_rate = train_df['Survived'].mean()
survival_count = train_df['Survived'].sum()
print(f"\n🎯 整體生存統計:")
print(f"   存活人數: {survival_count} / {len(train_df)}")
print(f"   整體存活率: {survival_rate:.1%}")

# =============================================================================
# 里程碑 2: 資料分析與視覺化
# =============================================================================
print("\n" + "=" * 60)
print("📈 里程碑 2: 資料分析與視覺化")
print("=" * 60)

# 合併資料集進行統一分析
test_df['Survived'] = np.nan
full_data = pd.concat([train_df, test_df], ignore_index=True)
print(f"✅ 資料合併完成: {full_data.shape[0]} 行")

# 分析各特徵與生存率的關係
print(f"\n📊 各特徵與生存率分析:")

# 1. 艙等分析
pclass_survival = train_df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
print(f"\n🎫 艙等生存率:")
for pclass in pclass_survival.index:
    count = pclass_survival.loc[pclass, 'count']
    rate = pclass_survival.loc[pclass, 'mean']
    print(f"   {pclass}等艙: {count:3d}人, 存活率 {rate:.1%}")

# 2. 性別分析
sex_survival = train_df.groupby('Sex')['Survived'].agg(['count', 'mean'])
print(f"\n👫 性別生存率:")
for sex in sex_survival.index:
    count = sex_survival.loc[sex, 'count']
    rate = sex_survival.loc[sex, 'mean']
    print(f"   {sex:6s}: {count:3d}人, 存活率 {rate:.1%}")

# 3. 年齡分析
train_with_age = train_df.dropna(subset=['Age'])
age_groups = pd.cut(train_with_age['Age'], bins=[0, 12, 18, 30, 50, 100], 
                   labels=['兒童(0-12)', '青少年(13-18)', '青年(19-30)', '中年(31-50)', '老年(51+)'])
age_survival = train_with_age.groupby(age_groups)['Survived'].agg(['count', 'mean'])
print(f"\n👶 年齡組生存率:")
for age_group in age_survival.index:
    count = age_survival.loc[age_group, 'count']
    rate = age_survival.loc[age_group, 'mean']
    print(f"   {age_group}: {count:3d}人, 存活率 {rate:.1%}")

# 4. 出發港口分析
embarked_survival = train_df.groupby('Embarked')['Survived'].agg(['count', 'mean'])
print(f"\n🚢 出發港口生存率:")
port_names = {'S': '南安普敦', 'C': '瑟堡', 'Q': '皇后鎮'}
for port in embarked_survival.index:
    if pd.notna(port):
        count = embarked_survival.loc[port, 'count']
        rate = embarked_survival.loc[port, 'mean']
        print(f"   {port}({port_names.get(port, port)}): {count:3d}人, 存活率 {rate:.1%}")

# =============================================================================
# 視覺化分析圖表
# =============================================================================
print(f"\n📊 生成視覺化圖表...")

# 設定圖表樣式
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)

# 創建綜合分析圖表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Titanic Survival Analysis - Visualization Report', fontsize=16, fontweight='bold')

# 1. 整體生存率圓餅圖
survival_counts = train_df['Survived'].value_counts()
colors = ['#ff6b6b', '#4ecdc4']
labels = ['Died', 'Survived']
axes[0, 0].pie(survival_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, explode=(0.05, 0))
axes[0, 0].set_title('Overall Survival Distribution')

# 2. 性別生存率長條圖
sex_survival_data = train_df.groupby(['Sex', 'Survived']).size().unstack()
sex_survival_data.plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(['Died', 'Survived'])
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. 艙等生存率長條圖
pclass_survival_data = train_df.groupby(['Pclass', 'Survived']).size().unstack()
pclass_survival_data.plot(kind='bar', ax=axes[0, 2], color=['#ff6b6b', '#4ecdc4'])
axes[0, 2].set_title('Survival by Passenger Class')
axes[0, 2].set_xlabel('Passenger Class')
axes[0, 2].set_ylabel('Count')
axes[0, 2].legend(['Died', 'Survived'])
axes[0, 2].tick_params(axis='x', rotation=0)

# 4. 年齡分布直方圖
train_df['Age'].hist(bins=30, ax=axes[1, 0], alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')

# 5. 票價分布箱線圖 (按艙等)
train_df.boxplot(column='Fare', by='Pclass', ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Class')
axes[1, 1].set_xlabel('Passenger Class')
axes[1, 1].set_ylabel('Fare')

# 6. 出發港口生存率長條圖
embarked_survival_data = train_df.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
embarked_survival_data.plot(kind='bar', ax=axes[1, 2], color=['#ff6b6b', '#4ecdc4'])
axes[1, 2].set_title('Survival by Embarkation Port')
axes[1, 2].set_xlabel('Embarkation Port')
axes[1, 2].set_ylabel('Count')
axes[1, 2].legend(['Died', 'Survived'])
axes[1, 2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('titanic_visualization_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ 綜合分析圖表已儲存: titanic_visualization_analysis.png")

# 創建詳細的生存率分析圖表
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Detailed Survival Analysis', fontsize=16, fontweight='bold')

# 1. 性別 x 艙等 生存率熱力圖
pivot_table = train_df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes2[0, 0])
axes2[0, 0].set_title('Survival Rate by Gender x Class')

# 2. 年齡組生存率
train_with_age = train_df.dropna(subset=['Age'])
age_groups = pd.cut(train_with_age['Age'], bins=[0, 12, 18, 30, 50, 100], 
                   labels=['Child\n(0-12)', 'Teen\n(13-18)', 'Young Adult\n(19-30)', 'Middle Age\n(31-50)', 'Senior\n(51+)'])
age_survival_rate = train_with_age.groupby(age_groups)['Survived'].mean()
age_survival_rate.plot(kind='bar', ax=axes2[0, 1], color='lightcoral')
axes2[0, 1].set_title('Survival Rate by Age Group')
axes2[0, 1].set_ylabel('Survival Rate')
axes2[0, 1].tick_params(axis='x', rotation=45)

# 3. 家庭大小 vs 生存率
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
family_survival = train_df.groupby('FamilySize')['Survived'].agg(['count', 'mean'])
family_survival['mean'].plot(kind='bar', ax=axes2[1, 0], color='lightgreen')
axes2[1, 0].set_title('Survival Rate by Family Size')
axes2[1, 0].set_xlabel('Family Size')
axes2[1, 0].set_ylabel('Survival Rate')

# 4. 票價區間 vs 生存率
fare_bins = pd.qcut(train_df['Fare'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
fare_survival = train_df.groupby(fare_bins)['Survived'].mean()
fare_survival.plot(kind='bar', ax=axes2[1, 1], color='gold')
axes2[1, 1].set_title('Survival Rate by Fare Range')
axes2[1, 1].set_xlabel('Fare Range')
axes2[1, 1].set_ylabel('Survival Rate')
axes2[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('titanic_detailed_survival_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ 詳細生存率分析圖表已儲存: titanic_detailed_survival_analysis.png")

# 創建相關性分析圖表
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')

# 準備數值化的資料進行相關性分析
correlation_data = train_df.copy()
correlation_data['Sex_num'] = correlation_data['Sex'].map({'male': 0, 'female': 1})
correlation_data['Embarked_num'] = correlation_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 選擇數值特徵進行相關性分析
numeric_features = ['Survived', 'Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num']
corr_matrix = correlation_data[numeric_features].corr()

# 1. 相關性熱力圖
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes3[0])
axes3[0].set_title('Feature Correlation Matrix')

# 2. 與生存率的相關性長條圖
survival_corr = corr_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=False)
colors = ['red' if x < 0 else 'green' for x in survival_corr.values]
survival_corr.plot(kind='bar', ax=axes3[1], color=colors)
axes3[1].set_title('Feature Correlation with Survival')
axes3[1].set_ylabel('Correlation Coefficient')
axes3[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes3[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('titanic_correlation_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ 相關性分析圖表已儲存: titanic_correlation_analysis.png")

# 關閉圖表以避免阻塞程式執行
plt.close('all')
print(f"\n📊 視覺化分析完成！已生成 3 個圖表檔案:")

# =============================================================================
# 里程碑 3: 特徵工程
# =============================================================================
print("\n" + "=" * 60)
print("🔧 里程碑 3: 特徵工程")
print("=" * 60)

# 創建特徵工程後的資料副本
data_processed = full_data.copy()

# 1. 稱謂提取
def extract_title(name):
    """從姓名中提取稱謂"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

data_processed['Title'] = data_processed['Name'].apply(extract_title)

# 合併稀有稱謂
def map_title(title):
    """合併稀有稱謂"""
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    return title_mapping.get(title, 'Rare')

data_processed['Title_Clean'] = data_processed['Title'].apply(map_title)
print("✅ 稱謂提取完成")

# 2. 票號處理
def extract_ticket_prefix(ticket):
    """提取票號前綴"""
    if pd.isna(ticket):
        return 'Unknown'
    ticket = str(ticket).replace(' ', '').replace('.', '').replace('/', '').upper()
    match = re.match(r'([A-Z]+)', ticket)
    if match:
        return match.group(1)
    else:
        return 'X'

data_processed['Ticket_Prefix'] = data_processed['Ticket'].apply(extract_ticket_prefix)
print("✅ 票號處理完成")

# 3. 客艙處理
def extract_cabin_class(cabin):
    """提取客艙等級"""
    if pd.isna(cabin):
        return 'NoCabin'
    cabin_letter = str(cabin)[0]
    return cabin_letter if cabin_letter.isalpha() else 'NoCabin'

data_processed['Cabin_Class'] = data_processed['Cabin'].apply(extract_cabin_class)
print("✅ 客艙處理完成")

# 4. 家庭大小特徵
data_processed['FamilySize'] = data_processed['SibSp'] + data_processed['Parch'] + 1
data_processed['IsAlone'] = (data_processed['FamilySize'] == 1).astype(int)

def categorize_family_size(size):
    """家庭大小分類"""
    if size == 1:
        return 'Alone'
    elif size <= 4:
        return 'Small'
    else:
        return 'Large'

data_processed['FamilySize_Group'] = data_processed['FamilySize'].apply(categorize_family_size)
print("✅ 家庭大小特徵完成")

# 5. 缺失值處理
print(f"\n🔧 缺失值處理:")

# Embarked 缺失值處理
mode_embarked = data_processed['Embarked'].mode()[0]
data_processed['Embarked'].fillna(mode_embarked, inplace=True)
print(f"   Embarked: 用 '{mode_embarked}' 填補")

# Fare 缺失值處理
median_fare = data_processed['Fare'].median()
data_processed['Fare'].fillna(median_fare, inplace=True)
print(f"   Fare: 用中位數 {median_fare:.2f} 填補")

# Age 缺失值處理 - 使用隨機森林預測
print(f"   Age: 使用隨機森林預測...")
age_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'FamilySize']

# 創建數值編碼
data_processed['Sex_num'] = data_processed['Sex'].map({'male': 0, 'female': 1})
data_processed['Embarked_num'] = data_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_processed['Title_num'] = data_processed['Title_Clean'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4
})

age_features.extend(['Sex_num', 'Embarked_num', 'Title_num'])

known_age = data_processed[data_processed['Age'].notna()]
unknown_age = data_processed[data_processed['Age'].isna()]

if len(unknown_age) > 0:
    rf_age = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_age.fit(known_age[age_features], known_age['Age'])
    predicted_ages = rf_age.predict(unknown_age[age_features])
    data_processed.loc[data_processed['Age'].isna(), 'Age'] = predicted_ages
    print(f"   Age: 預測了 {len(unknown_age)} 個缺失值")

# 6. 創建衍生特徵
def categorize_age(age):
    """年齡分組"""
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 30:
        return 'Young_Adult'
    elif age <= 50:
        return 'Middle_Age'
    else:
        return 'Senior'

data_processed['Age_Group'] = data_processed['Age'].apply(categorize_age)

# 票價分組
data_processed['Fare_Group'] = pd.qcut(data_processed['Fare'], q=4, 
                                      labels=['Low', 'Medium_Low', 'Medium_High', 'High'])

print("✅ 衍生特徵創建完成")

# 7. 類別變數編碼
categorical_columns = ['Sex', 'Embarked', 'Title_Clean', 'Cabin_Class', 
                      'FamilySize_Group', 'Age_Group', 'Fare_Group']

label_mappings = {}
for col in categorical_columns:
    if col in data_processed.columns:
        unique_values = data_processed[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        label_mappings[col] = mapping
        data_processed[f'{col}_encoded'] = data_processed[col].map(mapping)

print("✅ 類別變數編碼完成")

# 準備最終特徵
final_features = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
    'Sex_encoded', 'Embarked_encoded', 'Title_Clean_encoded',
    'Cabin_Class_encoded', 'FamilySize_Group_encoded',
    'Age_Group_encoded', 'Fare_Group_encoded'
]

# 分離訓練和測試資料
train_processed = data_processed[:len(train_df)].copy()
test_processed = data_processed[len(train_df):].copy()

print(f"\n📊 特徵工程完成:")
print(f"   最終特徵數量: {len(final_features)}")
print(f"   訓練集: {train_processed.shape[0]} 行")
print(f"   測試集: {test_processed.shape[0]} 行")

# =============================================================================
# 里程碑 4: 模型訓練與預測
# =============================================================================
print("\n" + "=" * 60)
print("🤖 里程碑 4: 模型訓練與預測")
print("=" * 60)

# 準備訓練資料
X_train = train_processed[final_features]
y_train = train_processed['Survived']
X_test = test_processed[final_features]

print(f"✅ 訓練資料準備完成:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

# 多模型比較函數
def train_multiple_models(X_train, y_train):
    """訓練多個機器學習模型"""
    models = {}
    
    # 1. 邏輯回歸
    models['Logistic Regression'] = LogisticRegression(random_state=42)
    
    # 2. K近鄰
    models['K-Neighbors'] = KNeighborsClassifier(n_neighbors=5)
    
    # 3. 支持向量機 (線性)
    models['SVM Linear'] = SVC(kernel='linear', random_state=42)
    
    # 4. 支持向量機 (RBF)
    models['SVM RBF'] = SVC(kernel='rbf', random_state=42)
    
    # 5. 高斯樸素貝葉斯
    models['Gaussian NB'] = GaussianNB()
    
    # 6. 決策樹
    models['Decision Tree'] = DecisionTreeClassifier(criterion='entropy', random_state=42)
    
    # 7. 隨機森林
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 訓練所有模型
    trained_models = {}
    print(f"\n🚀 開始訓練多個模型...")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        trained_models[name] = model
        print(f"   {name}: 訓練準確率 {train_score:.4f} ({train_score:.1%})")
    
    return trained_models

# 訓練多個模型
all_models = train_multiple_models(X_train, y_train)

# 交叉驗證評估
print(f"\n🔄 5折交叉驗證評估:")
cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in all_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"   {name:18s}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 選擇最佳模型
best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
best_model = all_models[best_model_name]
best_cv_score = cv_results[best_model_name]['mean']

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   交叉驗證準確率: {best_cv_score:.4f} ({best_cv_score:.1%})")

# 特徵重要性分析 (針對 Random Forest)
if 'Random Forest' in all_models:
    rf_model = all_models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': final_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 隨機森林特徵重要性 (前5名):")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']:20s}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")

# 使用最佳模型進行預測
print(f"\n🎯 使用最佳模型進行測試集預測...")
test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

survival_count = test_predictions.sum()
survival_rate = test_predictions.mean()

print(f"✅ 預測完成:")
print(f"   預測存活人數: {int(survival_count)} / {len(test_predictions)}")
print(f"   預測存活率: {survival_rate:.1%}")

# =============================================================================
# 里程碑 5: 結果分析與文件生成
# =============================================================================
print("\n" + "=" * 60)
print("📋 里程碑 5: 結果分析與文件生成")
print("=" * 60)

# 生成提交檔案
original_test = pd.read_csv('csv/test.csv')
submission = pd.DataFrame({
    'PassengerId': original_test['PassengerId'],
    'Survived': test_predictions.astype(int)
})

submission_filename = 'titanic_complete_submission.csv'
submission.to_csv(submission_filename, index=False)
print(f"✅ 提交檔案已生成: {submission_filename}")

# 儲存最佳模型
model_filename = f'titanic_best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ 最佳模型已儲存: {model_filename}")

# 生成詳細分析報告
analysis_report = {
    "專案名稱": "Kaggle-Titanic: 鐵達尼號生存預測完整實作",
    "完成時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "資料統計": {
        "訓練集大小": len(train_df),
        "測試集大小": len(test_df),
        "原始特徵數": len(train_df.columns) - 1,
        "最終特徵數": len(final_features),
        "整體存活率": f"{survival_rate:.1%}"
    },
    "模型比較結果": {name: f"{result['mean']:.4f} ± {result['std']:.4f}" 
                    for name, result in cv_results.items()},
    "最佳模型": {
        "模型名稱": best_model_name,
        "交叉驗證分數": f"{best_cv_score:.4f}",
        "預測存活率": f"{survival_rate:.1%}"
    },
    "特徵重要性": feature_importance.head(10).to_dict('records') if 'Random Forest' in all_models else [],
    "關鍵發現": {
        "性別影響": f"女性存活率 {sex_survival.loc['female', 'mean']:.1%}, 男性 {sex_survival.loc['male', 'mean']:.1%}",
        "艙等影響": f"1等艙 {pclass_survival.loc[1, 'mean']:.1%}, 3等艙 {pclass_survival.loc[3, 'mean']:.1%}",
        "年齡影響": "兒童存活率較高",
        "特徵工程": f"從 {len(train_df.columns)-1} 個原始特徵擴展到 {len(final_features)} 個"
    }
}

report_filename = 'titanic_complete_analysis_report.json'
with open(report_filename, 'w', encoding='utf-8') as f:
    json.dump(analysis_report, f, ensure_ascii=False, indent=2)
print(f"✅ 分析報告已生成: {report_filename}")

# 性能改進建議
print(f"\n💡 模型優化建議:")
train_score = best_model.score(X_train, y_train)
overfitting = train_score - best_cv_score

if overfitting > 0.1:
    print("   ⚠️  嚴重過擬合:")
    print("      - 增加正則化參數")
    print("      - 減少模型複雜度")
    print("      - 使用更多數據")
elif overfitting > 0.05:
    print("   ⚡ 中度過擬合:")
    print("      - 調整超參數")
    print("      - 特徵選擇")
else:
    print("   ✅ 模型泛化良好")

print(f"\n🚀 進階優化方向:")
print("   1. 超參數調優 (GridSearchCV)")
print("   2. 特徵選擇和降維")
print("   3. 模型融合 (Ensemble)")
print("   4. 深度學習方法")

# 最終總結
print(f"\n" + "=" * 80)
print("🎉 鐵達尼號生存預測完整分析完成！")
print("=" * 80)

print(f"\n📊 最終成果總覽:")
print(f"   🔢 處理資料: {len(full_data):,} 筆記錄")
print(f"   🧪 特徵工程: {len(train_df.columns)-1} → {len(final_features)} 個特徵")
print(f"   🤖 最佳模型: {best_model_name}")
print(f"   🎯 預測準確率: {best_cv_score:.1%} (交叉驗證)")
print(f"   📈 預測存活率: {survival_rate:.1%}")

print(f"\n📁 生成檔案:")
print(f"   • {submission_filename} - Kaggle 提交檔案")
print(f"   • {model_filename} - 最佳訓練模型")
print(f"   • {report_filename} - 詳細分析報告")

print(f"\n🎯 下一步行動:")
print("   1. 📤 提交結果到 Kaggle 競賽")
print("   2. 📊 分析排行榜表現")
print("   3. 🔧 根據反饋優化模型")
print("   4. 💡 嘗試更進階的技術")

print(f"\n🏆 恭喜完成完整的機器學習專案！")
print("   這個腳本展示了從資料探索到模型部署的完整流程。")
print("   您可以基於這個基礎繼續優化和改進模型性能。")
print("\n" + "=" * 80)

# 可選：顯示最佳模型的詳細評估
if input("\n是否顯示最佳模型的詳細評估？(y/n): ").lower() == 'y':
    # 分割一部分數據進行詳細評估
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 重新訓練最佳模型
    best_model.fit(X_train_split, y_train_split)
    y_val_pred = best_model.predict(X_val_split)
    
    print(f"\n📊 最佳模型詳細評估:")
    print(f"   驗證集準確率: {accuracy_score(y_val_split, y_val_pred):.4f}")
    
    # 混淆矩陣
    cm = confusion_matrix(y_val_split, y_val_pred)
    print(f"\n📋 混淆矩陣:")
    print(f"           預測")
    print(f"實際    死亡  存活")
    print(f"死亡    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"存活    {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # 分類報告
    print(f"\n📈 詳細分類報告:")
    print(classification_report(y_val_split, y_val_pred, 
                              target_names=['死亡', '存活']))

print(f"\n✨ 分析完成！感謝使用完整版鐵達尼號生存預測分析腳本！") 