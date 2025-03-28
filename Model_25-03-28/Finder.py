# Finder.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from eastmoney_api import EastMoneyAPI

api = EastMoneyAPI()


# 定义计算技术指标函数
def calculate_indicators(df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(window=14).mean() /
                                   df['close'].diff().where(lambda x: x < 0, 0).abs().rolling(window=14).mean())))
    df['bb_upper'] = df['ma20'] + (df['close'].rolling(window=20).std() * 2)
    df['bb_lower'] = df['ma20'] - (df['close'].rolling(window=20).std() * 2)
    return df.dropna()


# 单只股票训练函数
def train_stock(stock_code, stock_name):
    df = api.get_kline_data(stock_code, num=100)
    if df is None or len(df) < 20:
        print(f"{stock_name} 数据不足，跳过...")
        return None

    df = calculate_indicators(df)

    # 生成决策标签
    df['decision'] = 0
    df.loc[(df['ma5'] > df['ma20']) & (df['rsi'] < 30), 'decision'] = 1
    df.loc[(df['ma5'] < df['ma20']) & (df['rsi'] > 70), 'decision'] = 2

    X = df[['ma5', 'ma20', 'macd', 'rsi', 'bb_lower', 'bb_upper']].values
    y = df['decision'].values

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # 检查是否有足够类别1数据
    if class_counts.get(1, 0) < 2:
        print(f"{stock_name} ({stock_code}) 类别1样本不足，跳过训练...")
        return None

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据过采样
    X_minority = X_scaled[y == 1]
    y_minority = y[y == 1]
    X_majority = X_scaled[y == 0]
    y_majority = y[y == 0]

    # 过采样少数类
    n_samples = min(X_majority.shape[0], len(X_minority))
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, y_minority, replace=True, n_samples=n_samples, random_state=42)

    # 合并数据
    X_resampled = np.vstack((X_majority, X_minority_oversampled))
    y_resampled = np.concatenate((y_majority, y_minority_oversampled))

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 模型训练
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 获取特征重要性
    feature_importances = model.feature_importances_
    feature_names = ['ma5', 'ma20', 'macd', 'rsi', 'bb_lower', 'bb_upper']
    weights = dict(zip(feature_names, feature_importances))

    print(f"{stock_name} ({stock_code}) 特征权重:", weights)

    return weights


# 批量股票训练
def train_initial_weights(stock_list):
    all_weights = {}
    for stock_name, stock_code in stock_list.items():
        weights = train_stock(stock_code, stock_name)
        if weights:
            all_weights[stock_name] = weights

    if not all_weights:
        print("无有效股票数据，训练未成功！")
        return None

    # 计算所有股票的平均权重
    avg_weights = {}
    for feature in ['ma5', 'ma20', 'macd', 'rsi', 'bb_lower', 'bb_upper']:
        avg_weights[feature] = np.mean([w[feature] for w in all_weights.values()])

    # 归一化
    total = sum(avg_weights.values())
    normalized_weights = {k: v / total for k, v in avg_weights.items()}

    print("所有股票平均归一化权重:", normalized_weights)

    return normalized_weights
