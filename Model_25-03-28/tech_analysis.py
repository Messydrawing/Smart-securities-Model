# tech_analysis.py
import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    """
    计算技术指标 (MA, MACD, RSI, 布林带, ATR, KDJ等)
    并提供卡尔曼滤波对价格进行平滑处理的功能
    """
    @staticmethod
    def calc_ma(df, period=5, col="close"):
        return df[col].rolling(window=period).mean()

    @staticmethod
    def calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
        df["ema_fast"] = df["close"].ewm(span=fastperiod, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slowperiod, adjust=False).mean()
        df["dif"] = df["ema_fast"] - df["ema_slow"]
        df["dea"] = df["dif"].ewm(span=signalperiod, adjust=False).mean()
        df["macd"] = (df["dif"] - df["dea"]) * 2
        return df

    @staticmethod
    def calc_rsi(df, period=14):
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calc_bollinger_bands(df, period=20, std_multiplier=2):
        df["ma20"] = df["close"].rolling(window=period).mean()
        df["std20"] = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["ma20"] + std_multiplier * df["std20"]
        df["bb_lower"] = df["ma20"] - std_multiplier * df["std20"]
        return df

    @staticmethod
    def calc_atr(df, period=14):
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        return df

    @staticmethod
    def calc_volatility(df, window=20):
        """计算波动率"""
        return df['close'].pct_change().rolling(window).std()

    @staticmethod
    def calculate_all_indicators(df):
        """确保按正确顺序计算指标"""
        # 先计算MA5
        df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()  # 允许最小1个周期

        # 然后计算其他指标
        df = TechnicalAnalyzer.calc_bollinger_bands(df)
        df = TechnicalAnalyzer.calc_macd(df)
        df = TechnicalAnalyzer.calc_rsi(df)
        df = TechnicalAnalyzer.calc_atr(df)
        df = TechnicalAnalyzer.kdj(df)

        # 添加波动率
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std()
        return df.fillna(0)  # 填充剩余NaN

    @staticmethod
    def kalman_filter(series, Q=1e-5, R=0.001):
        """
        对一维序列应用简单的卡尔曼滤波，返回平滑后的序列（pandas.Series）
        Q: 过程噪声协方差
        R: 测量噪声协方差
        """
        n = len(series)
        xhat = np.zeros(n)     # 状态估计
        P = np.zeros(n)        # 估计误差协方差
        xhatminus = np.zeros(n)  # 预测值
        Pminus = np.zeros(n)
        K = np.zeros(n)        # 卡尔曼增益

        xhat[0] = series.iloc[0]
        P[0] = 1.0

        for k in range(1, n):
            # 预测
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            # 更新
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (series.iloc[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return pd.Series(xhat, index=series.index)

    @staticmethod
    def bollinger_bands(close_series, period=20, std_multiplier=2):
        """
        根据收盘价数据计算布林带
        参数：
            close_series: pandas.Series，收盘价数据
            period: 窗口期数，默认为20
            std_multiplier: 标准差乘数，默认为2
        返回：
            boll_upper: 上轨
            boll_mid: 中轨（均线）
            boll_lower: 下轨
        """
        if isinstance(close_series, pd.Series):
            df = pd.DataFrame({'close': close_series})
        else:
            df = close_series.copy()
        df = TechnicalAnalyzer.calc_bollinger_bands(df, period=period, std_multiplier=std_multiplier)
        boll_upper = df["bb_upper"]
        boll_mid = df["ma20"]
        boll_lower = df["bb_lower"]
        return boll_upper, boll_mid, boll_lower

    @staticmethod
    def kdj(df, period=9, k_period=3, d_period=3):
        """
        计算KDJ指标，并将计算结果添加到DataFrame中
        参数：
            df: 包含 'high', 'low', 'close' 列的DataFrame
            period: 计算RSV窗口期，默认为9
            k_period: K值平滑系数，默认为3
            d_period: D值平滑系数，默认为3
        返回：
            修改后的df，包含新增的 'kdj_k', 'kdj_d', 'kdj_j' 列
        """
        # 计算窗口内最低价和最高价
        low_min = df["low"].rolling(window=period, min_periods=1).min()
        high_max = df["high"].rolling(window=period, min_periods=1).max()
        # 计算RSV值（防止除零错误）
        rsv = (df["close"] - low_min) / (high_max - low_min + 1e-8) * 100
        # 计算K值与D值（使用指数加权平均）
        kdj_k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
        kdj_d = kdj_k.ewm(alpha=1 / d_period, adjust=False).mean()
        # 计算J值
        kdj_j = 3 * kdj_k - 2 * kdj_d
        # 将计算结果添加到DataFrame中
        df["kdj_k"] = kdj_k
        df["kdj_d"] = kdj_d
        df["kdj_j"] = kdj_j
        return df



