# 数据预处理和特征工程

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. 加载数据（假设我们已经将数据保存到CSV文件）
data = pd.read_csv('btc_usdt_data.csv', index_col='timestamp', parse_dates=True)

def preprocess_data(file_path):
    # 1. 加载数据
    data = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    
    # 2. 创建新的特征
    data = add_features(data)
    
    # 3. 处理缺失值
    data.dropna(inplace=True)
    
    # 4. 特征缩放
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_df

def add_features(df):
    # 添加简单移动平均线 (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # 添加相对强弱指标 (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 添加波动率
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # 添加价格变化百分比
    df['price_change'] = df['close'].pct_change()
    
    return df

# 如果你想在这个文件中测试函数，可以添加以下代码：
if __name__ == "__main__":
    processed_data = preprocess_data('btc_usdt_data.csv')
    print(processed_data.head())

# 5. 显示处理后的数据

# 6. 保存处理后的数据（可选）
    processed_data.to_csv('processed_btc_usdt_data.csv')

# 使用说明：
# 1. 确保你已经运行了之前的数据获取脚本并生成了'btc_usdt_data.csv'文件
# 2. 运行此脚本来处理数据和添加新特征
# 3. 处理后的数据将被打印出来，并可选择保存到新的CSV文件