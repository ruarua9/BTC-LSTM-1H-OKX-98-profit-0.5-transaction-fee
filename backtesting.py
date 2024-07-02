# 简单回测系统
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def backtest(data, model, initial_balance=10000, transaction_fee=0.005):
    balance = initial_balance
    position = 0
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    predictions = []
    actual_prices = []
    buy_signals = []
    sell_signals = []

    print(f'Data length: {len(data)}')  # 打印数据长度进行检查
    
    for i in range(60, len(data)):  # 使用过去60个时间段的数据进行预测
        sequence = scaled_data[i-60:i]
        sequence = np.reshape(sequence, (1, sequence.shape[0], sequence.shape[1]))
        prediction = model.predict(sequence)[0][0]
        actual_price = data.iloc[i]['close']
        
        predictions.append(prediction)
        actual_prices.append(actual_price)
        
        if prediction > actual_price * (1 + transaction_fee) and balance > 0:
            # 买入信号
            position = balance / actual_price
            balance = 0
            buy_signals.append((data.index[i], actual_price))
        elif prediction < actual_price * (1 - transaction_fee) and position > 0:
            # 卖出信号
            balance = position * actual_price * (1 - transaction_fee)
            position = 0
            sell_signals.append((data.index[i], actual_price))

    # 确保 actual_prices 列表不为空
    print(f'Length of actual_prices: {len(actual_prices)}')

    # 计算最终资产价值
    final_balance = balance + position * data.iloc[-1]['close']
    
    # 可视化预测结果和交易信号
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[60:], actual_prices, label='Actual Prices')
    plt.plot(data.index[60:], predictions, label='Predicted Prices')
    
    buy_signals_x, buy_signals_y = zip(*buy_signals) if buy_signals else ([], [])
    sell_signals_x, sell_signals_y = zip(*sell_signals) if sell_signals else ([], [])
    
    plt.scatter(buy_signals_x, buy_signals_y, marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(sell_signals_x, sell_signals_y, marker='v', color='r', label='Sell Signal', alpha=1)
    
    plt.title('Actual vs Predicted Prices with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    return final_balance

if __name__ == "__main__":
    # 加载数据和模型以进行测试
    data = pd.read_csv('processed_btc_usdt_data.csv', index_col='timestamp', parse_dates=True)
    print(data.head())  # 打印数据，进行检查
    model = load_model('lstm_model.keras')
    
    final_balance = backtest(data, model)
    total_return = (final_balance - 10000) / 10000 * 100

    print(f'初始资金: $10,000')
    print(f'最终资金: ${final_balance:.2f}')
    print(f'总回报率: {total_return:.2f}%')






































































