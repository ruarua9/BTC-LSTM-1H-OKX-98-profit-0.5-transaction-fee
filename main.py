import pandas as pd
from data_collection import fetch_ohlcv
from data_preprocessing import preprocess_data
from lstm_model import train_lstm_model, save_model
from backtesting import backtest

def main():
    # 1. 数据收集
    print("正在收集数据...")
    ohlcv_data = fetch_ohlcv('BTC/USDT', '1h', 1000)
    
    # 将列表转换为DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    df.to_csv('btc_usdt_data.csv')
    print("数据已保存到 btc_usdt_data.csv")

    # 2. 数据预处理
    print("正在预处理数据...")
    processed_data = preprocess_data('btc_usdt_data.csv')
    print(processed_data.head())  # 打印预处理后的数据，进行检查
    processed_data.to_csv('processed_btc_usdt_data.csv')
    print("预处理后的数据已保存到 processed_btc_usdt_data.csv")

    # 3. 训练LSTM模型
    print("正在训练LSTM模型...")
    model, X_test, y_test = train_lstm_model('processed_btc_usdt_data.csv')
    save_model(model, 'lstm_model.keras')
    print("模型已保存到 lstm_model.keras")

    # 4. 回测
    print("正在进行回测...")
    final_balance = backtest(processed_data, model)
    total_return = (final_balance - 10000) / 10000 * 100
    print(f'初始资金: $10,000')
    print(f'最终资金: ${final_balance:.2f}')
    print(f'总回报率: {total_return:.2f}%')

if __name__ == "__main__":
    main()

# 使用说明:
# 1. 确保你已经设置好了环境并安装了所有必要的库
# 2. 运行这个脚本来执行整个流程,从数据收集到回测
# 3. 检查输出结果和生成的文件
































