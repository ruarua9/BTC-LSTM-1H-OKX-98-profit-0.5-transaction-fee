# OKX数据获取指南
import ccxt
import pandas as pd
import time
from requests.exceptions import RequestException


# 1. 设置OKX API（请替换为你的实际API密钥）
exchange = ccxt.okx({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'password': 'YOUR_API_PASSWORD',
    'enableRateLimit': True
})

# 2. 定义数据获取函数
def fetch_ohlcv(symbol, timeframe, limit):
    exchange = ccxt.okx()
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            exchange.load_markets()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except ccxt.NetworkError as e:
            if attempt < max_retries - 1:
                print(f"网络错误: {e}. 重试中... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise
        except ccxt.ExchangeError as e:
            print(f"交易所错误: {e}")
            raise
        except RequestException as e:
            print(f"请求错误: {e}")
            raise
        except Exception as e:
            print(f"未知错误: {e}")
            raise

def process_data(ohlcv_data):
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def save_data(df, filename):
    df.to_csv(filename)
    print(f"数据已保存到 {filename}")

if __name__ == "__main__":
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000

    try:
        ohlcv_data = fetch_ohlcv(symbol, timeframe, limit)
        df = process_data(ohlcv_data)
        save_data(df, 'btc_usdt_data.csv')
        print(f"成功获取并保存了 {len(df)} 条数据")
    except Exception as e:
        print(f"获取或处理数据时出错: {e}")



# 使用说明：
# 1. 替换'YOUR_API_KEY'，'YOUR_SECRET_KEY'和'YOUR_API_PASSWORD'为你的实际OKX API密钥
# 2. 运行此脚本来获取数据
# 3. 数据将被打印出来，并可选择保存到CSV文件