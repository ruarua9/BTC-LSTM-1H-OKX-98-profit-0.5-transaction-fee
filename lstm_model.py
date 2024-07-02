# LSTM模型实现
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def prepare_data(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])  # 预测'close'价格
    return np.array(X), np.array(y)

def train_lstm_model(data_path, epochs=100, batch_size=32):
       # 1. 加载处理后的数据
    data = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

       # 2. 准备数据
    X, y = prepare_data(data.values)

       # 3. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # 4. 构建LSTM模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

       # 5. 训练模型
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    # 可视化训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
       # 6. 评估模型
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss}')

    return model, X_test, y_test

def save_model(model, filepath):
    if not filepath.endswith('.keras'):
        filepath = filepath.rsplit('.', 1)[0] + '.keras'
    model.save(filepath)
    print(f"模型已保存为 {filepath}")

if __name__ == "__main__":
    model, X_test, y_test = train_lstm_model('processed_btc_usdt_data.csv')
       
       # 7. 进行预测
    last_sequence = X_test[-1:]
    prediction = model.predict(last_sequence)
    print(f'下一个时间段的预测收盘价: {prediction[0][0]}')

       # 保存模型
    save_model(model, 'lstm_model.keras')


## 使用说明：
# 1. 确保你已经运行了之前的数据处理脚本并生成了'processed_btc_usdt_data.csv'文件
# 2. 运行此脚本来构建和训练LSTM模型
# 3. 脚本将输出测试损失和对下一个时间段收盘价的预测
