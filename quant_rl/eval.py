import yfinance as yf
import pandas as pd
from env import StockTradingEnv
from stable_baselines3 import PPO
import numpy as np
import random

eval_set = ["ORCL", "IBM", "ADBE", "CRM", "QCOM", "TXN", "CSCO", "SNOW", "ZM", "NET"]
# eval_set = ["SAP", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "INTC", "AMD"]
eval_set = ["PFE", "BA", "INTC"]

# 获取纳斯达克100指数的成分股列表
nasdaq100_tickers = yf.Ticker('^NDX').constituents.keys()
# 随机选择10只股票
random_tickers = random.sample(list(nasdaq100_tickers), 10)
print("随机选择的10只股票：", random_tickers)

def eval(ticker="MCD"):
    model = PPO.load("logs/best_model/best_model.zip")
    # 创建测试集环境
    test_data = yf.download(ticker, start="2023-01-01", end="2023-12-01")
    test_data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in test_data.columns]
    test_data.index = pd.to_datetime(test_data.index)
    test_data = test_data.drop(columns=['adj close'])

    print(test_data)

    test_env = StockTradingEnv(test_data, window_size=20)
    obs, _ = test_env.reset()

    # 确保观测值是正确的形状
    obs = obs.astype(np.float32)

    random_model = PPO("MlpPolicy", test_env, verbose=1)

    # 模拟交易
    for _ in range(len(test_data) - 20):
        action, _ = random_model.predict(obs)
        obs, reward, done, _ ,_ = test_env.step(action)
        test_env.render()
        if done:
            break

    # 最终资产价值
    final_balance = test_env.balance + test_env.position * test_data.iloc[-1]['close']
    print(f"Final Balance: {final_balance:.2f}")

    return final_balance

# final_list = []
# for ticker in eval_set:
#     final_list.append(eval(ticker=ticker))
# for i in range(len(eval_set)):
#     print(f"Final Balance for {eval_set[i]}: {final_list[i]:.2f}")
# print(f"Average Final Balance: {np.mean(final_list):.2f}")
