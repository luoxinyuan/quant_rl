from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
import yfinance as yf
import pandas as pd
from env import StockTradingEnv
# from feature_extractor import CustomFeatureExtractor

# tensorboard --logdir ppo_stock_trading_tensorboard/ --port 6006


train_set = ["SAP", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "INTC", "AMD"]
eval_set = ["ORCL", "IBM", "ADBE", "CRM", "QCOM", "TXN", "CSCO", "SNOW", "ZM", "NET"]

# policy_kwargs = dict(
#     features_extractor_class=CustomFeatureExtractor,
#     features_extractor_kwargs=dict(features_dim=256),
# )

def train(ticker="SAP", start="2023-11-01", end="2024-11-01", total_timesteps=100000):
    # 准备数据
    data = yf.download(ticker, start=start, end=end)
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    data.index = pd.to_datetime(data.index)
    data = data.drop(columns=['adj close'])

    # 创建环境
    env = StockTradingEnv(data, window_size=20)
    check_env(env)  # 检查环境是否符合 Gym 标准
    vec_env = DummyVecEnv([lambda: env])  # 向量化环境

    tensorboard_log_dir = "ppo_stock_trading_tensorboard/"
    best_model_path = "logs/best_model/best_model.zip"

    # 检查是否有已保存的最佳模型
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model = PPO.load(best_model_path, env=vec_env, tensorboard_log=tensorboard_log_dir)
    else:
        print("No best model found, starting training from scratch.")
        model = PPO("MlpPolicy", 
                    vec_env, 
                    verbose=1, 
                    tensorboard_log=tensorboard_log_dir,
                    # policy_kwargs=policy_kwargs,
                    )
    # 定义回调函数
    eval_env = DummyVecEnv([lambda: env])  # 用于评估的环境
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="logs/best_model/",
        log_path="logs/results/",
        eval_freq=5000,  # 每隔 5000 步评估一次模型
        deterministic=True,
        render=False,
    )
    # 训练模型
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    # 保存模型
    model.save("ppo_stock_trading")
    print("Training complete. Best model saved.")

for ticker in train_set:
    print(f"Training model for {ticker}")
    train(ticker=ticker, start="2023-11-01", end="2024-11-01", total_timesteps=100000)