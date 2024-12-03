import backtrader as bt
from datetime import datetime
import yfinance as yf
import pandas as pd
from env import StockTradingEnv
from stable_baselines3 import PPO
import numpy as np

# 1. 创建简单的交易策略
class SmaCross(bt.Strategy):
    params = (('fast', 5), ('slow', 30),)

    def __init__(self):
        
        
        self.model = PPO.load("logs/best_model/best_model.zip")
        self.test_env = StockTradingEnv(test_data, window_size=20)
        self.obs, _ = self.test_env.reset()
        
        # self.rsi = bt.indicators.RSI(self.data.close, period=14)
        # self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)

    def notify_order(self, order):
        """订单状态更新时调用"""
        if order.status in [order.Completed]:  # 如果订单完成
            if order.isbuy():  # 买入订单
                print(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Date: {self.data.datetime.date()}")
            elif order.issell():  # 卖出订单
                print(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Date: {self.data.datetime.date()}")

            self.order = None  # 清空订单

    def notify_trade(self, trade):
        """交易完成时调用"""
        if trade.isclosed:  # 如果交易已完成
            print("--------------------")
            # print(f"TRADE PROFIT, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

    def next(self):
        # 定义止盈和止损比例
        take_profit = 10  # take_profit = take_model()
        stop_loss = 0.9   # baseline
        profit_stop_loss = 0.95
        cash = self.broker.getcash()
        price = self.data.close[0]
        if len(self.data) < 20:
            return
        action, _ = self.model.predict(self.obs)
        self.obs, reward, done, _ ,_ = self.test_env.step(action)
        # self.recent_high = max(self.data.high.get(size=30))

        # 计算买入数量，使用 50% 仓位
        size = (cash * self.allocation) // price

        if self.position.size == 0:  # 如果当前无持仓
            # 买入信号 buy_model()
            # if self.data.close[0] < self.bb.lines.bot[0]:
            if self.sma_fast[0] > self.sma_slow[0] and self.sma_fast[-1] <= self.sma_slow[-1]:
                self.buy(size=size)
                self.entry_price = self.data.close[0]  # 记录买入价格
                self.higest_price = self.entry_price
        else:
            # 记录历史最高价格
            if self.data.close[0] > self.higest_price:
                self.higest_price = self.data.close[0]
            # 如果有持仓，检查是否达到止盈或止损条件
            if self.data.close[0] >= self.entry_price * take_profit:
                self.sell(size=self.position.size)  # 达到止盈条件，卖出
            elif self.data.close[0] <= self.higest_price * profit_stop_loss:
                self.sell(size=self.position.size)  # 达到盈利止损条件，卖出
            elif self.data.close[0] <= self.entry_price * stop_loss:
                self.sell(size=self.position.size)  # 达到止损条件，卖出
            elif self.sma_fast[0] < self.sma_slow[0] and self.sma_fast[-1] >= self.sma_slow[-1]:
                self.sell(size=self.position.size)

# 下载股票数据
ticker = "SAP"
data = yf.download(ticker, start="2023-11-01", end="2024-12-01")
print(data)
data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
data.index = pd.to_datetime(data.index)
data_bt = bt.feeds.PandasData(dataname=data)
# 计算该股票区间内变化率
print(f"Price change: {data['close'][-1] / data['close'][0] - 1:.2%}")

# 4. 设置回测环境
cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)  # 添加策略
cerebro.adddata(data_bt)          # 加载数据
cerebro.broker.set_cash(10000) # 初始资金
cerebro.broker.setcommission(commission=0.001)  # 设置交易佣金（0.1%）

# 5. 启动回测
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

# 5. 可视化结果
cerebro.plot()
