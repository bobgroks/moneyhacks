from binance.client import AsyncClient, Client
import asyncio
import config
from alpha_hedge_strategy import Strategy
import numpy as np
import pandas as pd
from decimal import Decimal

symbol_lis = ['BTCUSDT', 'ETHUSDT']
signals = [1, 0]
order_size = 100



class Trade:
    def __int__(self, api_key, api_secret, symbol_lis, signals, order_size):
        self.client = Client(api_key, api_secret)
        self.symbol_lis = symbol_lis
        self.signals = np.array(signals)
        self.order_size = order_size
        self.m = len(self.symbol_lis)

        # get basic info of each futures contract
        info_df = pd.DataFrame(self.client.futures_exchange_info()['symbols']).set_index('symbol').loc[self.symbol_lis]
        info_df = info_df.loc[(info_df['contractType'] == 'PERPETUAL') & (info_df['status'] == 'TRADING')]

        tick_size = []
        step_size = []
        notional = []
        for i in range(self.m):
            tick_size.append(info_df['filters'][i][0]['tickSize'])
            step_size.append(info_df['filters'][i][1]['stepSize'])
            notional.append(info_df['filters'][i][-2]['notional'])
        self.tick_size = np.array(tick_size).astype(np.float64)
        self.step_size = np.array(step_size).astype(np.float64)
        self.notional = np.array(notional).astype(np.float64)



    async def place_batch_order(self, batch_orders):
        await self.client.futures_place_batch_order(batchOrders=batch_orders)



    def create_batch_orders(self):
        # calculate expected weight
        allocated_weight = self.signals
        if np.nansum(np.abs(allocated_weight)) == 0:
            allocated_weight = np.array([0] * self.m)
        else:
            allocated_weight = allocated_weight / np.nansum(np.abs(allocated_weight))

        # get present position
        position_df = pd.DataFrame(self.client.futures_position_information()).set_index('symbol')
        present_position = position_df.loc[self.symbol_lis, 'positionAmt'].astype(np.float64)

        # get present price
        ticker_df = pd.DataFrame(self.client.futures_symbol_ticker()).set_index('symbol')
        present_price = ticker_df.loc[self.symbol_lis, 'price'].astype(np.float64).values

        # calculate trade position
        trade_position = (allocated_weight * self.order_size) / present_price - present_position

        # quantity filter
        trade_position = trade_position // self.step_size * self.step_size
        # price filter
        trade_price = np.array([present_price[i] - self.tick_size[i] if trade_position[i] > 0 else present_price[i] - self.tick_size[i] for i in range(self.m)])
        # notional filter
        no_trade_cond = np.abs(trade_position * trade_price) < self.notional
        trade_position[no_trade_cond] = 0.0
        # create batch orders
        orders_container = []
        for i in range(self.m):
            if trade_position[i] != 0:
                if trade_position > 0:
                    side = 'BUY'
                else:
                    side = 'SELL'
                orders_container.append({'symbol': self.symbol_lis[i], 'type': 'LIMIT', 'timeInForce': 'GTC', 'side': side,
                                         'price': str(trade_price[i]), 'quantity': str(trade_position[i]), 'positionSide': 'BOTH'})
        batch_size = 5
        order_batches = [orders_container[i:i + batch_size] for i in range(0, len(orders_container), batch_size)]
        return order_batches



    def run(self):
        order_batches = self.create_batch_orders()
        loop = asyncio.get_event_loop()
        for bacth in order_batches:
            loop.run_until_complete(self.client.futures_place_batch_order(batchOrders=bacth))

        loop.close()



