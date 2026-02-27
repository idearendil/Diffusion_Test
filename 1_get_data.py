from pykrx import stock
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os
import pandas as pd

start_date = datetime(2013, 10, 24)
end_date = datetime(2025, 12, 31)

date_list = []
current_date = start_date

while current_date <= end_date:
    date_list.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=30)

tickers = set()
for a_date in tqdm(date_list):
    tickers_temp = stock.get_market_ticker_list(a_date, market="KOSPI")
    time.sleep(0.1)
    tickers = set(tickers).union(set(tickers_temp))

tickers = list(tickers)
print(len(tickers))

out_dir = "raw_data/"
os.makedirs(out_dir, exist_ok=True)

existing_csvs = [
    f for f in os.listdir(out_dir)
    if f.endswith(".csv")
]

print(f"Existing CSVs: {len(existing_csvs)}, ex: {existing_csvs[0]}")

start_date_str = start_date.strftime("%Y%m%d")
end_date_str = end_date.strftime("%Y%m%d")

for a_ticker in tqdm(tickers):
    if a_ticker + ".csv" in existing_csvs:
        continue
    df_1 = stock.get_market_ohlcv(start_date_str, end_date_str, a_ticker)
    time.sleep(1)
    df_2 = stock.get_market_trading_value_by_date(start_date_str, end_date_str, a_ticker, on="매수")
    if len(df_2) == 0:
        continue
    df_2 = df_2.rename(columns={
            '기관합계': '매수_기관합계',
            '기타법인': '매수_기타법인',
            '개인': '매수_개인',
            '외국인합계': '매수_외국인합계',
        })
    df_2.drop(columns=['전체'], inplace=True)
    time.sleep(1)
    df_3 = stock.get_market_trading_value_by_date(start_date_str, end_date_str, a_ticker, on="매도")
    if len(df_3) == 0:
        continue
    df_3 = df_3.rename(columns={
            '기관합계': '매도_기관합계',
            '기타법인': '매도_기타법인',
            '개인': '매도_개인',
            '외국인합계': '매도_외국인합계',
        })
    time.sleep(1)
    df = pd.concat([df_1, df_2, df_3], axis=1)
    df.index = pd.to_datetime(df.index)
    df.to_csv(out_dir + f"{a_ticker}.csv", encoding="utf-8-sig")
