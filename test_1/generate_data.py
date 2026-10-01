import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# ---------------------------------------------------------
# 1. 讀取本地 CSV 檔案與設定預測天數
# ---------------------------------------------------------
CSV_FILE = "Stock_combined.csv"

# 預測目標天數 (以交易日估算)
PREDICTION_HORIZONS = {
    "1D": 1,
    "5D": 5,
    "10D": 10,
    "20D": 20,
    "1Q": 63,
    "0.5Y": 126,
    "1Y": 252,
}

# 簡單的公司名稱與產業映射字典 (如果需要，可自行擴充或保持簡化)
COMPANY_INFO = {
    1101: {"name": "台泥", "industry": "水泥工業", "domain": "taiwancement.com"},
    1102: {"name": "亞泥", "industry": "水泥工業", "domain": "acc.com.tw"},
    1216: {"name": "統一", "industry": "食品工業", "domain": "uni-president.com.tw"},
    2330: {"name": "台積電", "industry": "半導體", "domain": "tsmc.com"},
    2454: {"name": "聯發科", "industry": "半導體", "domain": "mediatek.com"},
    2317: {"name": "鴻海", "industry": "其他電子", "domain": "foxconn.com"},
    2308: {"name": "台達電", "industry": "電子零組件", "domain": "deltaww.com"},
    2881: {"name": "富邦金", "industry": "金融保險", "domain": "fubon.com"},
    2882: {"name": "國泰金", "industry": "金融保險", "domain": "cathayholdings.com"},
}


def main():
    print("🚀 正在讀取 Stock_combined.csv...")
    df = pd.read_csv(CSV_FILE)

    # 確保日期排序與格式正確
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Stock", "Date"])

    # 轉成 Pivot Table: Index為日期, Column為股票代碼, Value為 Return_Ln
    returns_pivot = df.pivot(
        index="Date", columns="Stock", values="Return_Ln"
    ).dropna(how="all")
    returns_pivot = returns_pivot.fillna(0)  # 補齊空值

    # 計算最新市值與平均對數報酬率
    latest_date = df["Date"].max()
    latest_df = df[df["Date"] == latest_date].copy()

    # 取前 50 大交易量或成分股 (若總數多於 50)
    top_stocks = returns_pivot.columns[:50]
    returns_pivot = returns_pivot[top_stocks]

    print(
        f"✅ 成功處理 {len(top_stocks)} 支股票，時間區間：{returns_pivot.index.min().strftime('%Y-%m-%d')} ~ {returns_pivot.index.max().strftime('%Y-%m-%d')}"
    )

    # 2. 計算 Markowitz Return-Risk (年化)
    # 日對數報酬率 -> 年化報酬率與年化標準差
    annual_returns = returns_pivot.mean() * 252
    annual_volatility = returns_pivot.std() * np.sqrt(252)

    # 3. PCA 降維計算 (components = 2)
    print("📊 執行 PCA 降維分析...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(returns_pivot.T)
    pca_df = pd.DataFrame(
        pca_features, index=returns_pivot.columns, columns=["PC1", "PC2"]
    )

    # 4. Ridge Regression 預測未來 Return
    print("🤖 執行 Ridge 迴歸模型預測...")
    predictions = {stock: {} for stock in top_stocks}

    for stock in top_stocks:
        stock_returns = returns_pivot[stock]
        df_single = pd.DataFrame({"Return": stock_returns})

        for name, steps in PREDICTION_HORIZONS.items():
            df_single[f"Target_{name}"] = df_single["Return"].shift(-steps)

            # 特徵：過去 5 天的 Return
            for lag in range(1, 6):
                df_single[f"Lag_{lag}"] = df_single["Return"].shift(lag)

            feature_cols = [f"Lag_{lag}" for lag in range(1, 6)]
            train_data = df_single.dropna()

            if len(train_data) > 50:
                X = train_data[feature_cols]
                y = train_data[f"Target_{name}"]

                model = Ridge(alpha=1.0)
                model.fit(X, y)

                latest_x = (
                    df_single[feature_cols].iloc[-1].values.reshape(1, -1)
                )
                pred_val = model.predict(latest_x)[0]
                predictions[stock][name] = round(float(pred_val), 2)
            else:
                predictions[stock][name] = 0.0

    # 5. 彙整數據並導出 data.json
    output_companies = []
    for stock in top_stocks:
        stock_str = str(stock)
        info = COMPANY_INFO.get(
            stock,
            {
                "name": f"股票 {stock_str}",
                "industry": "上市企業",
                "domain": "google.com",
            },
        )

        # 計算市值：收盤價 * 發行股數
        stock_latest = latest_df[latest_df["Stock"] == stock]
        if not stock_latest.empty:
            close_p = stock_latest["Close"].values[0]
            shares = stock_latest["Outstanding_Shares"].values[0]
            market_cap_val = (close_p * shares) / 1e8  # 單位：億台幣
            market_cap_str = f"{market_cap_val:.2f} 億"
        else:
            market_cap_str = "N/A"

        company_data = {
            "ticker": stock_str,
            "full_ticker": f"{stock_str}.TW",
            "name": info["name"],
            "industry": info["industry"],
            "market_cap": market_cap_str,
            "logo": f"https://logo.clearbit.com/{info['domain']}",
            "return_risk": {
                "expected_return": round(
                    float(annual_returns.get(stock, 0)) * 100, 2
                ),
                "volatility": round(
                    float(annual_volatility.get(stock, 0)) * 100, 2
                ),
            },
            "pca": {
                "pc1": round(float(pca_df.loc[stock, "PC1"]), 4),
                "pc2": round(float(pca_df.loc[stock, "PC2"]), 4),
            },
            "predictions": predictions.get(stock, {}),
        }
        output_companies.append(company_data)

    final_json = {
        "study_period": "2019-01-02 to 2026-06-18",
        "data_source": "Stock_combined.csv",
        "total_companies": len(output_companies),
        "companies": output_companies,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("🎉 成功從 CSV 檔案生成 data.json！可以直接給前端網頁使用了。")


if __name__ == "__main__":
    main()