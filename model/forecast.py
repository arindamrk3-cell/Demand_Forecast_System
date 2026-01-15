import pandas as pd,os
from statsmodels.tsa.statespace.sarimax import SARIMAX
def generate_forecast(csv_path,steps):
    os.makedirs("output", exist_ok=True)
    os.makedirs("static", exist_ok=True)   
    df=pd.read_csv(csv_path,low_memory=False)
    required={"Date","Qty"}
    if not required.issubset(df.columns):
        print("invalid CSV format")
        raise ValueError("Invalid CSV format. Required Columns:Date,Qty")
    df.columns=df.columns.str.strip()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    if "Status" in df.columns:
        df=df[df["Status"].str.contains("Shipped",na=False)]
    df=df.dropna(subset=["Date","Qty"])
    df["Qty"]=pd.to_numeric(df["Qty"],errors="coerce")
    daily_demand=df.groupby("Date")["Qty"].sum().reset_index()
    daily_demand=daily_demand.sort_values("Date")
    daily_demand.set_index("Date",inplace=True)
    daily_demand=daily_demand.asfreq("D")
    daily_demand["Qty"]=daily_demand["Qty"].interpolate()
    daily_demand=daily_demand.iloc[:-3]
    daily_demand=daily_demand.tail(200)
    ##Train SARIMA model
    model=SARIMAX(
        daily_demand["Qty"],
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_invertibility=False,
        enforce_stationarity=False
    )
    model_fit=model.fit()
    forecast=model_fit.forecast(steps=steps)
    forecast=forecast.clip(lower=0)

    output_path="output/forecast.csv"
    forecast.to_csv(output_path)
    #forecast.to_csv("static/forecast.csv")

    print("Forecats Generated")
    history=daily_demand.tail(30)
    history_date=history.index.astype(str).tolist()
    history_values=history["Qty"].tolist()
    return output_path,history_date,history_values