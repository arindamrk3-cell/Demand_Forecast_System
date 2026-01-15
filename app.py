from flask import Flask,render_template,request
import os,pandas as pd
from model.forecast import generate_forecast
app=Flask(__name__)
uploads="uploads"
os.makedirs("uploads",exist_ok=True)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/forecast",methods=['POST'])
def forecast():
        try:
            file=request.files["file"]
            days=int(request.form["days"])
            filepath=os.path.join(uploads,file.filename)
            file.save(filepath)

            forecast_path,history_dates,history_values=generate_forecast(filepath,days)
            forecast_df=pd.read_csv(forecast_path)

            return render_template("forecast.html",forecast_dates=forecast_df.iloc[:,0].tolist(),forecast_values=forecast_df.iloc[:,1].tolist(),history_dates=history_dates,history_values=history_values,download=True)
        except ValueError as e:
            return render_template("index.html",error=str(e))
if __name__=="__main__":
    app.run()