from flask import Flask,request,render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
from src.exception import CustomException
import sys

application = Flask(__name__)

app = application

@app.route('/', methods=['GET'])
def homepage():
    return render_template("home.html")

@app.route('/predict', methods=['GET','POST'])
def resultpage():
    try:
        if request.method == "GET":
            return render_template("home.html")       
        else:
            data = CustomData(
                gender=request.form.get('gender'),
                age=float(request.form.get('age')),
                hypertension=request.form.get('hypertension'),
                heart_disease=request.form.get('heart_disease'),
                ever_married=request.form.get('ever_married'),
                work_type=request.form.get('work_type'),
                Residence_type=request.form.get('Residence_type'),
                avg_glucose_level=float(request.form.get('avg_glucose_level')),
                bmi=float(request.form.get('bmi')),
                smoking_status=request.form.get('smoking_status')
            )

            pred_df=data.get_data_as_dataframe()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            print("after Prediction")
            if results[0] == 1:
                message = "Risk - Most likely to get Stroke"
            else:
                message = "No Risk of Stroke"
            return render_template('results.html',result=message)

    except Exception as e:
        raise CustomException(e,sys)

if __name__== "__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)

