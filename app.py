from flask import Flask, request, jsonify, render_template
import os, sys
from src.pipeline.prediction_pipeline import PredictOutput, CustomData

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
         data=CustomData(
            indus=float(request.form.get('indus')),
            nox = float(request.form.get('nox')),
            rm = float(request.form.get('rm')),
            tax = float(request.form.get('tax')),
            ptratio = float(request.form.get('ptratio')),
            lstat = float(request.form.get('lstat')),
            dis = request.form.get('dis'),
            age= request.form.get('age')
        )
         final_new_data=data.get_data_as_dataframe()
         predict_pipeline=PredictOutput()
         pred=predict_pipeline.get_output(final_new_data)

         results=round(pred[0],2)

         return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run("0.0.0.0", debug=True)

