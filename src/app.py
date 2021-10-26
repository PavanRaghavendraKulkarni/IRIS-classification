from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open("iri.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('myform.html')
@app.route("/myform", methods=["POST"])
def home():
    petal_length=request.form['Petal-length']
    petal_width=request.form['Petal-width']
    sepal_length=request.form['Sepal-length']
    sepal_width=request.form['sepal-width']

    arr=np.array([[petal_length,petal_width,sepal_length,sepal_width]])
    pred=model.predict(arr)
    return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)
