from flask import Flask
import joblib

app = Flask(__name__)

loaded_model = joblib.load("./finalized_model")
vect_model = joblib.load("./vect_model")

@app.route('/<string:name>/')
def predict(name):
    indata3=[name]  
    indatafeature = vect_model.transform(indata3)
    pred = loaded_model.predict(indatafeature)
    return "ham" if pred == [1] else "spam"

if __name__ == '__main__':
    app.run(debug=False)
