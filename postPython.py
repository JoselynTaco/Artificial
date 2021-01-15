import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request, json

app = Flask(__name__)

# {"RI":0,"Na":0,"Mg":0,"Al":0,"Si":0,"K":0,"Ca":0,"Ba":0,"Fe":0}

scaler = MinMaxScaler()
Min=pickle.load(open('Min.p', 'rb' ))
Max=pickle.load(open('Max.p', 'rb' ))
mod=pickle.load(open('mod.p','rb')) 

@app.route('/')
def holamundo():
    return 'Aqui'


@app.route('/modeloRN', methods=['POST'])
def modeloRN():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    X = escalado(datos,Min,Max)
    modelo = pickle.load(open('modeloRN.p','rb'))
    #modelo = abrirModelo('modeloRN.p')
    predict=modelo.predict_classes(X)
    result = str(predict[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/modeloRNPCA', methods=['POST'])
def modeloRNPCA():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    datos = np.atleast_2d(datos)
    modPCA = mod.transform(datos)
    modelo = pickle.load(open('modeloRNPCA.p','rb'))
    #modelo = abrirModelo('modeloRNPCA.p')

    result = str(modelo.predict_classes(modPCA)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/modeloKNN', methods=['POST'])
def modeloKNN():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    X = escalado(datos,Min,Max)
    modelo = pickle.load(open('modeloKNN.p','rb'))
    #modelo = abrirModelo('modeloKNN.p')
    result = str(modelo.predict(X)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/modeloKNNPCA', methods=['POST'])
def modeloKNNPCA():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    datos = np.atleast_2d(datos)
    modPCA = mod.transform(datos)
    modelo = pickle.load(open('modeloKNNPCA.p','rb'))
    #modelo = abrirModelo('modeloKNNPCA.p')
    
    result = str(modelo.predict(modPCA)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/modeloSVM', methods=['POST'])
def modeloSVM():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    X = escalado(datos,Min,Max)
    modelo = pickle.load(open('modeloSVM.p','rb'))
   # modelo = abrirModelo('modeloSVM.p')
    result = str(modelo.predict(X)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/modeloSVMPCA', methods=['POST'])
def modeloSVMPCA():
    content = request.get_json()
    datos = np.array([content['RI'],content['Na'],content['Mg'],content['Al'],content['Si'],content['K'],content['Ca'],content['Ba'],content['Fe']])
    datos = np.atleast_2d(datos)
    modPCA = mod.transform(datos)
    modelo = pickle.load(open('modeloSVMPCA.p','rb'))
    #modelo = abrirModelo('modeloSVMPCA.p')

    result = str(modelo.predict(modPCA)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response


def abrirModelo(ruta):
    modelo = pickle.load(open(ruta, 'rb'))
    return modelo

def escalado(X,Min,Max):

    i = range(0,9)
    for i in i:
        xstd = (X[i]-Min[i])/(Max[i]-Min[i])
        xsc = xstd*(1-0)+0
        X[i]=xsc
    X = np.atleast_2d(X)
    return X

if __name__ == "__main__":
    app.run()
