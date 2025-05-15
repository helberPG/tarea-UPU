from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Cargar el modelo Random Forest y las columnas
modelo = joblib.load('modelo_randomforest.pkl')
columnas = joblib.load('columnas_modelo.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        # Obtener los datos del formulario
        data = request.get_json()
        print("Datos recibidos:", data)  # Depuración

        # Verificar si los datos se recibieron correctamente
        if not data:
            return jsonify({'error': 'No se recibieron datos.'}), 400

        # Convertir los datos a un DataFrame y ordenar las columnas
        features = pd.DataFrame([data], columns=columnas)
        features = features.apply(pd.to_numeric, errors='coerce')

        # Verificar si hay valores nulos después de la conversión
        if features.isnull().any().any():
            print("Error: Datos no numéricos recibidos.")
            return jsonify({'error': 'Datos no válidos o incompletos.'}), 400

        # Realizar la predicción
        prediction = modelo.predict(features)
        resultado = int(prediction[0])

        print("Resultado de la predicción:", resultado)  # Depuración

        return jsonify({'clasificacion': resultado})
    except Exception as e:
        print("Error durante la predicción:", str(e))  # Depuración
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)