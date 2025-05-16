from flask import Flask, request, render_template, jsonify, send_file
import joblib
import pandas as pd
import io
from xhtml2pdf import pisa

app = Flask(__name__)

# Cargar modelo y columnas
modelo = joblib.load('modelo_randomforest.pkl') 
columnas = joblib.load('columnas_modelo.pkl')

# Clases de salida
clasificadores = ['Normal', 'Sospechoso', 'Patológico']

@app.route('/')
@app.route('/prediccion_salud_fetal')
def formulario():
    return render_template('Formulario.html')

@app.route('/predecir_formulario', methods=['POST'])
def predecir_formulario():
    if request.is_json:
        datos = request.get_json()
        datos = {col: float(datos[col]) for col in columnas}
        df = pd.DataFrame([datos])
        pred = modelo.predict(df)[0]
        return jsonify({'prediccion': clasificadores[int(pred)]})  # <-- Cambio aquí
    else:
        # Procesar datos del formulario HTML
        datos = {col: float(request.form[col]) for col in columnas}
        df = pd.DataFrame([datos])
        pred = modelo.predict(df)[0]
        prob = modelo.predict_proba(df)[0].tolist()
        resultado = {
            'prediccion': clasificadores[int(pred)],
            'probabilidades': [round(p * 100, 2) for p in prob],
            'datos': datos
        }
        return render_template('Formulario.html', resultado=resultado)

@app.route('/descargar_pdf', methods=['POST'])
def descargar_pdf():
    datos = request.form.to_dict()
    pred = datos.pop('prediccion')
    probabilidades = datos.pop('probabilidades')

    html = f"""
    <h1>Resultado de Predicción de Salud Fetal</h1>
    <p><strong>Predicción:</strong> {pred}</p>
    <p><strong>Probabilidades:</strong> {probabilidades}</p>
    <h2>Datos del formulario:</h2>
    <ul>
        {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in datos.items())}
    </ul>
    """

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, download_name='resultado_prediccion.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
