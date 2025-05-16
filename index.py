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
    try:
        datos = request.get_json()
        if not datos:
            return jsonify({'error': 'No se recibieron datos en formato JSON'}), 400

        # Convertir a float y crear DataFrame
        datos = {col: float(datos[col]) for col in columnas}
        df = pd.DataFrame([datos])

        # Predicción y probabilidades
        pred = modelo.predict(df)[0]
        prob = modelo.predict_proba(df)[0].tolist()

        return jsonify({
            'prediccion': clasificadores[int(pred)],
            'probabilidades': [round(p * 100, 2) for p in prob]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/descargar_pdf', methods=['POST'])
def descargar_pdf():
    datos = request.form.to_dict()
    pred = datos.pop('prediccion')
    probabilidades = datos.pop('probabilidades')  # Probablemente llega como string tipo '[26.32, 26.32, 47.36]'

    # Convierte probabilidades de string a lista y luego a texto separado por comas
    import ast
    try:
        prob_list = ast.literal_eval(probabilidades)
        prob_texto = ', '.join(f"{p:.2f}%" for p in prob_list)
    except:
        prob_texto = probabilidades  # fallback si no se puede parsear

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            h1 {{
                text-align: center;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Resultado de Predicción de Salud Fetal</h1>
        <p><strong>Predicción:</strong> {pred}</p>
        <p><strong>Probabilidades:</strong> {prob_texto}</p>
        <h2>Datos del formulario:</h2>
        <ul>
            {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in datos.items())}
        </ul>
    </body>
    </html>
    """

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, download_name='resultado_prediccion.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
