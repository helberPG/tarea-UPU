flask import Flask, request, render_template, jsonify, send_file
import joblib
import pandas as pd
import io
from xhtml2pdf import pisa
import ast

app = Flask(__name__)
from
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
    pred = datos.pop('prediccion', 'No disponible')
    probabilidades = datos.pop('probabilidades', 'No disponibles')

    # Convertir las probabilidades a lista
    try:
        prob_list = ast.literal_eval(probabilidades)
        prob_texto = ', '.join(f"{p:.2f}%" for p in prob_list)
    except:
        prob_texto = probabilidades

    # Extraer datos del paciente
    nombre = datos.pop('nombre', 'No especificado')
    email = datos.pop('email', 'No especificado')
    telefono = datos.pop('telefono', 'No especificado')

    recomendaciones_dict = {
        "Normal": {
            "descripcion": "El feto presenta signos vitales dentro de rangos saludables.",
            "recomendaciones": [
                "Continuar con los controles prenatales regulares.",
                "Mantener una dieta balanceada y hábitos saludables.",
                "Seguir las indicaciones médicas normales según el trimestre del embarazo.",
                "Evitar factores de riesgo como el estrés, el tabaco o el alcohol."
            ]
        },
        "Sospechoso": {
            "descripcion": "Hay alteraciones leves en los parámetros cardiotocográficos que podrían indicar problemas si se agravan.",
            "recomendaciones": [
                "Realizar un seguimiento más frecuente mediante ecografías o cardiotocografías adicionales.",
                "Consultar con un especialista en medicina fetal o gineco-obstetra.",
                "Evitar actividades físicas intensas y reposar según lo recomendado.",
                "Evaluar posibles causas como hipertensión gestacional, diabetes u otros factores maternos."
            ]
        },
        "Patológico": {
            "descripcion": "Se detectan anomalías significativas en el estado fetal, lo que podría implicar riesgo para el feto.",
            "recomendaciones": [
                "Acudir inmediatamente al centro de salud o a urgencias obstétricas.",
                "Realizar una evaluación clínica detallada: ecografía Doppler, monitoreo continuo, análisis de líquidos, etc.",
                "Evaluar la posibilidad de un parto inducido o cesárea si se confirma sufrimiento fetal.",
                "Supervisión médica constante en un entorno hospitalario."
            ]
        }
    }

    info = recomendaciones_dict.get(pred, {"descripcion": "No disponible", "recomendaciones": []})

    
    # Crear HTML organizado
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #333;
                margin-top: 30px;
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }}
            ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            li {{
                margin-bottom: 6px;
            }}
            .prediccion {{
                background-color: yellow;
                padding: 8px;
                font-weight: bold;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <h1>Resultado de Predicción de Salud Fetal</h1>

        <h2>Datos del Paciente</h2>
        <ul>
            <li><strong>Nombre:</strong> {nombre}</li>
            <li><strong>Correo electrónico:</strong> {email}</li>
            <li><strong>Teléfono:</strong> {telefono}</li>
        </ul>

        <h2>Datos del Formulario</h2>
        <ul>
            {''.join(f'<li><strong>{k.replace("_", " ").capitalize()}:</strong> {v}</li>' for k, v in datos.items())}
        </ul>

        <h2>Resultado de Predicción</h2>
        <p><span class="prediccion">Predicción: {pred}</span></p>
        <p><strong>Probabilidades:</strong> {prob_texto}</p>
        <h2>Descripción de la Clasificación</h2>
        <p>{info["descripcion"]}</p>

        <h2>Recomendaciones</h2>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in info["recomendaciones"])}
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

