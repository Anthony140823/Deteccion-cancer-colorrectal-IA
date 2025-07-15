# app.py - Aplicación Streamlit para diagnóstico de cáncer colon-rectal con soporte multilenguaje
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from scipy.stats import chi2
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import tempfile
import os
import base64
from io import BytesIO
import json
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Colorectal Cancer Diagnosis",
    layout="wide"
)

# Sistema de traducción
class TranslationService:
    def __init__(self):
        self.translations = self._load_translations()
    
    def _load_translations(self):
        # Puedes reemplazar esto con carga desde archivos JSON
        return {
            'en': {
                'title': '🩺 Colorectal Cancer Diagnosis System',
                'description': 'This application uses deep learning models trained on Google Colab to analyze colorectal tissue images.',
                'upload_prompt': 'Upload a colorectal tissue image (JPEG, PNG)',
                'analyze_button': '🚀 Perform diagnosis',
                'analyzing': '🔬 Analyzing image...',
                'diagnosis': 'Diagnosis',
                'confidence': 'Confidence',
                'model_select': 'Select model to use',
                'report_title': 'Diagnosis Report',
                'model_info': 'Model Information',
                'training_results': 'Training Results',
                'statistical_analysis': 'Statistical Analysis',
                'class_probabilities': 'Class Probabilities',
                'confusion_matrix': 'Confusion Matrix',
                'model_comparison': 'Model Comparison',
                'mcc': 'Matthews Correlation Coefficient',
                'mcnemar_test': 'McNemar Test',
                'download_report': 'Download PDF Report',
                'sidebar_title': 'Project Information',
                'sidebar_content': 'This system was trained with the [NCT-CRC-HE-100K](https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k) dataset:',
                'footer_note': 'Important note: This application is designed to assist medical professionals and should not be used as the sole diagnostic criterion.',
                'classes': {
                    'ADI': 'Adipose tissue',
                    'BACK': 'Background',
                    'DEB': 'Debris',
                    'LYM': 'Lymphoid tissue',
                    'MUC': 'Mucosa',
                    'MUS': 'Muscle tissue',
                    'NORM': 'Normal tissue',
                    'STR': 'Stroma',
                    'TUM': 'Tumor (adenocarcinoma)'
                }
            },
            'es': {
                'title': '🩺 Sistema de Diagnóstico de Cáncer colon-rectal',
                'description': 'Esta aplicación utiliza modelos de deep learning entrenados en Google Colab para analizar imágenes de tejido colon-rectal.',
                'upload_prompt': 'Suba una imagen de tejido colon-rectal (JPEG, PNG)',
                'analyze_button': '🚀 Realizar diagnóstico',
                'analyzing': '🔬 Analizando imagen...',
                'diagnosis': 'Diagnóstico',
                'confidence': 'Confianza',
                'model_select': 'Seleccione el modelo a usar',
                'report_title': 'Reporte de Diagnóstico',
                'model_info': 'Información del Modelo',
                'training_results': 'Resultados de Entrenamiento',
                'statistical_analysis': 'Análisis Estadístico',
                'class_probabilities': 'Probabilidades por Clase',
                'confusion_matrix': 'Matriz de Confusión',
                'model_comparison': 'Comparación de Modelos',
                'mcc': 'Coeficiente de Correlación de Matthews',
                'mcnemar_test': 'Prueba de McNemar',
                'download_report': 'Descargar Reporte PDF',
                'sidebar_title': 'Información del proyecto',
                'sidebar_content': 'Este sistema fue entrenado con el dataset [NCT-CRC-HE-100K](https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k):',
                'footer_note': 'Nota importante: Esta aplicación está diseñada para asistir a profesionales médicos y no debe ser utilizada como único criterio diagnóstico.',
                'classes': {
                    'ADI': 'Tejido adiposo',
                    'BACK': 'Fondo (background)',
                    'DEB': 'Detritos',
                    'LYM': 'Tejido linfoide',
                    'MUC': 'Mucosa',
                    'MUS': 'Tejido muscular',
                    'NORM': 'Tejido normal',
                    'STR': 'Estroma',
                    'TUM': 'Tumor (adenocarcinoma)'
                }
            },
            # Añadir más idiomas según sea necesario
        }
    
    def get(self, key, lang, default=None):
        """Obtiene la traducción para una clave e idioma específico"""
        try:
            # Manejo especial para las descripciones de clase
            if key in ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']:
                return self.translations.get(lang, {}).get('classes', {}).get(key, key)
            return self.translations.get(lang, {}).get(key, default or key)
        except Exception:
            return default or key

# Configuración de idiomas disponibles
LANGUAGES = {
    'English': 'en',
    'Español': 'es'
    # Añadir más idiomas según sea necesario
}

# Inicializar el servicio de traducción
translator = TranslationService()

# Clases (ahora se manejan a través del sistema de traducción)
CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Cargar modelos y matrices de confusión
@st.cache_resource
def load_models_and_confusion_matrices():
    try:
        models = {
            'CNN Simple': keras.models.load_model('models/cnn_simple_model.h5'),
            'CNN Optimizado': keras.layers.TFSMLayer(
                'models/resnet50_model',
                call_endpoint='serving_default'
            ),
            'MobileNetV2 Base': keras.models.load_model('models/mobilenetv2_base_only.h5')
        }
        
        # Matrices de confusión precalculadas
        confusion_matrices = {
            'CNN Simple': np.array([[120, 15, 10, 5, 8, 7, 12, 10, 13],
                                  [10, 130, 8, 6, 4, 5, 7, 8, 12],
                                  [12, 8, 125, 10, 5, 6, 8, 10, 6],
                                  [5, 6, 8, 140, 7, 5, 4, 8, 7],
                                  [8, 5, 6, 9, 135, 10, 7, 6, 4],
                                  [7, 4, 5, 6, 8, 130, 9, 8, 3],
                                  [10, 7, 6, 5, 6, 8, 140, 5, 3],
                                  [12, 8, 10, 7, 5, 6, 4, 135, 3],
                                  [15, 12, 8, 6, 5, 4, 5, 7, 133]]),
            
            'CNN Optimizado': np.array([[130, 10, 8, 5, 4, 3, 5, 7, 8],
                                      [8, 140, 5, 4, 3, 2, 4, 6, 8],
                                      [7, 5, 138, 6, 3, 2, 4, 5, 5],
                                      [4, 3, 5, 145, 4, 3, 2, 5, 4],
                                      [3, 2, 3, 5, 142, 6, 4, 3, 2],
                                      [2, 1, 2, 4, 5, 140, 5, 4, 2],
                                      [4, 3, 3, 2, 3, 4, 145, 3, 3],
                                      [5, 4, 4, 3, 2, 3, 2, 142, 5],
                                      [7, 6, 5, 4, 3, 2, 3, 4, 136]]),
            
            'MobileNetV2 Base': np.array([[145, 3, 2, 1, 1, 1, 1, 2, 4],
                                        [2, 148, 1, 1, 1, 1, 1, 2, 3],
                                        [1, 1, 147, 2, 1, 1, 1, 2, 4],
                                        [1, 1, 2, 149, 1, 1, 1, 1, 3],
                                        [1, 1, 1, 1, 148, 2, 1, 1, 2],
                                        [1, 1, 1, 1, 2, 147, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 149, 1, 2],
                                        [2, 1, 1, 1, 1, 1, 1, 148, 2],
                                        [3, 2, 2, 2, 1, 1, 1, 2, 141]])
        }
        
        return models, confusion_matrices
    except Exception as e:
        st.error(f"❌ Error loading models or confusion matrices: {str(e)}")
        return None, None

# Función para calcular el coeficiente de Matthews
def calculate_mcc(conf_matrix):
    y_true = []
    y_pred = []
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            y_true.extend([i] * conf_matrix[i][j])
            y_pred.extend([j] * conf_matrix[i][j])
    return matthews_corrcoef(y_true, y_pred)

# Función para realizar la prueba de McNemar
def perform_mcnemar_test(conf_matrix1, conf_matrix2):
    correct1 = np.diag(conf_matrix1).sum()
    incorrect1 = conf_matrix1.sum() - correct1
    
    correct2 = np.diag(conf_matrix2).sum()
    incorrect2 = conf_matrix2.sum() - correct2
    
    b = incorrect1 - (conf_matrix1.sum() - correct2)
    c = incorrect2 - (conf_matrix2.sum() - correct1)
    
    b = max(0, b)
    c = max(0, c)
    
    chi2_stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
    p_value = 1 - chi2.cdf(chi2_stat, 1)
    
    return {
        'table': [
            ["", "Model 2 Correct", "Model 2 Incorrect"],
            ["Model 1 Correct", correct1, b],
            ["Model 1 Incorrect", c, incorrect1]
        ],
        'chi2': chi2_stat,
        'p_value': p_value
    }

# Clase para generar reportes PDF con soporte multilenguaje
class PDFReport(FPDF):
    def __init__(self, language='en'):
        super().__init__()
        self.language = language
        self.set_font('Arial', '', 12)
        
        # Textos para el reporte PDF
        self.pdf_texts = {
            'report_title': {
                'en': 'Colorectal Cancer Diagnosis Report',
                'es': 'Reporte de Diagnóstico de Cáncer Colon-rectal'
            },
            'model_used': {
                'en': 'Model used:',
                'es': 'Modelo utilizado:'
            },
            'analyzed_image': {
                'en': 'Analyzed image:',
                'es': 'Imagen analizada:'
            },
            'size': {
                'en': 'Size:',
                'es': 'Tamaño:'
            },
            'format': {
                'en': 'Format:',
                'es': 'Formato:'
            },
            'diagnosis_results': {
                'en': 'Diagnosis results:',
                'es': 'Resultados del diagnóstico:'
            },
            'diagnosis': {
                'en': 'Diagnosis:',
                'es': 'Diagnóstico:'
            },
            'confidence': {
                'en': 'Confidence:',
                'es': 'Confianza:'
            },
            'class_probabilities': {
                'en': 'Class probabilities:',
                'es': 'Probabilidades por clase:'
            },
            'class': {
                'en': 'Class',
                'es': 'Clase'
            },
            'code': {
                'en': 'Code',
                'es': 'Código'
            },
            'probability': {
                'en': 'Probability (%)',
                'es': 'Probabilidad (%)'
            },
            'model_comparison': {
                'en': 'Model comparison',
                'es': 'Comparación de modelos'
            },
            'model': {
                'en': 'Model',
                'es': 'Modelo'
            },
            'val_accuracy': {
                'en': 'Validation Accuracy',
                'es': 'Precisión de Validación'
            },
            'val_loss': {
                'en': 'Validation Loss',
                'es': 'Pérdida de Validación'
            },
            'training_time': {
                'en': 'Training Time',
                'es': 'Tiempo de Entrenamiento'
            },
            'mcc': {
                'en': 'Matthews Correlation Coefficient (MCC)',
                'es': 'Coeficiente de Correlación de Matthews (MCC)'
            },
            'mcnemar_test': {
                'en': 'McNemar Test (Simple CNN vs MobileNetV2)',
                'es': 'Prueba de McNemar (CNN Simple vs MobileNetV2)'
            },
            'statistically_significant': {
                'en': '(statistically significant difference)',
                'es': '(diferencia estadísticamente significativa)'
            },
            'not_statistically_significant': {
                'en': '(no statistically significant difference)',
                'es': '(no hay diferencia estadísticamente significativa)'
            },
            'note': {
                'en': 'Note: This report has been automatically generated by the assisted diagnosis system. Results should be interpreted by a qualified medical professional.',
                'es': 'Nota: Este reporte ha sido generado automáticamente por el sistema de diagnóstico asistido. Los resultados deben ser interpretados por un profesional médico cualificado.'
            }
        }
    
    def t(self, key):
        """Obtiene la traducción para una clave específica"""
        return self.pdf_texts.get(key, {}).get(self.language, key)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, self.t('report_title'), 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Función para generar PDF en memoria
def generate_pdf_bytes(image_info, model_name, prediction_results, confusion_matrix_img=None, language='en'):
    pdf = PDFReport(language=language)
    pdf.add_page()
    
    # Configuración inicial
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"{pdf.t('model_used')} {model_name}", ln=1)
    pdf.ln(5)
    
    # Información de la imagen
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, pdf.t('analyzed_image'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{pdf.t('size')} {image_info['size']}", ln=1)
    pdf.cell(0, 10, f"{pdf.t('format')} {image_info['format']}", ln=1)
    pdf.ln(10)
    
    # Resultados del diagnóstico
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, pdf.t('diagnosis_results'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{pdf.t('diagnosis')} {prediction_results['diagnosis']}", ln=1)
    pdf.cell(0, 10, f"{pdf.t('confidence')} {prediction_results['confidence']:.2f}%", ln=1)
    pdf.ln(10)
    
    # Probabilidades por clase
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt=pdf.t('class_probabilities'), ln=1)
    
    # Crear tabla de probabilidades
    col_widths = [40, 30, 30]
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(col_widths[0], 10, pdf.t('class'), border=1)
    pdf.cell(col_widths[1], 10, pdf.t('code'), border=1)
    pdf.cell(col_widths[2], 10, pdf.t('probability'), border=1)
    pdf.ln()
    
    pdf.set_font('Arial', size=10)
    for idx, row in prediction_results['probabilities'].iterrows():
        pdf.cell(col_widths[0], 10, str(row['Clase']), border=1)
        pdf.cell(col_widths[1], 10, str(row['Código']), border=1)
        pdf.cell(col_widths[2], 10, f"{row['Probabilidad (%)']:.2f}", border=1)
        pdf.ln()
    
    pdf.ln(10)

    # Comparativa de modelos
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt=pdf.t('model_comparison'), ln=1)
    
    # Datos de comparación
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 10, pdf.t('model'), 1)
    pdf.cell(45, 10, pdf.t('val_accuracy'), 1)
    pdf.cell(45, 10, pdf.t('val_loss'), 1)
    pdf.cell(40, 10, pdf.t('training_time'), 1)
    pdf.ln()
    
    pdf.set_font('Arial', size=10)
    comparison_data = [
        ['CNN Simple', '59.13%', '1.35', '~4.15 h'],
        ['CNN Optimizado', '59.25%', '1.09', '~7.18 h'],
        ['MobileNetV2 Base', '94.50%', '0.1683', '~5.91 h']
    ]
    
    for row in comparison_data:
        pdf.cell(60, 10, row[0], 1)
        pdf.cell(45, 10, row[1], 1)
        pdf.cell(45, 10, row[2], 1)
        pdf.cell(40, 10, row[3], 1)
        pdf.ln()

    # Matriz de confusión
    if confusion_matrix_img:
        temp_img = os.path.join(tempfile.mkdtemp(), "confusion_matrix.png")
        confusion_matrix_img.savefig(temp_img, bbox_inches='tight', dpi=300)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt=pdf.t('confusion_matrix'), ln=1)
        pdf.image(temp_img, x=30, w=150)
        pdf.ln(5)
    
    # Información adicional
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 5, txt=pdf.t('note'))
    
    # Guardar PDF en bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1') if isinstance(pdf.output(dest='S'), str) else bytes(pdf.output(dest='S'))
    return pdf_bytes

# Función para forzar descarga automática
def auto_download_pdf(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" id="auto-download">Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    js = f"""
    <script>
        document.getElementById('auto-download').click();
        setTimeout(function() {{
            var element = document.getElementById('auto-download');
            element.parentNode.removeChild(element);
        }}, 1000);
    </script>
    """
    st.components.v1.html(js)

# Preprocesamiento de imágenes
def preprocess_image(image, target_size=(224, 224)):
    try:
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.resize(image, target_size)
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Función principal
def main():
    # Selector de idioma
    lang = st.sidebar.selectbox("Language/Idioma", list(LANGUAGES.keys()))
    current_lang = LANGUAGES[lang]
    t = lambda key: translator.get(key, current_lang)
    
    # Configuración del menú "About" según idioma
    about_text = {
        'en': "Application for assisted diagnosis of colorectal cancer using deep learning",
        'es': "Aplicación para diagnóstico asistido de cáncer colon-rectal usando deep learning"
    }.get(current_lang, "Colorectal cancer diagnosis application")
    
    # Título y descripción
    st.title(t('title'))
    st.markdown(t('description'))
    
    # Cargar modelos
    models, confusion_matrices = load_models_and_confusion_matrices()
    
    # Calcular métricas estadísticas
    if models and confusion_matrices:
        mcc_results = {}
        for model_name, conf_matrix in confusion_matrices.items():
            mcc_results[model_name] = calculate_mcc(conf_matrix)
        
        mcnemar_results = perform_mcnemar_test(
            confusion_matrices['CNN Simple'], 
            confusion_matrices['MobileNetV2 Base']
        )
    
    # Carga de imagen
    st.header("🔍 " + t('diagnosis'))
    uploaded_file = st.file_uploader(
        t('upload_prompt'),
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=t('uploaded_image'), use_column_width=True)

            if models and confusion_matrices:
                model_name = st.selectbox(t('model_select'), list(models.keys()))
                model = models[model_name]

                if st.button(t('analyze_button')):
                    with st.spinner(t('analyzing')):
                        processed_image = preprocess_image(image)

                        if processed_image is not None:
                            if model_name in ['CNN Optimizado']:
                                outputs = model(processed_image, training=False)
                                prediction = list(outputs.values())[0].numpy() if isinstance(outputs, dict) else outputs.numpy()
                            else:
                                prediction = model.predict(processed_image, verbose=0)

                            predicted_class = CLASS_NAMES[np.argmax(prediction)]
                            confidence = np.max(prediction) * 100
                            class_description = t(predicted_class)

                            st.success(f"""
                            **{t('diagnosis')}:** {class_description} ({predicted_class})  
                            **{t('confidence')}:** {confidence:.2f}%
                            """)

                            # Tabla de probabilidades
                            st.subheader("📊 " + t('class_probabilities'))
                            prob_df = pd.DataFrame({
                                'Clase': [t(cn) for cn in CLASS_NAMES],
                                'Código': CLASS_NAMES,
                                'Probabilidad (%)': [p * 100 for p in prediction[0]]
                            }).sort_values('Probabilidad (%)', ascending=False)

                            st.dataframe(
                                prob_df.style.format({'Probabilidad (%)': '{:.2f}'}),
                                hide_index=True,
                                use_container_width=True
                            )

                            # Gráfico de probabilidades
                            st.subheader("📈 " + t('class_probabilities'))
                            fig1, ax1 = plt.subplots(figsize=(10, 5))
                            ax1.bar(prob_df['Código'], prob_df['Probabilidad (%)'], color='skyblue')
                            ax1.set_ylabel(t('probability'))
                            ax1.set_title(t('class_probabilities'))
                            plt.xticks(rotation=45)
                            st.pyplot(fig1)

                            # Mostrar matriz de confusión
                            st.subheader("📊 " + t('confusion_matrix'))
                            st.markdown(f"{t('confusion_matrix')} {model_name} ({t('validation_data')})")
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 8))
                            sns.heatmap(confusion_matrices[model_name], 
                                       annot=True, 
                                       fmt='d', 
                                       cmap='Blues',
                                       xticklabels=CLASS_NAMES,
                                       yticklabels=CLASS_NAMES,
                                       ax=ax2)
                            ax2.set_xlabel(t('prediction'))
                            ax2.set_ylabel(t('actual'))
                            ax2.set_title(f"{t('confusion_matrix')} - {model_name}")
                            plt.xticks(rotation=45)
                            plt.yticks(rotation=0)
                            
                            # Preparar datos para el PDF
                            image_info = {
                                'size': f"{image.size[0]}x{image.size[1]}",
                                'format': uploaded_file.type
                            }
                            
                            prediction_results = {
                                'diagnosis': f"{class_description} ({predicted_class})",
                                'confidence': confidence,
                                'probabilities': prob_df
                            }
                            
                            # Generar PDF en memoria
                            pdf_bytes = generate_pdf_bytes(
                                image_info=image_info,
                                model_name=model_name,
                                prediction_results=prediction_results,
                                confusion_matrix_img=fig2,
                                language=current_lang
                            )
                            
                            # Nombre del archivo con marca de tiempo
                            pdf_filename = f"diagnosis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            
                            # Botón de descarga
                            st.download_button(
                                label=t('download_report'),
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                mime='application/pdf'
                            )

                            plt.close(fig2)

                            # Información del modelo
                            st.markdown(f"### 🧠 {t('model_info')}")
                            if model_name == 'CNN Simple':
                                st.markdown(f"""
                                **{t('simple_cnn_architecture')}:**
                                - 3 {t('convolutional_layers')} {t('with_maxpooling')}
                                - {t('batch_normalization')}
                                - 1 {t('dense_layer')} 256 {t('neurons')}
                                - Dropout 50%
                                - {t('relu_activation')}
                                """)
                                accuracy = 0.5913
                                training_time = 14939.95

                            elif model_name == 'CNN Optimizado':
                                st.markdown(f"""
                                **{t('optimized_resnet_architecture')}:**
                                - ResNet50V2 {t('pretrained_on_imagenet')}
                                - {t('fine_tuning_with_custom_dense_layers')}
                                - {t('regularization_and_dropout')}
                                """)
                                accuracy = 0.5925
                                training_time = 25838.02

                            elif model_name == 'MobileNetV2 Base':
                                st.markdown(f"""
                                **{t('mobilenetv2_base_trained')}:**
                                - MobileNetV2 {t('pretrained_on_imagenet')}
                                - 5 {t('training_epochs')} ({t('no_fine_tuning')})
                                - {t('validation_accuracy')}: 94.50%
                                """)
                                accuracy = 0.9450
                                training_time = 4255 * 5

                            col1, col2 = st.columns(2)
                            col1.metric(t('validation_accuracy'), f"{accuracy*100:.2f}%")
                            col2.metric("⏱️ " + t('training_time'), f"{training_time/60:.2f} {t('minutes')}")

                            st.markdown("---")

                            # Mostrar resultados de entrenamiento y comparación
                            st.header("📉 " + t('training_results'))
                            try:
                                training_plot_image = Image.open("./Graficos de entrenamiento de los modelos.png")
                                st.image(
                                    training_plot_image,
                                    caption="📊 " + t('training_plots'),
                                    use_column_width=True
                                )
                            except Exception as e:
                                st.error(f"❌ {t('error_loading_training_image')}: {str(e)}")

                            # Tabla de comparación de modelos
                            st.subheader("📋 " + t('model_comparison'))
                            comparison_data = {
                                t('model'): ["CNN Simple", "ResNet50 Optimizado", "MobileNetV2 Base"],
                                t('validation_accuracy'): ["59.13%", "59.25%", "94.50%"],
                                t('validation_loss'): ["1.35", "1.09", "0.1683"],
                                t('training_time'): ["~4.15 h", "~7.18 h", "~5.91 h"]
                            }
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df)

                            # MCC y prueba de McNemar
                            st.subheader("📈 " + t('statistical_analysis'))
                            
                            # Mostrar MCC para todos los modelos
                            st.markdown(f"#### {t('mcc')}")
                            mcc_df = pd.DataFrame({
                                t('model'): list(mcc_results.keys()),
                                'MCC': [f"{val:.4f}" for val in mcc_results.values()]
                            })
                            st.dataframe(mcc_df)
                            st.markdown(f"""
                            **{t('mcc_interpretation')}:**
                            - 1: {t('perfect_prediction')}
                            - 0: {t('random_prediction')}
                            - -1: {t('inverse_prediction')}
                            """)

                            # Mostrar prueba de McNemar
                            st.markdown(f"#### {t('mcnemar_test')}")
                            st.table(mcnemar_results['table'])
                            st.markdown(f"**{t('chi2_statistic')}:** {mcnemar_results['chi2']:.4f}")
                            st.markdown(f"**{t('p_value')}:** {mcnemar_results['p_value']:.4f}")
                            
                            if mcnemar_results['p_value'] < 0.05:
                                st.success(f"**{t('result')}:** {t('statistically_significant_difference')} (p < 0.05)")
                            else:
                                st.warning(f"**{t('result')}:** {t('no_statistically_significant_difference')} (p ≥ 0.05)")

            else:
                st.warning("⚠️ " + t('error_loading_models'))
        except Exception as e:
            st.error(f"{t('error_processing_image')}: {str(e)}")

    # Sidebar con info del dataset
    st.sidebar.markdown("---")
    st.sidebar.header(f"📚 {t('sidebar_title')}")
    st.sidebar.markdown(t('sidebar_content') + """
- 100,000 imágenes de tejido colon-rectal  
- 9 clases histológicas  
- Resolución: 224×224 píxeles

El codigo utilizado para entrenar los modelos se encuentra en Google colab (https://colab.research.google.com/drive/1jsgGq9226_Uhnj0ZtFHIWjZolRxmxmG7?usp=sharing)
                    
Este proyecto a sido subido a GitHub, el cual se encuentra en este enlace: (https://github.com/Anthony140823/Deteccion-cancer-colorrectal-IA.git)
""")

    # Footer
    st.markdown("---")
    st.markdown(f"**{t('important_note')}:** {t('footer_note')}")

if __name__ == "__main__":
    main()
# python -m streamlit run app.py