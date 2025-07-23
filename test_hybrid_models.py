import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Función para preprocesar imagen
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Cargar modelos
print("Cargando modelos...")
try:
    hybrid_attention = tf.keras.models.load_model('models/Fast_HybridAttention_final.h5')
    hybrid_autoencoder = tf.keras.models.load_model('models/Fast_HybridAutoencoder_final.h5')
    cnn_simple = tf.keras.models.load_model('models/cnn_simple_model.h5')
    print("Todos los modelos cargados exitosamente")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    exit()

# Buscar imágenes de prueba
test_images = []
pruebas_dir = "PRUEBAS"
if os.path.exists(pruebas_dir):
    test_images = [os.path.join(pruebas_dir, f) for f in os.listdir(pruebas_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]  # Solo las primeras 3

if not test_images:
    print("No se encontraron imágenes de prueba en la carpeta PRUEBAS")
    exit()

print(f"Encontradas {len(test_images)} imágenes de prueba")

# Probar cada imagen con cada modelo
for img_path in test_images:
    print(f"\n--- Procesando {os.path.basename(img_path)} ---")
    
    try:
        # Preprocesar para diferentes tamaños
        img_224 = preprocess_image(img_path, (224, 224))
        img_96 = preprocess_image(img_path, (96, 96))
        
        # Predicciones CNN Simple (224x224)
        pred_cnn = cnn_simple.predict(img_224, verbose=0)
        print(f"CNN Simple - Predicción: {np.argmax(pred_cnn[0])}, Confianza: {np.max(pred_cnn[0]):.4f}")
        print(f"CNN Simple - Distribución: {pred_cnn[0]}")
        
        # Predicciones Hybrid Attention (96x96)
        pred_attention = hybrid_attention.predict(img_96, verbose=0)
        print(f"Hybrid Attention - Predicción: {np.argmax(pred_attention[0])}, Confianza: {np.max(pred_attention[0]):.4f}")
        print(f"Hybrid Attention - Distribución: {pred_attention[0]}")
        
        # Predicciones Hybrid Autoencoder (96x96)
        pred_autoencoder = hybrid_autoencoder.predict(img_96, verbose=0)
        print(f"Hybrid Autoencoder - Predicción: {np.argmax(pred_autoencoder[0])}, Confianza: {np.max(pred_autoencoder[0]):.4f}")
        print(f"Hybrid Autoencoder - Distribución: {pred_autoencoder[0]}")
        
        # Verificar si las predicciones de los modelos híbridos son uniformes (indicativo de problema)
        attention_std = np.std(pred_attention[0])
        autoencoder_std = np.std(pred_autoencoder[0])
        print(f"Std Attention: {attention_std:.6f}, Std Autoencoder: {autoencoder_std:.6f}")
        
        if attention_std < 0.01:
            print("⚠️ ALERTA: Hybrid Attention tiene predicciones muy uniformes!")
        if autoencoder_std < 0.01:
            print("⚠️ ALERTA: Hybrid Autoencoder tiene predicciones muy uniformes!")
            
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")

print("\n--- Resumen ---")
print("Si los modelos híbridos muestran predicciones uniformes (todas las probabilidades muy similares),")
print("esto indica que los modelos no fueron entrenados correctamente o tienen problemas de arquitectura.")
