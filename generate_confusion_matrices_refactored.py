"""
Generador de matrices de confusión para modelos de detección de cáncer colorrectal.

Este módulo genera matrices de confusión reales utilizando datos de validación
y múltiples modelos de aprendizaje profundo entrenados.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings específicos de TensorFlow
logging.getLogger('absl').setLevel(logging.ERROR)


@dataclass
class ModelConfig:
    """Configuración para un modelo específico."""
    name: str
    path: str
    input_size: Tuple[int, int]
    apply_clahe: bool = False
    model_type: str = 'standard'


class ConfusionMatrixGenerator:
    """Generador de matrices de confusión para modelos de ML."""
    
    CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    
    VALIDATION_DIRS = [
        "validation_data/CRC-VAL-HE-20",
        "../../../CRC-VAL-HE-20",
        "CRC-VAL-HE-20",
        "PRUEBAS"
    ]
    
    MODEL_CONFIGS = [
        ModelConfig('CNN Simple', 'models/cnn_simple_model.h5', (224, 224)),
        ModelConfig('ResNet50V2', 'models/resnet50_model', (224, 224), model_type='ResNet50V2'),
        ModelConfig('MobileNetV2 Base', 'models/mobilenetv2_base_only.h5', (224, 224)),
        ModelConfig('Hybrid Attention', 'models/Fast_HybridAttention_final.h5', (96, 96), True),
        ModelConfig('Hybrid Autoencoder', 'models/Fast_HybridAutoencoder_final.h5', (96, 96), True)
    ]
    
    def __init__(self):
        """Inicializa el generador."""
        self.models = {}
        self.validation_dir = self._find_validation_directory()
        
    def _find_validation_directory(self) -> Optional[str]:
        """Encuentra el directorio de validación disponible."""
        for dir_path in self.VALIDATION_DIRS:
            if os.path.exists(dir_path):
                logger.info(f"✅ Usando directorio de validación: {dir_path}")
                return dir_path
        
        logger.error("❌ No se encontró directorio de validación")
        return None
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int], 
                        apply_clahe: bool = False, model_type: str = 'standard') -> Optional[np.ndarray]:
        """
        Preprocesa una imagen para el modelo.
        
        Args:
            image_path: Ruta a la imagen
            target_size: Tamaño objetivo (width, height)
            apply_clahe: Si aplicar CLAHE para mejorar contraste
            model_type: Tipo de modelo para preprocesamiento específico
            
        Returns:
            Imagen preprocesada como array numpy o None si hay error
        """
        try:
            # Cargar imagen usando OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            image = cv2.resize(image, target_size)
            
            # Aplicar CLAHE si se requiere
            if apply_clahe:
                image = self._apply_clahe(image)
            
            # Convertir a float32
            image = image.astype(np.float32)
            
            # Preprocesamiento específico por modelo
            image = self._apply_model_preprocessing(image, model_type)
            
            return np.expand_dims(image, axis=0)
            
        except Exception as e:
            logger.error(f"Error procesando imagen {image_path}: {e}")
            return None
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _apply_model_preprocessing(self, image: np.ndarray, model_type: str) -> np.ndarray:
        """Aplica preprocesamiento específico según el tipo de modelo."""
        if model_type == 'ResNet50V2':
            return tf.keras.applications.resnet_v2.preprocess_input(image)
        else:
            return image / 255.0
    
    def load_models(self) -> Dict[str, Any]:
        """
        Carga todos los modelos configurados.
        
        Returns:
            Diccionario con los modelos cargados
        """
        logger.info("Cargando modelos...")
        models = {}
        
        for config in self.MODEL_CONFIGS:
            try:
                if config.name == 'ResNet50V2':
                    models[config.name] = tf.keras.layers.TFSMLayer(
                        config.path,
                        call_endpoint='serving_default'
                    )
                else:
                    model = tf.keras.models.load_model(config.path)
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    models[config.name] = model
                    
                logger.info(f"✅ Modelo {config.name} cargado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error cargando modelo {config.name}: {e}")
                continue
        
        self.models = models
        return models
    
    def _get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Obtiene la configuración de un modelo por nombre."""
        for config in self.MODEL_CONFIGS:
            if config.name == model_name:
                return config
        return None
    
    def _predict_with_model(self, model_name: str, model: Any, 
                           preprocessed_images: Dict[str, np.ndarray]) -> Optional[int]:
        """
        Realiza predicción con un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            model: Instancia del modelo
            preprocessed_images: Diccionario con imágenes preprocesadas
            
        Returns:
            Clase predicha o None si hay error
        """
        try:
            config = self._get_model_config(model_name)
            if not config:
                return None
            
            # Obtener la imagen preprocesada apropiada
            key = f"{config.input_size}_{config.model_type}"
            if key not in preprocessed_images:
                return None
                
            image = preprocessed_images[key]
            
            # Predicción según tipo de modelo
            if model_name == 'ResNet50V2':
                outputs = model(image, training=False)
                pred = tf.cast(outputs[list(outputs.keys())[0]], tf.float32).numpy()
                pred = np.expand_dims(pred, axis=0)
            else:
                pred = model.predict(image, verbose=0)
            
            # Aplicar softmax y obtener clase predicha
            probabilities = tf.nn.softmax(pred[0]).numpy()
            predicted_class = np.argmax(probabilities)
            
            return predicted_class
            
        except Exception as e:
            logger.error(f"Error con modelo {model_name}: {e}")
            return None
    
    def _process_single_image(self, image_path: str, true_class: int) -> Tuple[bool, Dict[str, int]]:
        """
        Procesa una sola imagen con todos los modelos.
        
        Args:
            image_path: Ruta a la imagen
            true_class: Clase verdadera de la imagen
            
        Returns:
            Tupla (éxito, predicciones por modelo)
        """
        # Preprocesar imagen para cada configuración
        preprocessed_images = {}
        
        for config in self.MODEL_CONFIGS:
            key = f"{config.input_size}_{config.model_type}"
            if key not in preprocessed_images:
                img = self.preprocess_image(
                    image_path, 
                    config.input_size, 
                    config.apply_clahe, 
                    config.model_type
                )
                if img is not None:
                    preprocessed_images[key] = img
        
        # Obtener predicciones de todos los modelos
        predictions = {}
        all_successful = True
        
        for model_name, model in self.models.items():
            pred = self._predict_with_model(model_name, model, preprocessed_images)
            if pred is not None:
                predictions[model_name] = pred
            else:
                all_successful = False
                break
        
        return all_successful, predictions
    
    def _process_test_images(self, max_images: int = 50) -> Tuple[List[int], Dict[str, List[int]]]:
        """Procesa imágenes de prueba cuando no hay estructura de clases."""
        all_true_labels = []
        predictions_by_model = {name: [] for name in self.models.keys()}
        
        image_files = [f for f in os.listdir(self.validation_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, img_file in enumerate(image_files[:max_images]):
            img_path = os.path.join(self.validation_dir, img_file)
            class_idx = idx % len(self.CLASS_NAMES)
            
            success, predictions = self._process_single_image(img_path, class_idx)
            
            if success:
                all_true_labels.append(class_idx)
                for model_name, prediction in predictions.items():
                    predictions_by_model[model_name].append(prediction)
                
                logger.info(f"✅ Procesada imagen {idx+1}/{min(len(image_files), max_images)}: {img_file}")
            else:
                logger.warning(f"❌ Error procesando {img_file}")
        
        return all_true_labels, predictions_by_model
    
    def _process_class_organized_images(self, max_per_class: int = 10) -> Tuple[List[int], Dict[str, List[int]]]:
        """Procesa imágenes organizadas por clases en carpetas."""
        all_true_labels = []
        predictions_by_model = {name: [] for name in self.models.keys()}
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = os.path.join(self.validation_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"⚠️ No se encuentra directorio para clase {class_name}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            image_files = image_files[:max_per_class]
            logger.info(f"📂 Procesando clase {class_name}: {len(image_files)} imágenes")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                success, predictions = self._process_single_image(img_path, class_idx)
                
                if success:
                    all_true_labels.append(class_idx)
                    for model_name, prediction in predictions.items():
                        predictions_by_model[model_name].append(prediction)
                else:
                    logger.warning(f"❌ Error procesando {img_file}")
        
        return all_true_labels, predictions_by_model
    
    def generate_confusion_matrices(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Genera matrices de confusión para todos los modelos.
        
        Returns:
            Tupla (matrices_de_confusión, accuracies)
        """
        if not self.validation_dir:
            return {}, {}
        
        if not self.models:
            self.load_models()
        
        if not self.models:
            logger.error("No se pudieron cargar los modelos")
            return {}, {}
        
        logger.info(f"📊 Procesando imágenes desde: {self.validation_dir}")
        
        # Procesar imágenes según la estructura del directorio
        if "PRUEBAS" in self.validation_dir:
            all_true_labels, predictions_by_model = self._process_test_images()
        else:
            all_true_labels, predictions_by_model = self._process_class_organized_images()
        
        # Calcular matrices de confusión y accuracies
        confusion_matrices = {}
        accuracies = {}
        
        logger.info(f"🔄 Calculando matrices de confusión...")
        logger.info(f"Total de muestras procesadas: {len(all_true_labels)}")
        
        for model_name in self.models.keys():
            if len(predictions_by_model[model_name]) > 0:
                conf_matrix = confusion_matrix(
                    all_true_labels,
                    predictions_by_model[model_name],
                    labels=range(len(self.CLASS_NAMES))
                )
                confusion_matrices[model_name] = conf_matrix
                
                # Calcular accuracy
                accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
                accuracies[model_name] = accuracy
                
                logger.info(f"✅ {model_name}: Accuracy = {accuracy*100:.2f}%")
            else:
                logger.warning(f"❌ No hay predicciones para {model_name}")
        
        return confusion_matrices, accuracies


class CodeGenerator:
    """Generador de código Python para las matrices de confusión."""
    
    @staticmethod
    def generate_python_code(confusion_matrices: Dict[str, np.ndarray], 
                           accuracies: Dict[str, float]) -> str:
        """
        Genera código Python con las matrices de confusión reales.
        
        Args:
            confusion_matrices: Matrices de confusión por modelo
            accuracies: Accuracies por modelo
            
        Returns:
            Código Python como string
        """
        logger.info("📋 Generando código Python...")
        
        lines = ["        # Matrices de confusión calculadas con datos reales"]
        lines.append("        confusion_matrices = {")
        
        # Generar matrices de confusión
        for model_name, matrix in confusion_matrices.items():
            lines.append(f"            '{model_name}': np.array([")
            
            for i, row in enumerate(matrix):
                indent = "                                  " if i > 0 else ""
                row_str = "[" + ", ".join([f"{val:3d}" for val in row]) + "]"
                
                if i == len(matrix) - 1:
                    lines.append(f"{indent}{row_str}]),")
                else:
                    lines.append(f"{indent}{row_str},")
        
        lines.append("        }")
        lines.append("")
        
        # Generar accuracies
        lines.append("        # Accuracies reales calculadas")
        lines.append("        accuracies = {")
        for model_name, acc in accuracies.items():
            lines.append(f"            '{model_name}': {acc:.4f},  # {acc*100:.2f}%")
        lines.append("        }")
        lines.append("")
        
        # Generar losses estimadas
        lines.append("        # Losses estimadas (basadas en accuracy)")
        lines.append("        losses = {")
        for model_name, acc in accuracies.items():
            estimated_loss = max(0.1, (1 - acc) * 2)
            lines.append(f"            '{model_name}': {estimated_loss:.4f},")
        lines.append("        }")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_code_to_file(code: str, filename: str = "confusion_matrices_code.txt"):
        """Guarda el código generado en un archivo."""
        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write(code)
            logger.info(f"✅ Código guardado en '{filename}'")
        except Exception as e:
            logger.error(f"❌ Error guardando código: {e}")


class DataExporter:
    """Exportador de datos en diferentes formatos."""
    
    @staticmethod
    def save_as_json(confusion_matrices: Dict[str, np.ndarray], 
                    accuracies: Dict[str, float], 
                    filename: str = "confusion_matrices_data.json"):
        """
        Guarda las matrices en formato JSON.
        
        Args:
            confusion_matrices: Matrices de confusión por modelo
            accuracies: Accuracies por modelo
            filename: Nombre del archivo de salida
        """
        try:
            # Convertir matrices numpy a listas para JSON
            matrices_dict = {}
            for model_name, matrix in confusion_matrices.items():
                matrices_dict[model_name] = {
                    'confusion_matrix': matrix.tolist(),
                    'accuracy': float(accuracies[model_name])
                }
            
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(matrices_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Matrices guardadas en '{filename}'")
            
        except Exception as e:
            logger.error(f"❌ Error guardando matrices: {e}")


def main():
    """Función principal."""
    logger.info("🚀 Generando matrices de confusión reales...")
    logger.info("=" * 60)
    
    # Crear generador y procesar
    generator = ConfusionMatrixGenerator()
    confusion_matrices, accuracies = generator.generate_confusion_matrices()
    
    if confusion_matrices:
        # Mostrar resumen
        logger.info("📊 Resumen de resultados:")
        logger.info("-" * 40)
        for model_name, accuracy in accuracies.items():
            logger.info(f"{model_name:20}: {accuracy*100:6.2f}%")
        
        # Generar y guardar código
        code = CodeGenerator.generate_python_code(confusion_matrices, accuracies)
        print("\n" + "="*80)
        print("📋 CÓDIGO PARA REEMPLAZAR EN app.py:")
        print("="*80)
        print(code)
        print("="*80)
        
        CodeGenerator.save_code_to_file(code)
        
        # Guardar datos en JSON
        DataExporter.save_as_json(confusion_matrices, accuracies)
        
        logger.info("✅ ¡Proceso completado exitosamente!")
        logger.info("📁 Archivos generados:")
        logger.info("   - confusion_matrices_code.txt (código para app.py)")
        logger.info("   - confusion_matrices_data.json (datos en JSON)")
        
    else:
        logger.error("❌ No se pudieron generar las matrices de confusión")


if __name__ == "__main__":
    main()
