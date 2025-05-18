import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ============ CONFIGURACIÓN INICIAL ============
# Configuración de GPU (automática si está disponible)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU detectada y configurada")
    except RuntimeError as e:
        print(f"⚠️ Error al configurar GPU: {e}")

# ============ FUNCIÓN PARA CARGAR EL DATASET ============
def load_dataset(data_path="../dataset", img_size=(128, 128), test_size=0.2, batch_size=32, max_samples_per_class=None):
    """
    Carga el dataset optimizado para evitar problemas de memoria.
    
    Args:
        data_path (str): Ruta al dataset.
        img_size (tuple): Tamaño de redimensionamiento (alto, ancho).
        test_size (float): Porcentaje para datos de prueba.
        batch_size (int): Tamaño del lote para procesamiento.
        max_samples_per_class (int): Límite de imágenes por clase (opcional).
    
    Returns:
        train_dataset (tf.data.Dataset): Datos de entrenamiento.
        test_dataset (tf.data.Dataset): Datos de prueba.
        class_names (list): Nombres de las clases.
    """
    # Definición de clases
    classes = {
        'papel': 0,
        'organico': 1,
        'metal': 2,
        'plastico': 3
    }
    
    print("\n=== SMART TRASH - DATA PREPROCESSING ===")
    print(f"🔍 Buscando dataset en: {os.path.abspath(data_path)}")
    print(f"🖼️ Tamaño de imagen: {img_size}")
    print(f"📦 Máximo de imágenes por clase: {max_samples_per_class or 'Sin límite'}\n")

    # Verificación de estructura del dataset
    missing_folders = [cls for cls in classes if not os.path.exists(os.path.join(data_path, cls))]
    if missing_folders:
        raise FileNotFoundError(f"❌ Carpetas faltantes: {missing_folders}")

    # Paso 1: Recopilar rutas de imágenes y etiquetas
    image_paths = []
    labels = []
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_path, class_name)
        class_count = 0
        
        # Recorrer todas las subcarpetas
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if max_samples_per_class and class_count >= max_samples_per_class:
                        break
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)
                    class_count += 1
        
        print(f"📂 {class_name}: {class_count} imágenes")
        if class_count == 0:
            print(f"   ⚠️ ¿Las imágenes están en subcarpetas? Ej: '{class_name}/subcarpeta/imagen.png'")

    if not image_paths:
        raise ValueError("❌ No se encontraron imágenes válidas (.png/.jpg/.jpeg)")

    # Paso 2: Dividir en train/test (a nivel de rutas, no datos)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Paso 3: Crear un generador de datos eficiente (tf.data.Dataset)
    def preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = img / 255.0  # Normalización [0, 1]
        return img, label

    # Convertir etiquetas a one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=len(classes))
    test_labels = to_categorical(test_labels, num_classes=len(classes))

    # Crear datasets de TensorFlow (carga perezosa)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("\n✅ ¡Dataset cargado exitosamente!")
    print(f"📊 Estadísticas:")
    print(f" - Total imágenes: {len(image_paths)}")
    print(f" - Entrenamiento: {len(train_paths)}")
    print(f" - Prueba: {len(test_paths)}")
    print(f" - Tamaño de lote: {batch_size}")

    return train_dataset, test_dataset, list(classes.keys())

# ============ EJECUCIÓN PRINCIPAL ============
if __name__ == "__main__":
    try:
        # Configuración recomendada para evitar problemas de memoria
        train_data, test_data, class_names = load_dataset(
            data_path="../dataset",
            img_size=(64, 64),  # Reducir si hay problemas de RAM
            batch_size=16,
            max_samples_per_class=2000  # Opcional: limita imágenes por clase
        )

        # Ejemplo de modelo simple (para probar el dataset)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, validation_data=test_data, epochs=2)  # Prueba rápida

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        print("\n🔧 Posibles soluciones:")
        print("1. Verifica que todas las carpetas (papel, organico, metal, plastico) existen")
        print("2. Asegúrate de que las imágenes están en subcarpetas (ej: plastico/botellas/img1.png)")
        print("3. Reduce 'img_size' o 'max_samples_per_class' si hay problemas de RAM")
        print("4. Ejecuta desde la carpeta 'training/'")