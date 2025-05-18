import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import os

def build_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Construye una CNN para clasificación de residuos con capas personalizables
    
    Args:
        input_shape: Tupla (height, width, channels)
        num_classes: Número de categorías de residuos
        
    Returns:
        Modelo de Keras compilado
    """
    model = Sequential([
        # Bloque 1
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Bloque 2
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Bloque 3
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Clasificación
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilación con learning rate adaptable
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def save_model(model, model_dir="saved_models", model_name="trash_classifier"):
    """
    Guarda el modelo en formato .h5 y guarda la arquitectura como imagen
    
    Args:
        model: Modelo entrenado
        model_dir: Directorio para guardar
        model_name: Nombre base para los archivos
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Guardar imagen de la arquitectura
    plot_path = os.path.join(model_dir, f"{model_name}_architecture.png")
    plot_model(model, to_file=plot_path, show_shapes=True)
    print(f"Model architecture plot saved to {plot_path}")

def load_model(model_path):
    """
    Carga un modelo pre-entrenado desde disco
    
    Args:
        model_path: Ruta al archivo .h5
        
    Returns:
        Modelo de Keras cargado
    """
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    # Ejemplo de uso
    model = build_model()
    model.summary()
    
    # Guardar modelo de ejemplo (comentar si no necesario)
    save_model(model)
    
    # Visualización de la arquitectura
    try:
        plot_model(model, to_file='model_architecture.png', show_shapes=True)
        print("Architecture plot generated!")
    except ImportError:
        print("Install graphviz to generate architecture plot:")
        print("sudo apt-get install graphviz && pip install pydot")