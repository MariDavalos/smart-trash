import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Configuración
DATA_PATH = "/home/lissetalfaro/Descargas/smart-trash-new/dataset"
IMG_SIZE = (64, 64)
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASSES = ['organico', 'papel', 'plastico','metal']  # Asegúrate de que coincidan con tus carpetas

def load_dataset(data_path):
    images = []
    labels = []
    
    # Mapeo de clases a números (0: organico, 1: papel, 2: plastico)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASSES)}
    
    print("🔍 Buscando imágenes en las categorías: organico, papel, plastico...")

    for class_name in CLASSES:
        class_dir = os.path.join(data_path, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️ ¡Cuidado! No se encontró la carpeta: {class_dir}")
            continue
            
        print(f"\n📂 Procesando: {class_name}...")
        img_count = 0

        # Buscar en TODAS las subcarpetas (como plastic_cup_lids/, newspaper/, etc.)
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(VALID_EXTS):
                    file_path = os.path.join(root, file)
                    try:
                        img = Image.open(file_path).convert('RGB').resize(IMG_SIZE)
                        img_array = np.array(img) / 255.0  # Normalizar
                        
                        images.append(img_array)
                        labels.append(class_to_idx[class_name])
                        img_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Error en {file_path}: {str(e)}")
        
        print(f"   ✅ Imágenes encontradas en {class_name}: {img_count}")

    if not images:
        raise ValueError("❌ No se encontraron imágenes. ¿Están en las carpetas correctas?")

    X = np.array(images)
    y = to_categorical(labels, num_classes=len(CLASSES))

    print(f"\n📊 Dataset cargado:")
    print(f"- Total imágenes: {len(X)}")
    print(f"- Distribución: Orgánico={np.sum(y[:,0])}, Papel={np.sum(y[:,1])}, Plástico={np.sum(y[:,2])}")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) + (CLASSES,)

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train_model():
    print("\n🚀 Iniciando entrenamiento...")
    try:
        X_train, X_test, y_train, y_test, classes = load_dataset(DATA_PATH)
        
        model = build_model(X_train.shape[1:], len(classes))
        model.summary()
        
        print("\n🔥 Entrenando... (Paciencia, esto puede tomar unos minutos)")
        history = model.fit(X_train, y_train,
                          epochs=15,
                          batch_size=32,
                          validation_data=(X_test, y_test))
        
        print("\n🎉 ¡Modelo entrenado! ✔️")
        print("   - Precisión en entrenamiento: {:.2f}%".format(history.history['accuracy'][-1]*100))
        print("   - Precisión en validación: {:.2f}%".format(history.history['val_accuracy'][-1]*100))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 ¿Qué verificar?")
        print("1. ¿Las carpetas 'organico', 'papel' y 'plastico' existen en", DATA_PATH, "?")
        print("2. ¿Hay imágenes dentro de ellas? Ejecuta:")
        print(f"   find {DATA_PATH} -type f | grep -iE '\.jpg|\.png|\.jpeg' | wc -l")

if __name__ == "__main__":
    train_model()