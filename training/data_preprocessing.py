import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(data_path="../dataset", img_size=(224, 224), test_size=0.2):
    # Nombres EXACTOS de tus carpetas (como aparecen en ls)
    classes = {
        'papel': 0,
        'organico': 1,
        'metal': 2,
        'plastico': 3
    }
    
    X, y = [], []
    print("🔍 Iniciando carga del dataset...")

    # Verificación de estructura
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Directorio principal no encontrado: {os.path.abspath(data_path)}")

    for class_name, label in classes.items():
        class_dir = os.path.join(data_path, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"❌ Carpeta de clase '{class_name}' no encontrada en {data_path}")

        print(f"\n📂 Procesando: {class_name}")
        count = 0

        # Procesar todas las subcarpetas
        for item in os.listdir(class_dir):
            item_path = os.path.join(class_dir, item)
            
            if os.path.isdir(item_path):  # Es una subcarpeta
                for img_file in os.listdir(item_path):
                    img_path = os.path.join(item_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, img_size)
                            X.append(img)
                            y.append(label)
                            count += 1
                    except Exception as e:
                        print(f"⚠️ Error en {img_path}: {str(e)}")

        print(f"   → Imágenes procesadas: {count}")

    if len(X) == 0:
        raise ValueError("❌ No se encontraron imágenes válidas en las subcarpetas")

    # Conversión final
    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(np.array(y), num_classes=len(classes))
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

if __name__ == "__main__":
    try:
        print("\n=== Smart Trash - Data Preprocessing ===")
        DATA_DIR = "../dataset"  # Ruta relativa
        
        print(f"\nBuscando dataset en: {os.path.abspath(DATA_DIR)}")
        X_train, X_test, y_train, y_test = load_dataset(DATA_DIR)
        
        print("\n✅ ¡Dataset cargado exitosamente!")
        print(f"Total imágenes: {len(X_train) + len(X_test)}")
        print(f" - Entrenamiento: {len(X_train)}")
        print(f" - Prueba: {len(X_test)}")
        
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        print("\n🔧 Soluciones posibles:")
        print("1. Verifica que '../dataset' contiene exactamente estas carpetas:")
        print("   - papel/")
        print("   - organico/")
        print("   - metal/")
        print("   - plastico/")
        print("2. Cada carpeta principal debe contener subcarpetas con imágenes")
        print("3. Ejecuta desde la carpeta 'training/'")
