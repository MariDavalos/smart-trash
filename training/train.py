import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_arch import build_model, save_model
from data_preprocessing import load_dataset
from evaluation import plot_training_metrics
import os
import numpy as np

# Configuración
EPOCHS = 50
BATCH_SIZE = 32
MODEL_NAME = "trash_classifier_v1"
DATA_PATH = "dataset"  # Ruta a tu dataset

def train_model():
    # 1. Cargar y preprocesar datos
    print("\n🔍 Cargando dataset...")
    X_train, X_test, y_train, y_test, class_names = load_dataset(DATA_PATH)
    
    # 2. Construir modelo
    print("\n🛠️ Construyendo modelo...")
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(class_names))
    model.summary()

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f"saved_models/best_{MODEL_NAME}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    # 4. Entrenamiento
    print("\n🚀 Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Guardar modelo final
    print("\n💾 Guardando modelo...")
    save_model(model, model_name=MODEL_NAME)
    
    # 6. Evaluación
    print("\n📊 Evaluando modelo...")
    plot_training_metrics(history)
    
    # Mostrar accuracy final
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Accuracy final en test: {test_acc*100:.2f}%")

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Ejecutar entrenamiento
    train_model()
    
    print("\n🎉 ¡Entrenamiento completado!")
    print(f"Modelo guardado en: saved_models/{MODEL_NAME}.h5")