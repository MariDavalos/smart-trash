import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from datetime import datetime

def plot_training_metrics(history):
    """
    Genera gráficos de accuracy y loss durante el entrenamiento.
    
    Args:
        history: Objeto History retornado por model.fit()
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Guardar gráficos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/training_metrics_{timestamp}.png')
    plt.close()

def generate_classification_report(model, X_test, y_test, class_names):
    """
    Genera y guarda un reporte de clasificación.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas reales
        class_names: Lista con nombres de las clases
    """
    # Predecir clases
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Generar reporte
    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names,
        output_dict=True
    )
    
    # Imprimir en consola
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names
    ))
    
    # Guardar reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/classification_report_{timestamp}.txt', 'w') as f:
        f.write(classification_report(
            y_true,
            y_pred_classes,
            target_names=class_names
        ))

def plot_confusion_matrix(model, X_test, y_test, class_names):
    """
    Genera y guarda una matriz de confusión.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas reales
        class_names: Lista con nombres de las clases
    """
    # Predecir clases
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Guardar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/confusion_matrix_{timestamp}.png')
    plt.close()

def save_model_performance(model, X_test, y_test, class_names):
    """
    Ejecuta todas las evaluaciones y guarda los resultados.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas reales
        class_names: Lista con nombres de las clases
    """
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    # Ejecutar evaluaciones
    generate_classification_report(model, X_test, y_test, class_names)
    plot_confusion_matrix(model, X_test, y_test, class_names)

if __name__ == "__main__":
    # Ejemplo de uso (debes cargar tus datos primero)
    from model_arch import build_model
    from data_preprocessing import load_dataset
    
    # Cargar datos y modelo (ejemplo)
    X_train, X_test, y_train, y_test = load_dataset('tu_dataset')
    model = build_model()
    model.load_weights('best_model.h5')  # O tu modelo entrenado
    
    # Nombres de las clases (ajusta según tu dataset)
    CLASS_NAMES = ['paper', 'metal', 'plastic', 'organic', 'glass', 'trash']
    
    # Evaluar modelo
    save_model_performance(model, X_test, y_test, CLASS_NAMES)