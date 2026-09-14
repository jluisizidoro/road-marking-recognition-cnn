import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# =======================================================================
# CONFIGURAÇÕES DO AMBIENTE
# =======================================================================
BASE_DIR = 'C:/Users/joao.134617/Desktop/Artigo mobilidade autonoma/'

# Diretório de validação / teste
TEST_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'validacao')  

IMAGE_SIZE = (320, 72)
BATCH_SIZE = 32
CLASS_NAMES = ['LFO2', 'LFO3', 'LMS2']

# 1. Carregar gerador de dados de teste (shuffle=False obrigatório)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False  
)

y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(f"\n✅ Total de imagens de teste carregadas: {test_generator.samples}")
print(f"✅ Classes mapeadas: {class_labels}")

# 2. Dicionário com todos os modelos treinados salvos na pasta 'resultado'
MODEL_DIR = os.path.join(BASE_DIR, 'resultado')

models_to_evaluate = {
    'VGG16': os.path.join(MODEL_DIR, 'VGG16_E50_B32_Dropout.h5'),
    'MobileNetV2': os.path.join(MODEL_DIR, 'MobileNetV2_E100_B32_Dropout.h5'),
    'ResNet50': os.path.join(MODEL_DIR, 'ResNet50_E50_B32_Dropout.h5')
}

# Loop para processar cada modelo individualmente
for model_name, model_path in models_to_evaluate.items():
    print(f"\n==================================================")
    print(f"       AVALIAÇÃO DETALHADA: {model_name}")
    print(f"==================================================")
    
    if not os.path.exists(model_path):
        print(f"❌ Arquivo do modelo não encontrado em: {model_path}. Pulando...")
        continue
        
    model = load_model(model_path)
    
    # Inferência no conjunto de teste
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Relatório de Classificação (Precision, Recall, F1-Score em dicionário)
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, digits=4, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report_str)
    
    # Matriz de Confusão formatada
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"Real_{c}" for c in class_labels], columns=[f"Pred_{c}" for c in class_labels])
    
    # Salvando Matriz em CSV
    cm_csv_path = os.path.join(MODEL_DIR, f'Matriz_Confusao_{model_name}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"📁 Matriz de confusão salva em: {cm_csv_path}")
    
    # ===================================================================
    # GERAÇÃO DO GRÁFICO DE BARRAS AGRUPADAS (Precision, Recall, F1-score)
    # ===================================================================
    precisions = [report_dict[cls]['precision'] for cls in class_labels]
    recalls = [report_dict[cls]['recall'] for cls in class_labels]
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_labels]
    
    x = np.arange(len(class_labels))
    width = 0.25  # Largura das barras
    
    plt.figure(figsize=(8, 5))
    
    # Barras agrupadas com cores semelhantes ao modelo de referência
    plt.bar(x - width, precisions, width, label='Precision', color='#3b6ea5', edgecolor='black', linewidth=0.6)
    plt.bar(x, recalls, width, label='Recall', color='#e08243', edgecolor='black', linewidth=0.6)
    plt.bar(x + width, f1_scores, width, label='F1-score', color='#4ca64c', edgecolor='black', linewidth=0.6)
    
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=11, fontweight='bold')
    plt.title(f'Performance por Classe - {model_name}', fontsize=12, pad=12, fontweight='bold')
    plt.xticks(x, class_labels, fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    plt.tight_layout()
    
    # Salvando o gráfico de barras por classe
    grafico_metricas_path = os.path.join(MODEL_DIR, f'Grafico_Metricas_{model_name}.png')
    plt.savefig(grafico_metricas_path, dpi=300)
    print(f"📈 Gráfico de métricas salvo em: {grafico_metricas_path}")
    plt.close()

print("\n==================================================")
print("✅ Avaliação de todos os modelos concluída com sucesso!")
print("==================================================")
