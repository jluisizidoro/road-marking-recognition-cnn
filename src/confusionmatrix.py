import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# =======================================================================
# CONFIGURAÇÕES (Caminho atualizado para framesselecionados)
# =======================================================================
BASE_DIR = 'C:/Users/joao.134617/Desktop/Artigo mobilidade autonoma/'

# Apontando para a pasta correta onde estão as subpastas de validação
VAL_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'validacao') 
MODEL_DIR = os.path.join(BASE_DIR, 'resultado')

INPUT_SHAPE = (320, 72)
BATCH_SIZE = 32
CLASS_NAMES = ['LFO2', 'LFO3', 'LMS2']

print(f"Verificando diretório de validação: {VAL_DIR}")
if os.path.exists(VAL_DIR):
    subpastas = os.listdir(VAL_DIR)
    print(f"✅ Diretório encontrado! Subpastas (classes): {subpastas}")
else:
    print(f"❌ ATENÇÃO: O caminho {VAL_DIR} não foi localizado.")

MODELOS_PARA_AVALIAR = [
    {
        "arquivo": "VGG16_E50_B32_Dropout.h5", 
        "nome_amigavel": "VGG16 (E50, B32) - 91.22%"
    },
    {
        "arquivo": "MobileNetV2_E100_B32_Dropout.h5", 
        "nome_amigavel": "MobileNetV2 (E100, B32) - 85.96%"
    }
]

# Prepara o gerador de validação
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

if val_gen.samples == 0:
    print("\n❌ ERRO CRÍTICO: O gerador encontrou 0 imagens. Verifique se as pastas LFO2, LFO3 e LMS2 estão lá dentro.")
else:
    print(f"\n✅ Sucesso! Encontradas {val_gen.samples} imagens para avaliação.")
    y_true = val_gen.classes

    for item in MODELOS_PARA_AVALIAR:
        model_path = os.path.join(MODEL_DIR, item["arquivo"])
        
        print(f"\n========================================================")
        print(f" Avaliando modelo: {item['nome_amigavel']}")
        print(f"========================================================")
        
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}. Pulando...")
            continue
            
        model = tf.keras.models.load_model(model_path)
        
        pred_probabilities = model.predict(val_gen)
        y_pred = np.argmax(pred_probabilities, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, 
                    yticklabels=CLASS_NAMES, 
                    annot_kws={"size": 14, "weight": "bold"})
        
        plt.title(f'Matriz de Confusão - {item["nome_amigavel"]}', fontsize=11, pad=12, fontweight='bold')
        plt.xlabel('Classe Predita', fontsize=10, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        grafico_path = os.path.join(MODEL_DIR, f'Matriz_{item["arquivo"].replace(".h5", "")}.png')
        plt.savefig(grafico_path, dpi=300)
        print(f"📈 Gráfico salvo em: {grafico_path}")
        plt.close()

    print("\n✅ Processo finalizado com sucesso!")
