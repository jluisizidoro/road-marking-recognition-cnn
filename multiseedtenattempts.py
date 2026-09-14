import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

print(">>> [Início] Configurando ambiente para Validação Multi-Seed (10 Tentativas)...", flush=True)

# =======================================================================
# CONFIGURAÇÕES DE DIRETÓRIOS E PARÂMETROS
# =======================================================================
BASE_DIR = 'C:/Users/joao.134617/Desktop/Artigo mobilidade autonoma/'
TRAIN_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'treino')
VAL_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'validacao')
MODEL_DIR = os.path.join(BASE_DIR, 'resultado')

os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_SHAPE = (320, 72, 3)
CLASS_NAMES = ['LFO2', 'LFO3', 'LMS2']

# 10 Sementes estatísticas para máxima robustez científica
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

# Melhores parâmetros definidos no Grid Search anterior para cada arquitetura
CONFIGS_MODELOS = [
    {
        "nome": "VGG16",
        "func": VGG16,
        "epochs": 50,
        "batch_size": 32
    },
    {
        "nome": "MobileNetV2",
        "func": MobileNetV2,
        "epochs": 100,
        "batch_size": 32
    },
    {
        "nome": "ResNet50",
        "func": ResNet50,
        "epochs": 50,
        "batch_size": 32
    }
]

resultados_multiseed = []

print("==================================================================")
print("     INICIANDO TREINAMENTO ESTATÍSTICO (10 SEEDS POR REDE)       ")
print("==================================================================", flush=True)

for config in CONFIGS_MODELOS:
    model_name = config["nome"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    
    for seed in SEEDS:
        print(f"\n--- [ {model_name} ] | Seed: {seed} | Épocas: {epochs} | Batch: {batch_size} ---", flush=True)
        
        tf.keras.utils.set_random_seed(seed)
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=INPUT_SHAPE[:2], batch_size=batch_size,
            class_mode='categorical', classes=CLASS_NAMES, shuffle=True, seed=seed
        )
        val_gen = val_datagen.flow_from_directory(
            VAL_DIR, target_size=INPUT_SHAPE[:2], batch_size=batch_size,
            class_mode='categorical', classes=CLASS_NAMES, shuffle=False
        )
        
        num_classes = len(train_gen.class_indices)
        steps_per_epoch = math.ceil(train_gen.samples / batch_size)
        validation_steps = math.ceil(val_gen.samples / batch_size)
        
        # Construção da arquitetura com Dropout (0.05)
        base_model = config["func"](weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.05)(x)
        output = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Treinamento
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            verbose=2
        )
        
        final_acc = history.history['val_accuracy'][-1]
        print(f"✅ [{model_name} - Seed {seed}] Acurácia de Validação: {final_acc:.4f}", flush=True)
        
        resultados_multiseed.append({
            "Modelo": model_name,
            "Epocas": epochs,
            "Batch": batch_size,
            "Seed": seed,
            "Acuracia_Val": final_acc
        })

# =======================================================================
# CONSOLIDAÇÃO E CÁLCULO ESTATÍSTICO COMPLETO
# =======================================================================
df_multiseed = pd.DataFrame(resultados_multiseed)
csv_path = os.path.join(MODEL_DIR, "resultado_multiseed_10tentativas.csv")
df_multiseed.to_csv(csv_path, index=False)

resumo_estatistico = df_multiseed.groupby("Modelo")["Acuracia_Val"].agg(
    Acuracia_Media='mean', 
    Desvio_Padrao='std', 
    Melhor_Acuracia='max', 
    Pior_Acuracia='min'
).reset_index()

# Convertendo para porcentagem para a tabela e o gráfico
resumo_estatistico['Acuracia_Media_Pct'] = (resumo_estatistico['Acuracia_Media'] * 100).round(2)
resumo_estatistico['Desvio_Padrao_Pct'] = (resumo_estatistico['Desvio_Padrao'] * 100).round(2)
resumo_estatistico['Melhor_Acuracia_Pct'] = (resumo_estatistico['Melhor_Acuracia'] * 100).round(2)
resumo_estatistico['Pior_Acuracia_Pct'] = (resumo_estatistico['Pior_Acuracia'] * 100).round(2)

print("\n\n==================================================================")
print("     ANÁLISE ESTATÍSTICA FINAL (10 TENTATIVAS - MÉDIA E DESVIO)")
print("==================================================================")
print(resumo_estatistico[['Modelo', 'Acuracia_Media_Pct', 'Desvio_Padrao_Pct', 'Melhor_Acuracia_Pct', 'Pior_Acuracia_Pct']].to_markdown(index=False))
print(f"\n📁 Dados detalhados salvos em: {csv_path}")

# =======================================================================
# GERAÇÃO DO GRÁFICO ESTATÍSTICO COM BARRAS DE ERRO
# =======================================================================
print("\n>>> Gerando gráfico estatístico com desvio padrão...", flush=True)

plt.figure(figsize=(8, 6))
modelos = resumo_estatistico['Modelo']
medias = resumo_estatistico['Acuracia_Media_Pct']
desvios = resumo_estatistico['Desvio_Padrao_Pct']

# Gráfico de barras com o desvio padrão calculado entre as 10 seeds
barras = plt.bar(modelos, medias, yerr=desvios, capsize=8, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.85, edgecolor='black', linewidth=1.2)

plt.ylim(0, 105)
plt.ylabel('Acurácia Média de Validação (%)', fontsize=11, fontweight='bold')
plt.title('Validação Estatística Multi-Seed (10 Tentativas: Média $\pm$ Desvio Padrão)', fontsize=12, pad=15, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Escreve o valor numérico exato em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura + 3.5, f'{altura:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
grafico_path = os.path.join(MODEL_DIR, "Grafico_MultiSeed_10Tentativas.png")
plt.savefig(grafico_path, dpi=300)
print(f"📈 Gráfico estatístico salvo com sucesso em:\n{grafico_path}")

plt.show()
print("==================================================================")