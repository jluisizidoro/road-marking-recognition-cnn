import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import time
import pandas as pd

# =======================================================================
# CONFIGURAÇÕES FIXAS
# =======================================================================
BASE_DIR = 'C:/Users/joao.134617/Desktop/Artigo mobilidade autonoma/'
TRAIN_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'treino') # imagens de treino
VAL_DIR = os.path.join(BASE_DIR, 'framesselecionados', 'validacao')     # imagens de validação
# Caminho confirmado onde os modelos são salvos
MODEL_DIR = 'C:/Users/joao.134617/Desktop/Artigo mobilidade autonoma/resultado'
INPUT_SHAPE = (320, 72, 3)
CLASS_NAMES = ['LFO2', 'LFO3', 'LMS2'] # Nomes corretos (Letra 'O')

os.makedirs(MODEL_DIR, exist_ok=True)
models_dict = {
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "VGG16": VGG16,
    "EfficientNetB0": EfficientNetB0,
}

# =======================================================================
# 🎯 FUNÇÃO PRINCIPAL DE TREINAMENTO
# =======================================================================
def treinar_modelo_transfer_learning(model_name, epochs_frozen, batch_size):
    
    start_time = time.time()
    
    print(f"\n=======================================================================================")
    print(f"  EXECUTANDO: {model_name} | Épocas: {epochs_frozen} | Batch: {batch_size}")
    print(f"=======================================================================================")

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_gen = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=INPUT_SHAPE[:2], batch_size=batch_size,
            class_mode='categorical', classes=CLASS_NAMES, shuffle=True
        )
        val_gen = val_datagen.flow_from_directory(
            VAL_DIR, target_size=INPUT_SHAPE[:2], batch_size=batch_size,
            class_mode='categorical', classes=CLASS_NAMES, shuffle=False
        )
        print(f"Encontradas {train_gen.samples} imagens de treino e {val_gen.samples} de validação.")
        
    except Exception as e:
        print(f"ERRO: Falha ao carregar dados. Verifique os caminhos e nomes de classe. Erro: {e}")
        return {
            'model': model_name, 'epochs': epochs_frozen, 'batch': batch_size, 
            'accuracy': 0.0, 'time': round(time.time() - start_time, 2), 'status': 'Falha no Carregamento'
        }

    num_classes = len(train_gen.class_indices)

    base_model_func = models_dict.get(model_name)
    if not base_model_func:
        print(f"ERRO: Modelo '{model_name}' não suportado.")
        return {}

    base_model = base_model_func(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE
    )

    base_model.trainable = False 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    
    # ===================================================================
    #  ADIÇÃO DA REGULARIZAÇÃO DE DROPOUT (5% de desativação)
    # ===================================================================
    x = Dropout(0.05)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\n--- Treinamento das Camadas DENSAS ({epochs_frozen} épocas com Dropout) ---")
    
    steps_per_epoch = max(1, train_gen.samples // batch_size)
    validation_steps_correto = (val_gen.samples + batch_size - 1) // batch_size
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_frozen,
        validation_data=val_gen,
        validation_steps=validation_steps_correto,
        verbose=1 
    )
    
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    final_model_name = f"{model_name}_E{epochs_frozen}_B{batch_size}_Dropout.h5"
    model_save_path = os.path.join(MODEL_DIR, final_model_name)
    model.save(model_save_path)
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    print(f"\n       ✅ CONCLUÍDO. Modelo salvo: {final_model_name}")
    print(f"       Acurácia Final de Validação: {final_val_accuracy:.4f}")
    print(f"       Duração: {duration} segundos.")
    print("-----------------------------------------------------------------------------------------")
    
    return {
        'model': model_name, 
        'epochs': epochs_frozen, 
        'batch': batch_size, 
        'accuracy': final_val_accuracy, 
        'time': duration,
        'status': 'Sucesso'
    }

# =======================================================================
# EXECUÇÃO DO GRID SEARCH (MODO "COMPLETO - 36 EXECUÇÕES")
# =======================================================================

if __name__ == '__main__':
    
    print("ATENÇÃO: Rodando 'GRID SEARCH COM DROPOUT' (36 execuções).")
    print("Redes: [MobileNetV2, ResNet50, VGG16, EfficientNetB0]")
    print("Épocas: [20, 50, 100] | Batch Sizes: [16, 32, 64]")
    
    ALL_NETS = ["MobileNetV2", "ResNet50", "VGG16", "EfficientNetB0"]
    EPOCHS_LIST = [20, 50, 100]
    BATCH_SIZE_LIST = [16, 32, 64]
    
    all_results = []
    start_total = time.time() 
    
    print(f"\n=======================================================================================")
    print("                INICIANDO O GRID SEARCH COM DROPOUT                ")
    print(f"=======================================================================================")
    
    for net in ALL_NETS:
        for epochs in EPOCHS_LIST:
            for batch in BATCH_SIZE_LIST:
                result = treinar_modelo_transfer_learning(net, epochs, batch)
                if result:
                    all_results.append(result)

    end_total = time.time()
    
    # --- RESUMO FINAL ---
    print("\n\n=======================================================================================")
    print("                 RESUMO DE RESULTADOS DO GRID SEARCH (COM DROPOUT)")
    print("=======================================================================================")
    
    print(f"Duração Total: {round(end_total - start_total, 2)} segundos.")
    print(f"Total de Execuções Concluídas: {len(all_results)}")
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        results_csv_path = os.path.join(MODEL_DIR, 'grid_search_results_dropout.csv')
        if os.path.exists(results_csv_path):
             os.remove(results_csv_path) 
        df_results.to_csv(results_csv_path, index=False)
        print(f"\nResultados salvos em: {results_csv_path}")
        
        print("\n** Tabela de Resultados por Combinação (Ordenada por Acurácia) **")
        print(df_results.sort_values(by='accuracy', ascending=False).to_markdown(index=False))
        
        best_run = df_results.loc[df_results['accuracy'].idxmax()]
        print("\n🥇 MELHOR MODELO ENCONTRADO (COM DROPOUT):")
        print(f"   Rede: {best_run['model']}")
        print(f"   Combinação: E{best_run['epochs']}_B{best_run['batch']}")
        print(f"   Acurácia de Validação: {best_run['accuracy']:.4f}")
        print(f"   Tempo de Treino: {best_run['time']} seg")
        
    print("=======================================================================================")
