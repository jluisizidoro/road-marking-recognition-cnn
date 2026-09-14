import os

# ===========================
# Configuração
# ===========================
# Caminho da pasta onde estão os frames salvos
PASTA_FRAMES = "C:/Users/joao.134617/Desktop/LFO2 novo" 

# Define a taxa de seleção (1 a cada X frames). 
# Exemplo: 10 significa que vai manter o 0, 10, 20, 30... e apagar os outros.
INTERVALO = 10

if not os.path.exists(PASTA_FRAMES):
    print(f"Erro: A pasta não foi encontrada: {PASTA_FRAMES}")
else:
    arquivos = sorted(os.listdir(PASTA_FRAMES))
    imagens = [f for f in arquivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_antes = len(imagens)
    removidos = 0
    mantidos = 0

    for i, arquivo in enumerate(imagens):
        # Se o índice do arquivo não for múltiplo do intervalo, o arquivo é apagado
        if i % INTERVALO != 0:
            caminho_arquivo = os.path.join(PASTA_FRAMES, arquivo)
            os.remove(caminho_arquivo)
            removidos += 1
        else:
            mantidos += 1

    print("=== Limpeza concluída! ===")
    print(f"Total de imagens antes: {total_antes}")
    print(f"Imagens mantidas: {mantidos}")
    print(f"Imagens apagadas: {removidos}")