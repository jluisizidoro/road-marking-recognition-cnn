import cv2
import os
import glob

# =======================================================================
# CONFIGURAÇÕES DE DIRETÓRIOS
# =======================================================================
# Altere para a pasta da classe que deseja processar
PASTA_ENTRADA = "C:/Users/joao.134617/Desktop/LFO2 novo"   

# Pasta de saída para as imagens cortadas (sem distorção de proporção)
PASTA_SAIDA = "C:/Users/joao.134617/Desktop/LFO2 corte 2" 

os.makedirs(PASTA_SAIDA, exist_ok=True)

# Pega todas as imagens JPG da pasta de entrada
imagens = glob.glob(os.path.join(PASTA_ENTRADA, "*.jpg"))

print(f"Iniciando o corte de {len(imagens)} imagens (Removendo 3/5 superiores)...")

contador = 0

for caminho_img in imagens:
    # Lê a imagem original
    img = cv2.imread(caminho_img)
    
    if img is None:
        continue
        
    altura_original, largura_original, _ = img.shape
    
    # --- O CORTE DE 3/5 SUPERIORES ---
    # Calculamos onde fica 3/5 (60%) da altura total da imagem.
    # Tudo acima desse ponto (o céu e o horizonte) será descartado.
    ponto_corte = int(altura_original * (3 / 5))
    
    # Mantemos da linha de corte (3/5) até o final da imagem (5/5) e toda a largura
    img_cortada = img[ponto_corte:altura_original, 0:largura_original]
    
    # Nota: Como você pediu para não redimensionar para manter a resolução/proporção,
    # salvamos a imagem diretamente com a altura reduzida (2/5 da original) e a largura intacta.
    
    # Nome do arquivo de saída
    nome_arquivo = os.path.basename(caminho_img)
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo)
    
    # Salva a imagem cortada preservando a proporção original
    cv2.imwrite(caminho_saida, img_cortada, [cv2.IMWRITE_JPEG_QUALITY, 95])
    contador += 1

print(f"✅ Processo concluído! {contador} imagens cortadas e salvas em: {PASTA_SAIDA}")
