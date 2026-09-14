import cv2
import os

# ===========================
# Configurações
# ===========================
VIDEO = "C:/Users/joao.134617/Desktop/Video Project 200.mp4"  
PASTA_SAIDA = "C:/Users/joao.134617/Desktop/LFO2 novo"  

LARGURA = 320
ALTURA = 180

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ===========================
# Leitura do vídeo
# ===========================
cap = cv2.VideoCapture(VIDEO)

contador = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Redimensiona
    frame = cv2.resize(
        frame,
        (LARGURA, ALTURA),
        interpolation=cv2.INTER_AREA
    )

    # Salva JPG com qualidade 95
    cv2.imwrite(
        os.path.join(PASTA_SAIDA, f"LMS22teste_frame_{contador:06d}.jpg"),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    contador += 1

cap.release()

print(f"{contador} frames salvos.")
