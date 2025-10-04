import cv2
import numpy as np
import re
from easyocr import Reader

# Caracteres válidos
CHAR_LIST_PLACA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Padrões Regex para o Brasil
# Padrão Mercosul (AAA0A00)
MERCUSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
# Padrão Antigo (AAA0000)
ANTIGA_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$') 

lista_idiomas = 'en,pt'
idiomas = lista_idiomas.split(',')
gpu = True

# O resto do seu setup do EasyOCR...
img = cv2.imread('results/debug2/plate.png')

img_filtered = cv2.Canny(img, 100, 200)

cv2.imwrite(f"teste.png", car_cropped)


reader = Reader(idiomas, gpu=gpu)
result = reader.readtext(img_filtered, allowlist=CHAR_LIST_PLACA)

placas_validas = []

for (bbox, text, confidence) in result:
    texto_limpo = ''.join(filter(str.isalnum, text)).upper()
    
    # Ignorar resultados que não têm 7 caracteres
    if len(texto_limpo) != 7 or confidence < 0.60: # Diminuí o threshold de confiança para 0.60
        continue
        
    placa_candidata = texto_limpo
    
    # --- LÓGICA PRINCIPAL: TENTAR CORREÇÃO E VALIDAÇÃO ---
    
    # 1. Tenta validar no formato Mercosul primeiro (o seu caso)
    if MERCUSUL_PATTERN.match(placa_candidata):
        placas_validas.append((placa_candidata, "Mercosul (Válida)", confidence))
        continue
        
    # 2. Tenta validar no formato Antigo
    if ANTIGA_PATTERN.match(placa_candidata):
        placas_validas.append((placa_candidata, "Antiga (Válida)", confidence))
        continue

    # 3. Se nenhuma validação funcionou, aplica correções e testa novamente
    
    corrigido_mercosul = list(placa_candidata)
    corrigido_antiga = list(placa_candidata)
    
    # Mapeamento de confusões comuns: 'O' ou 'D' <-> '0' (zero)
    
    # Tenta correção no PADRÃO MERCOSUL (AAA#A##)
    # A correção só precisa focar nas posições de NÚMEROS (índices 3, 5, 6)
    for i in [3, 5, 6]:
        if corrigido_mercosul[i] in ['O', 'D']:
             corrigido_mercosul[i] = '0' 
    
    # Tenta correção no PADRÃO ANTIGO (AAA####)
    # A correção só precisa focar nas posições de NÚMEROS (índices 3, 4, 5, 6)
    for i in [3, 4, 5, 6]:
        if corrigido_antiga[i] in ['O', 'D']:
             corrigido_antiga[i] = '0' 

    # Converte de volta para string
    texto_corrigido_mercosul = "".join(corrigido_mercosul)
    texto_corrigido_antiga = "".join(corrigido_antiga)

    # 4. Revalida após a correção
    if MERCUSUL_PATTERN.match(texto_corrigido_mercosul):
        # Ex: FNO DCO3 (OCR) -> FNO 0C03 (Corrigido)
        placas_validas.append((texto_corrigido_mercosul, "Mercosul (Corrigida)", confidence))
        
    elif ANTIGA_PATTERN.match(texto_corrigido_antiga):
        # Ex: ABC 100D (OCR) -> ABC 1000 (Corrigido)
        placas_validas.append((texto_corrigido_antiga, "Antiga (Corrigida)", confidence))
        
print(f"Resultado final: {placas_validas}")
print("\nResultados brutos do OCR:")
print(result)