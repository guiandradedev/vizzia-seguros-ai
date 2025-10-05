import cv2
import numpy as np
import re
from Colors import Colors
import pytesseract
from easyocr import Reader

lista_idiomas = "en,pt"
idiomas = lista_idiomas.split(",")
gpu = True
kernel = np.ones((2, 2), np.uint8) 
reader = Reader(idiomas, gpu=gpu)

def convert_plate_to_string(plate):
    if plate is None or plate.ndim != 3: 
        Colors.error("Erro: Imagem de placa inválida.")
        return []

    # Definições
    CHAR_LIST_PLACA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    MERCUSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
    ANTIGA_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$')

    print("=== TESTANDO PADDLEOCR ===")
    
    all_texts = []
    
    # Tenta PaddleOCR com configuração mais simples
    try:
        from paddleocr import PaddleOCR
        
        # Configuração mais simples - removendo parâmetros problemáticos
        ocr = PaddleOCR(lang='en')  # Apenas o idioma
        
        result = ocr.ocr(plate)  # Volta para o método original ocr()
        
        print("Resultados PaddleOCR (estrutura completa):")
        print(f"Tipo do resultado: {type(result)}")
        print(f"Tamanho: {len(result) if result else 'None'}")
        
        if result:
            # PaddleOCR retorna: [[[bbox], [text, confidence]], ...]
            for page in result:  # Primeira camada (páginas)
                if page:  # Se não está vazio
                    for detection in page:  # Segunda camada (detecções)
                        try:
                            bbox = detection[0]  # Coordenadas
                            text_conf = detection[1]  # [texto, confiança]
                            
                            text = text_conf[0]
                            confidence = text_conf[1]
                            
                            print(f"  Texto: '{text}' (confiança: {confidence:.2f})")
                            
                            # Limpa o texto
                            clean_text = ''.join(char for char in str(text) if char.isalnum()).upper()
                            if len(clean_text) >= 3:
                                all_texts.append((clean_text, confidence, "PaddleOCR"))
                                print(f"  Texto limpo: '{clean_text}'")
                        
                        except Exception as e:
                            print(f"  Erro ao processar detecção PaddleOCR: {e}")
                            print(f"  Estrutura da detecção: {detection}")
        
        print(f"Textos coletados do PaddleOCR: {all_texts}")
        
    except Exception as e:
        print(f"Erro no PaddleOCR: {e}")
        print("Tentando métodos alternativos...")
        all_texts = []

    # Fallback para EasyOCR se PaddleOCR falhar
    if not all_texts:
        print("\n=== TESTANDO EASYOCR ===")
        try:
            # Pré-processa a imagem para melhorar OCR
            resized_plate = cv2.resize(plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
            
            # Aplica threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Salva versões para debug
            cv2.imwrite("debug_original.png", plate)
            cv2.imwrite("debug_resized.png", resized_plate)
            cv2.imwrite("debug_thresh.png", thresh)
            
            # Testa EasyOCR em diferentes versões
            images_to_test = [
                ("Original", plate),
                ("Redimensionada", resized_plate),
                ("Threshold", thresh)
            ]
            
            for name, img in images_to_test:
                try:
                    if len(img.shape) == 2:  # Imagem em escala de cinza
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = img
                    
                    resultados = reader.readtext(img_rgb, allowlist=CHAR_LIST_PLACA)
                    
                    print(f"\n--- EasyOCR {name} ---")
                    for (caixa, texto, prob) in resultados:
                        clean_text = ''.join(char for char in texto if char.isalnum()).upper()
                        if len(clean_text) >= 3:
                            print(f"  '{texto}' -> '{clean_text}' (confiança: {prob:.2f})")
                            all_texts.append((clean_text, prob, f"EasyOCR_{name}"))
                
                except Exception as e:
                    print(f"Erro no EasyOCR para {name}: {e}")
        
        except Exception as e:
            print(f"Erro geral no EasyOCR: {e}")

    # Fallback para Tesseract
    if not all_texts:
        print("\n=== TESTANDO TESSERACT ===")
        try:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            
            # Redimensiona para melhorar OCR
            resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            configs = [
                ("PSM 8", "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
                ("PSM 7", "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
                ("PSM 6", "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            ]
            
            for config_name, config in configs:
                try:
                    text = pytesseract.image_to_string(resized, config=config)
                    clean_text = ''.join(char for char in text if char.isalnum()).upper()
                    if len(clean_text) >= 3:
                        print(f"  {config_name}: '{clean_text}'")
                        all_texts.append((clean_text, 0.7, f"Tesseract_{config_name}"))
                except Exception as e:
                    print(f"Erro no Tesseract {config_name}: {e}")
        
        except Exception as e:
            print(f"Erro geral no Tesseract: {e}")

    # Processa os resultados
    if all_texts:
        print(f"\n=== PROCESSANDO {len(all_texts)} RESULTADOS ===")
        
        valid_plates = []
        
        for text, confidence, source in all_texts:
            print(f"\nProcessando: '{text}' de {source}")
            
            # Aplica correções
            corrected_texts = apply_corrections(text)
            
            for corrected in corrected_texts:
                if len(corrected) == 7:
                    is_mercosul = MERCUSUL_PATTERN.match(corrected)
                    is_antiga = ANTIGA_PATTERN.match(corrected)
                    
                    if is_mercosul:
                        valid_plates.append((corrected, "Mercosul", confidence, source))
                        print(f"  ✅ VÁLIDA (Mercosul): {corrected}")
                    elif is_antiga:
                        valid_plates.append((corrected, "Antiga", confidence, source))
                        print(f"  ✅ VÁLIDA (Antiga): {corrected}")
                    else:
                        print(f"  ❌ Formato inválido: {corrected}")
                else:
                    print(f"  ⚠️  Tamanho incorreto: {corrected} ({len(corrected)} chars)")
        
        # Remove duplicatas e retorna melhores resultados
        if valid_plates:
            unique_plates = {}
            for plate, tipo, conf, fonte in valid_plates:
                if plate not in unique_plates or conf > unique_plates[plate][1]:
                    unique_plates[plate] = (tipo, conf, fonte)
            
            final_results = [(plate, tipo, conf) for plate, (tipo, conf, fonte) in unique_plates.items()]
            final_results.sort(key=lambda x: x[2], reverse=True)
            
            return final_results
        else:
            print("❌ Nenhuma placa válida encontrada após correções")
            return []
    else:
        print("❌ Nenhum texto detectado por qualquer método")
        return []

def apply_corrections(texto):
    """Aplica correções comuns de OCR"""
    corrections = [texto]  # Versão original
    
    # Mapeamentos de caracteres comuns
    char_map = {
        'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8', 'Z': '2',
        '0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B', '2': 'Z'
    }
    
    # Aplica correções globais
    for old_char, new_char in char_map.items():
        if old_char in texto:
            corrections.append(texto.replace(old_char, new_char))
    
    # Remove duplicatas
    return list(set(corrections))

def init():
    print("Iniciando teste de OCR...")
    
    # Testa ambas as imagens
    test_files = [
        "results/debug2/plate.png",
        "discord-placa.png"
    ]
    
    for file_path in test_files:
        print(f"\n{'='*60}")
        print(f"TESTANDO: {file_path}")
        print(f"{'='*60}")
        
        plate = cv2.imread(file_path)
        if plate is not None:
            placas_encontradas = convert_plate_to_string(plate)
            
            if placas_encontradas:
                print(f"\n🎉 MELHOR RESULTADO: {placas_encontradas[0][0]} ({placas_encontradas[0][1]})")
                print(f"Confiança: {placas_encontradas[0][2]:.2f}")
                
                if len(placas_encontradas) > 1:
                    print("\nOutros resultados:")
                    for i, (placa, tipo, conf) in enumerate(placas_encontradas[1:], 2):
                        print(f"  {i}. {placa} ({tipo}) - {conf:.2f}")
            else:
                print("\n❌ Nenhuma placa válida encontrada")
        else:
            print(f"❌ Erro ao carregar: {file_path}")

if __name__ == "__main__":
    init()