from ultralytics import YOLO
import os
from Colors import Colors
import cv2
import numpy as np
import re
from easyocr import Reader
import pytesseract
from paddleocr import PaddleOCR
from openalpr import Alpr
from fast_plate_ocr import LicensePlateRecognizer


def detect_car(yolo_model, image_path, output_path):
    image = cv2.imread(image_path)
    if(image is None):
        Colors.error("Erro: Não foi possível carregar a imagem")
        exit()

    car_class_id = 2 # Classe do carro para limitar o tipo de detecções possíveis

    try:
        car_detection_model = YOLO(yolo_model)
    except Exception as e:
        Colors.error(f"Erro: Modelo não carregado, {e}")
        exit()

    # Executa o modelo geral para detectar o carro
    results = car_detection_model(
        source=image,
        classes=[car_class_id],
        conf=0.5,
    )

    # Verifica se há detecções de carros
    if len(results[0].boxes) == 0:
        Colors.error("Erro: Nenhum carro detectado na imagem")
        exit()

    if len(results[0].boxes) > 1:
        Colors.error("Erro: Múltiplos carros detectados na imagem")
        exit()

    # Se passou aqui tem especificamente 1 carro detectado
    car = results[0]

    # Pega o bounding box do carro
    xyxy = car.boxes.xyxy
    x1, y1, x2, y2 = map(int, xyxy[0].tolist())
    # Realiza o crop da imagem (salva somente os pixels do carro)
    car_cropped = image[y1:y2, x1:x2]

    # Salva a imagem do carro cortado para debug
    car_cropped_path = f"{output_path}/car_cropped_debug.jpg"
    cv2.imwrite(car_cropped_path, car_cropped)
    Colors.info(f"Imagem do carro salva como {car_cropped_path}")

    return car_cropped, car_cropped_path

def detect_plate(yolo_model, image, output_path):
    if image is None:
        Colors.error("Erro: Imagem não fornecida para detecção de placa")
        exit()
        
    if not isinstance(image, np.ndarray) or image.ndim != 3:  # Correção: deve ser != 3
        Colors.error("Erro: 'image' não é uma imagem válida ou foi corrompida.")
        Colors.error(f"Tipo: {type(image)}, Shape: {getattr(image, 'shape', 'N/A')}")
        exit()

    try:
        plate_detection_model = YOLO(yolo_model)
    except Exception as e:
        Colors.error(f"Erro: Modelo não carregado, {e}")
        exit()

    # Realiza a detecção da placa na imagem do carro
    plate_results = plate_detection_model(
        source=image,
        conf=0.3,
    )

    # Debug: mostra informações sobre as detecções
    print(f"Número de results: {len(plate_results)}")
    print(f"Número de boxes no primeiro result: {len(plate_results[0].boxes)}")
    print(f"Shape dos boxes: {plate_results[0].boxes.xyxy.shape}")

    # Verifica se há detecções de placas
    if len(plate_results[0].boxes) == 0:
        Colors.error("Erro: Nenhuma placa detectada na imagem")
        Colors.info("Verifique a imagem 'car_cropped_debug.jpg' para ver se o carro foi cortado corretamente")
        exit()

    if len(plate_results[0].boxes) > 1:
        Colors.warning("Aviso: Múltiplas placas detectadas, usando a primeira")

    # Se passou aqui tem pelo menos 1 placa detectada
    plate = plate_results[0]

    # Pega o bounding box da primeira placa
    xyxy = plate.boxes.xyxy
    x1, y1, x2, y2 = map(int, xyxy[0].tolist())
    # Realiza o crop da placa
    plate_cropped = image[y1:y2, x1:x2]

    plate_path = f"{output_path}/plate.png"
    cv2.imwrite(plate_path, plate_cropped)

    car_plate_detection_path = f"{output_path}/detection_result.jpg"
    plate.save(filename=car_plate_detection_path) # Salva a foto do carro com detecções

    Colors.success("Processamento concluído!")
    Colors.info("Arquivos salvos: plate.png, detection_result.jpg, car_cropped_debug.jpg")

    return plate_cropped, plate_path, car_plate_detection_path


def preprocess_plate_image_minimal(img):
    """
    Pré-processamento MÍNIMO: apenas escala de cinza e um leve blur Gaussiano.
    Otimizado para o EasyOCR (Deep Learning).
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Não usamos CLAHE ou Binarização para evitar perda de dados.
    return img_blur

def correct_and_validate_plate(placa_candidata, confidence, MERCUSUL_PATTERN, ANTIGA_PATTERN):
    """
    Aplica correções de OCR direcionadas e valida a placa (priorizando Mercosul).
    """
    placas_validas = []
    
    # 1. Tenta validação direta (sem correção) para Mercosul e Antigo
    if MERCUSUL_PATTERN.match(placa_candidata):
        placas_validas.append((placa_candidata, "Mercosul (Válida)", confidence))
        return placas_validas
        
    if ANTIGA_PATTERN.match(placa_candidata):
        # Passa por aqui, mas tentaremos correção Mercosul antes de aceitar.
        pass
        
    # 2. Prepara para a Correção (focando em Mercosul)
    corrigido_mercosul = list(placa_candidata)
    
    # === CORREÇÃO FOCADA EM MERCUSUL (AAA#A##) ===
    # Correção para NÚMEROS (índices 3, 5, 6)
    for i in [3, 5, 6]:
        if corrigido_mercosul[i] == 'I': corrigido_mercosul[i] = '1'
        elif corrigido_mercosul[i] in ['O', 'D']: corrigido_mercosul[i] = '0' 

    # Correção para LETRAS (índices 0, 1, 2, 4)
    for i in [0, 1, 2, 4]:
        if corrigido_mercosul[i] == '1': corrigido_mercosul[i] = 'I'
             
    texto_corrigido_mercosul = "".join(corrigido_mercosul)

    # 3. Revalida o resultado corrigido (Prioridade Mercosul)
    if MERCUSUL_PATTERN.match(texto_corrigido_mercosul):
        placas_validas.append((texto_corrigido_mercosul, "Mercosul (Corrigida)", confidence))
        return placas_validas

    # 4. Tenta validação no formato Antigo (com correção)
    corrigido_antiga = list(placa_candidata)
    
    # Correção para NÚMEROS (índices 3, 4, 5, 6)
    for i in [3, 4, 5, 6]:
        if corrigido_antiga[i] == 'I': corrigido_antiga[i] = '1' 
        elif corrigido_antiga[i] in ['O', 'D']: corrigido_antiga[i] = '0' 
    # Correção para LETRAS (índices 0, 1, 2)
    for i in [0, 1, 2]:
        if corrigido_antiga[i] == '1': corrigido_antiga[i] = 'I'
             
    texto_corrigido_antiga = "".join(corrigido_antiga)
    
    if ANTIGA_PATTERN.match(texto_corrigido_antiga):
        placas_validas.append((texto_corrigido_antiga, "Antiga (Corrigida)", confidence))
        return placas_validas

    return placas_validas

# --- Nova Função convert_plate_to_string ---
def convert_plate_to_string(plate):
    if plate is None or plate.ndim != 3: 
        Colors.error("Erro: Imagem de placa inválida.")
        return []
    
    m = LicensePlateRecognizer('cct-xs-v1-global-model')
    print(m.run(plate)[0].replace("_", ""))

    # # Definições
    # CHAR_LIST_PLACA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # MERCUSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
    # ANTIGA_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$')

    # # 1. PRÉ-PROCESSAMENTO MELHORADO
    # plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    


def init():
    print(os.path.abspath(os.getcwd()))
    plates_model = "../models/plate-model.pt" 
    yolov11_model = "../models/yolo11n.pt" 
    index = "4"
    image_path = f"../../images_test/image{index}.png" # Path da imagem
    output_path = f"results/debug{index}"


    if not os.path.exists(output_path):
        os.makedirs(output_path)
        Colors.info(f"Diretório criado: {output_path}")

    if not os.path.exists(image_path):
        Colors.error("Erro: Imagem não existe no path")
        exit()

    if not os.path.exists(yolov11_model) or not os.path.exists(plates_model):
        Colors.error("Erro: Modelos YoLo não existem no path")
        exit()

    car_image, car_image_path = detect_car(yolov11_model, image_path, output_path)
    plate, plate_path, car_plate_detection_path = detect_plate(plates_model, car_image, output_path)
    # plate = cv2.imread("results/debug2/plate.png")
    placas_encontradas = convert_plate_to_string(plate)
    
    if placas_encontradas:
        Colors.success(f"Placa(s) final(is) detectada(s): {placas_encontradas}")
    else:
        Colors.warning("Nenhuma placa válida encontrada após OCR e correção.")



if __name__ == '__main__':
    print("Iniciando detecção...")
    init()