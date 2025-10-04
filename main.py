from ultralytics import YOLO
import os
from Colors import Colors
import cv2
import numpy as np
import re
from easyocr import Reader

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

def convert_plate_to_string(plate):
    if plate is None:
        Colors.error("Erro: Imagem não fornecida para detecção de placa")
        exit()
        
    if not isinstance(plate, np.ndarray) or plate.ndim != 3: 
        Colors.error("Erro: 'image' não é uma imagem válida ou foi corrompida.")
        Colors.error(f"Tipo: {type(plate)}, Shape: {getattr(plate, 'shape', 'N/A')}")
        exit()

    
    MERCUSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')# Padrão Mercosul (AAA0A00)
    ANTIGA_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$') # Padrão Antigo (AAA0000)
    
    # Cria uma cópia da imagem original para desenhar os bounding boxes
    plate_with_boxes = plate.copy()
    
    img_filtered = cv2.Canny(plate, 100, 200)
    cv2.imwrite(f"teste.png", img_filtered)

    CHAR_LIST_PLACA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    lista_idiomas = 'en,pt'
    idiomas = lista_idiomas.split(',')
    gpu = True

    reader = Reader(idiomas, gpu=gpu)
    detections = reader.readtext(img_filtered, allowlist=CHAR_LIST_PLACA)

    print(f"Número de detecções encontradas: {len(detections)}")
    
    for i, detection in enumerate(detections):
        bbox, text, score = detection
        
        


def init():
    plates_model = "plate-model.pt" 
    yolov11_model = "yolo11n.pt" 
    index = "2"
    image_path = f"../images_test/image{index}.png" # Path da imagem
    output_path = f"../results/debug{index}"


    if not os.path.exists(output_path):
        os.makedirs(output_path)
        Colors.info(f"Diretório criado: {output_path}")

    if not os.path.exists(image_path):
        Colors.error("Erro: Imagem não existe no path")
        exit()

    if not os.path.exists(yolov11_model) or not os.path.exists(plates_model):
        Colors.error("Erro: Modelos YoLo não existem no path")
        exit()

    # car_image, car_image_path = detect_car(yolov11_model, image_path, output_path)
    # plate, plate_path, car_plate_detection_path = detect_plate(plates_model, car_image, output_path)
    plate = cv2.imread("results/debug2/plate.png")
    convert_plate_to_string(plate)


if __name__ == '__main__':
    print("Iniciando detecção...")
    init()