from flask import current_app
import numpy as np
from src.app.Colors import Colors
import cv2
import os
import uuid
from fast_plate_ocr import LicensePlateRecognizer

def detect_car(car_detection_model, image_path):
    image = cv2.imread(image_path)
    if(image is None):
        Colors.error("Erro: Não foi possível carregar a imagem")
        exit()

    car_class_id = 2 # Classe do carro para limitar o tipo de detecções possíveis

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

    # # Salva a imagem do carro cortado para debug
    # car_cropped_path = f"{output_path}/{unique_id}-car_cropped_debug.jpg"
    # cv2.imwrite(car_cropped_path, car_cropped)
    # Colors.info(f"Imagem do carro salva como {car_cropped_path}")

    return car_cropped


def detect_plate(plate_detection_model, image, output_path, unique_id):
    if image is None:
        Colors.error("Erro: Imagem não fornecida para detecção de placa")
        exit()
        
    if not isinstance(image, np.ndarray) or image.ndim != 3:  # Correção: deve ser != 3
        Colors.error("Erro: 'image' não é uma imagem válida ou foi corrompida.")
        Colors.error(f"Tipo: {type(image)}, Shape: {getattr(image, 'shape', 'N/A')}")
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

    # plate_path = f"{output_path}/{unique_id}-plate.png"
    # cv2.imwrite(plate_path, plate_cropped)

    # car_plate_detection_path = f"{output_path}/{unique_id}-detection_result.jpg"
    # plate.save(filename=car_plate_detection_path) # Salva a foto do carro com detecções

    # Colors.success("Processamento concluído!")
    # Colors.info(f"Arquivos salvos: {plate_path}, {car_plate_detection_path}")

    return plate_cropped

def convert_plate_to_string(plate, ocr_model):
    if plate is None or plate.ndim != 3: 
        Colors.error("Erro: Imagem de placa inválida.")
        return []
    
    # gray_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    # _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    resized_plate = cv2.resize(plate, (128, 64))
    # Adiciona dimensão de batch: (1, 64, 128, 3)
    input_image = np.expand_dims(resized_plate, axis=0)
    
    return ocr_model.run(input_image)[0].replace("_", "")

    # return ocr_model.run(thresh_image)[0].replace("_", "")


def process_plate(file):
    model = current_app.config['YOLO']
    plate_model = current_app.config['YOLO_PLATE']
    plate_ocr = current_app.config['PLATE_OCR']
    upload_folder = current_app.config['UPLOAD_FOLDER']
    
    # Salva o arquivo temporariamente no upload_folder
    unique_id = str(uuid.uuid4())
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, f"{unique_id}_{file.filename}")
    file.save(file_path)
    
    output_path = "outputs"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Chama detect_car com o caminho do arquivo salvo
    car_image = detect_car(model, file_path)
    plate = detect_plate(plate_model, car_image, output_path, unique_id)
    string = convert_plate_to_string(plate, plate_ocr)

    # Pré-processa a imagem
    # img = preprocess_image(file)
    
    # # Faz a predição com o modelo
    # predictions = model.predict(img)
    
    # # Exemplo de retorno do formato de predição
    # result = {
    #     'predictions': predictions.tolist(),
    #     'class': np.argmax(predictions, axis=1).tolist()
    # }
    
    # return result
    return string or "N/A"