from flask import current_app
import numpy as np
from src.app.Colors import Colors
import cv2
import os
import uuid
from fast_plate_ocr import LicensePlateRecognizer
from src.app.utils.err_api import ErrAPI

def detect_car(car_detection_model, image_path):
    image = cv2.imread(image_path)
    if(image is None):
        Colors.error("Erro: Não foi possível carregar a imagem")
        raise ErrAPI("Imagem nula ou inválida")

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
        raise ErrAPI("Nenhum carro detectado na imagem")

    if len(results[0].boxes) > 1:
        Colors.error("Erro: Múltiplos carros detectados na imagem")
        raise ErrAPI("Múltiplos carros detectados na imagem")

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
        raise ErrAPI("Imagem nula")
        
    if not isinstance(image, np.ndarray) or image.ndim != 3:  # Correção: deve ser != 3
        Colors.error("Erro: 'image' não é uma imagem válida ou foi corrompida.")
        Colors.error(f"Tipo: {type(image)}, Shape: {getattr(image, 'shape', 'N/A')}")
        raise ErrAPI("Imagem inválida")

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
        raise ErrAPI("Nenhuma placa detectada na imagem", 422)

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
    """Run OCR on a plate image and return the predicted string and a confidence score.

    The function asks the OCR model to return per-character confidences when possible
    and computes an overall confidence as the mean confidence of the predicted (non-blank)
    character slots. Returns a tuple: (predicted_string, confidence_float)
    """
    if plate is None or plate.ndim != 3:
        Colors.error("Erro: Imagem de placa inválida.")
        return ('', 0.0)

    resized_plate = cv2.resize(plate, (128, 64))
    # Adiciona dimensão de batch: (1, H, W, C)
    input_image = np.expand_dims(resized_plate, axis=0)

    try:
        # Request confidences as well
        res = ocr_model.run(input_image, return_confidence=True)
    except TypeError:
        # Fallback if model doesn't accept return_confidence (old API)
        strings = ocr_model.run(input_image)
        pred = strings[0].replace("_", "") if strings else ''
        return (pred, 0.0)
    except Exception as e:
        Colors.error(f"OCR error: {e}")
        return ('', 0.0)

    # res is expected to be a tuple: (list_of_strings, confidences_array)
    if isinstance(res, tuple) and len(res) == 2:
        strings, confs = res
    else:
        # Unexpected shape, try to parse as only strings
        strings = res if isinstance(res, list) else []
        confs = None

    pred_raw = strings[0] if strings else ''
    pred = pred_raw.replace("_", "")

    plate_confidence = 0.0
    try:
        if confs is not None:
            # confs shape: (N, plate_slots)
            slot_conf = np.array(confs)[0]  # first image
            # Build mask of slots that are not blank according to raw prediction
            mask = [ch != '_' for ch in list(pred_raw)]
            if any(mask):
                used = slot_conf[mask]
                # defensive: if used is empty fallback to mean of all slots
                if used.size > 0:
                    plate_confidence = float(np.mean(used))
                else:
                    plate_confidence = float(np.mean(slot_conf))
            else:
                # No non-blank slots (unexpected) -> use mean of all
                plate_confidence = float(np.mean(slot_conf))
        else:
            plate_confidence = 0.0
    except Exception:
        plate_confidence = 0.0

    # Clamp and ensure float
    try:
        plate_confidence = float(np.clip(plate_confidence, 0.0, 1.0))
    except Exception:
        plate_confidence = 0.0

    return (pred, plate_confidence)

    # return ocr_model.run(thresh_image)[0].replace("_", "")

def classify_color(color_model, image):
    yolo_result = color_model(source=image, conf=0.2)

    result = yolo_result[0]

    # Extract top1 index and confidence safely
    try:
        top1_idx = int(result.probs.top1)
    except Exception:
        # if it's a tensor or numpy array
        try:
            top1_idx = int(result.probs.top1.cpu().numpy())
        except Exception:
            # fallback: try as python int
            top1_idx = int(result.probs.top1)

    try:
        top1_conf = float(result.probs.top1conf.cpu().numpy())
    except Exception:
        try:
            top1_conf = float(result.probs.top1conf)
        except Exception:
            top1_conf = 0.0

    # English -> Portuguese mapping based on confusion matrix labels
    color_map_pt = {
        'beige': 'Bege',
        'black': 'Preto',
        'blue': 'Azul',
        'brown': 'Marrom',
        'gold': 'Dourado',
        'green': 'Verde',
        'grey': 'Cinza',
        'orange': 'Laranja',
        'pink': 'Rosa',
        'purple': 'Roxo',
        'red': 'Vermelho',
        'silver': 'Prata',
        'white': 'Branco',
        'yellow': 'Amarelo',
        'background': 'N/A'
    }

    # Get the english name from model.names (safely)
    try:
        eng_name = color_model.names[top1_idx]
    except Exception:
        # fallback: if names is dict-like, try direct access
        try:
            eng_name = color_model.names[str(top1_idx)]
        except Exception:
            eng_name = str(top1_idx)

    eng_name_lower = eng_name.lower().strip()
    pt_name = color_map_pt.get(eng_name_lower, eng_name)

    return [pt_name, top1_conf]

def detect_brand(brand_detector, car_image):
    yolo_result = brand_detector(source=car_image, conf=0.3)

    result = yolo_result[0]

    print(f"Número de boxes no brand detector: {len(result.boxes)}")

    if len(result.boxes) == 0:
        return 'N/A'

    # Pega a primeira detecção
    box = result.boxes[0]
    class_id = int(box.cls.cpu().numpy())

    # Mapeia o ID da classe para o nome da marca
    try:
        brand_name = brand_detector.names[class_id]
    except Exception:
        try:
            brand_name = brand_detector.names[str(class_id)]
        except Exception:
            brand_name = str(class_id)

    brand_conf = float(box.conf.cpu().numpy())
    print(brand_conf)
    return brand_name, brand_conf
def process_plate(file):
    model = current_app.config['YOLO']
    plate_model = current_app.config['YOLO_PLATE']
    plate_ocr = current_app.config['PLATE_OCR']
    color_model = current_app.config['COLOR']
    upload_folder = current_app.config['UPLOAD_FOLDER']
    brand_detector = current_app.config['BRAND_DETECTOR']
    
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
    try:
        car_image = detect_car(model, file_path)
        color = classify_color(color_model, car_image)
        plate = detect_plate(plate_model, car_image, output_path, unique_id)
        string, plate_conf = convert_plate_to_string(plate, plate_ocr)
        brand, brand_conf = detect_brand(brand_detector, car_image)
    except Exception as e:
        # Limpa o arquivo temporário
        os.remove(file_path)
        raise e

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
    return {
        'plate': {
            'text': string or 'N/A',
            'confidence': float(plate_conf) if isinstance(plate_conf, (int, float)) else 0.0
        },
        'color': {
            'text': color[0],
            'confidence': f"{color[1]:.5f}"
        },
        'brand': {
            'text': brand or 'N/A',
            'confidence': f"{brand_conf:.5f}"
        }
    }