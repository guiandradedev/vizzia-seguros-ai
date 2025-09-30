from ultralytics import YOLO
import os
from Colors import Colors
import cv2

# Definindo as variaveis base 
plates_model = "plate-model.pt" # Modelo do yolo com fine tuning de placas de carros brasileiros
yolov11_model = "yolo11n.pt" # Modelo default do YoLo
car_class_id = 2 # Classe do carro para limitar o tipo de detecções possíveis
index = ""
image_path = f"images_test/image{index}.png" # Path da imagem
output_path = f"results/debug{index}"

# Cria o diretório de saída se não existir
if not os.path.exists(output_path):
    os.makedirs(output_path)
    Colors.info(f"Diretório criado: {output_path}")

if not os.path.exists(image_path):
    Colors.error("Erro: Imagem não existe no path")
    exit()

if not os.path.exists(plates_model):
    Colors.error("Erro: Dataset de placas não existem no path")
    exit()

image = cv2.imread(image_path)
if(image is None):
    Colors.error("Erro: Não foi possível carregar a imagem")
    exit()

# Iniciando os modelos
try:
    car_detection_model = YOLO(yolov11_model)
    plate_detection_model = YOLO(plates_model)
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
cv2.imwrite(f"{output_path}/car_cropped_debug.jpg", car_cropped)
Colors.info(f"Imagem do carro salva como '{output_path}/car_cropped_debug.jpg'")

# Realiza a detecção da placa na imagem do carro
plate_results = plate_detection_model(
    source=car_cropped,
    conf=0.3,  # Reduzi a confiança para detectar mais placas
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
plate_cropped = car_cropped[y1:y2, x1:x2]

cv2.imwrite(f"{output_path}/plate.png", plate_cropped) # Salva a foto da placa
plate.save(filename=f"{output_path}/detection_result.jpg") # Salva a foto do carro com detecções

Colors.success("Processamento concluído!")
Colors.info("Arquivos salvos: plate.png, detection_result.jpg, car_cropped_debug.jpg")