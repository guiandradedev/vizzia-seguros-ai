from ultralytics import YOLO
from src.app.Colors import Colors

def load_yolo(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        Colors.error(f"Erro: Modelo não carregado, {e}")
        exit()