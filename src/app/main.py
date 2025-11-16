import os
from flask import Flask
from .routes import bp as routes
from src.app.models.load_yolo import load_yolo
from fast_plate_ocr import LicensePlateRecognizer
import pandas as pd

def create_app():
    app = Flask(__name__) # Cria a instância do flask

    # Configurações do Flask
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Define o diretório para uploads de imagens
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limita o tamanho máximo dos arquivos (16MB)
    app.config['SECRET_KEY'] = os.urandom(24)  # Chave secreta para sessões (se necessário)

    vehicles_df = pd.read_csv('src/data/VeiculosSubtraidos_2017_2025.csv')
    app.config['VEHICLES_DF'] = vehicles_df
    robery_df = pd.read_excel("src/data/robery_rate_df.xlsx")
    app.config['ROBERY_DF'] = robery_df

    yolo = load_yolo('./src/models/yolo11x.pt')
    plate = load_yolo('./src/models/plate-model.pt')
    brand = load_yolo('./src/models/brand-detector.pt')
    brand_classifier = load_yolo('./src/models/brand-classifier.pt')
    color = load_yolo('./src/models/color-classifier.pt')
    plate_ocr = LicensePlateRecognizer('cct-s-v1-global-model')
    app.config['YOLO'] = yolo 
    app.config['YOLO_PLATE'] = plate
    app.config['BRAND_DETECTOR'] = brand
    app.config['PLATE_OCR'] = plate_ocr
    app.config['COLOR'] = color
    app.config['BRAND_CLASSIFIER'] = brand_classifier

    app.register_blueprint(routes)  # Registra as rotas definidas em `app/api/vision.py`

    @app.route('/')
    def index():
        return "Vizzia Seguros AI API is running."

    return app
