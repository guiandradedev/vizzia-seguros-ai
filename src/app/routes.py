from flask import Blueprint, request, jsonify
from src.app.api.plate import process_plate
from src.app.api.criminal_stats import get_nearby_crimes_amount
from src.app.api.criminal_stats import classify_crime_amount, get_model_robery_quantity
from src.app.api.fipe_search import get_models_brand_by_year, get_fipe_by_info_resumed
from src.app.utils.err_api import ErrAPI
bp = Blueprint('main', __name__)

@bp.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result = process_plate(file)
        return jsonify(result)
    except ErrAPI as e:
        print(e.status_code)
        return jsonify({'error': str(e)}), e.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@bp.route('/estimate_details', methods=['POST'])
def estimate_details():
    data = request.get_json()

    cep = data.get('cep')
    dist = data.get('dist', 500)

    car_model = data.get('car_model')

    if not cep:
        return jsonify({'error' : 'campo CEP nulo'}), 400
    if not car_model:
        return jsonify({'error' : 'campo modelo do carro nulo'}), 400
    try:
        amount = get_nearby_crimes_amount(cep, dist)
        level = classify_crime_amount(amount)
        recorrencia = get_model_robery_quantity(car_model)

        return jsonify({
            "address": {
                'cep': cep,
                'radius': dist,
                'crimes_amount': amount,
                'danger_level': level # Valor de 0 a 10
            },
            "car": {
                'model': car_model,
                'robbery_recorrency': recorrencia
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@bp.route('/get_models_by_year', methods=['POST'])
def get_models_by_year():
    data = request.get_json()

    print(data)

    year_code = data.get('year_code')
    brand_code = data.get('brand_code')

    if not year_code:
        return jsonify({'error' : 'campo ano nulo'}), 400
    if not brand_code:
        return jsonify({'error' : 'campo marca nulo'}), 400
    try:
        models_json = get_models_brand_by_year(year_code=year_code, 
                                               brand_code=brand_code)

        return models_json
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/get_fipe', methods=['POST'])
def get_fipe_by_info():
    data = request.get_json()

    brand_code = data.get('brand_code')
    client_car_model = data.get('client_car_model')
    year = data.get('year')
    motorization = data.get('motorization')

    if not brand_code:
        return jsonify({'error' : 'campo ano nulo'}), 400
    if not client_car_model:
        return jsonify({'error' : 'campo marca nulo'}), 400
    if not year:
        return jsonify({'error' : 'campo ano nulo'}), 400
    if not motorization:
        return jsonify({'error' : 'campo motorização nulo'}), 400
    
    try:
        models_json = get_fipe_by_info_resumed(brand_code=brand_code,
                                               client_car_model=client_car_model,
                                               year=year,
                                               motorization=motorization)

        return models_json
    except Exception as e:
        return jsonify({'error': str(e)}), 500