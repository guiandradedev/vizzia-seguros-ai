from flask import Blueprint, request, jsonify
from src.app.api.plate import process_plate
from src.app.api.criminal_stats import get_nearby_crimes_amount
from src.app.api.criminal_stats import classify_crime_amount
from src.app.api.fipe_search import get_models_brand_by_year
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
@bp.route('/get_crime_amount', methods=['POST'])
def criminal_statistc_route():
    data = request.get_json()

    cep = data.get('cep')
    dist = data.get('dist', 500)

    if not cep:
        return jsonify({'error' : 'campo CEP nulo'}), 400
    try:
        amount = get_nearby_crimes_amount(cep, dist)
        level = classify_crime_amount(amount)

        return jsonify({
            'cep': cep,
            'radius': dist,
            'crimes_amount': amount,
            'danger_level': level # Valor de 0 a 10
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
