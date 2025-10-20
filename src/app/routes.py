from flask import Blueprint, request, jsonify
from src.app.api.plate import process_plate
from src.app.api.criminal_stats import get_nearby_crimes_amount
from src.app.api.criminal_stats import classify_crime_amount

bp = Blueprint('main', __name__)

@bp.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    result = process_plate(file)
    return jsonify(result)

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
