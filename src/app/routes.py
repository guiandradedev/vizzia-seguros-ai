from flask import Blueprint, request, jsonify
from src.app.api.plate import process_plate

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
