import cv2
import pytesseract
import numpy as np


def preprocess_plate(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale_percent = 250
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)

    return resized


def extract_plate_text(image_path: str):
    processed = preprocess_plate(image_path)

    custom_config = (
        "--oem 3 --psm 8 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(processed, config=custom_config)
    text = text.strip().replace(" ", "").replace("\n", "").replace("‘", "").replace("’", "")

    return text, processed


if __name__ == "__main__":
    # path = "results/debug2/plate.png"
    path = "discord-placa.png"


    result, processed_img = extract_plate_text(path)

    print(result)

    cv2.imwrite("debug_placa_processada.png", processed_img)