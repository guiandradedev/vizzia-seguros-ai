from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer('cct-xs-v1-global-model')
print(m.run('outputs/53a8f0ea-642a-4deb-8b40-01950d13a857-plate.png')[0].replace("_", ""))