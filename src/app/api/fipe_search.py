import requests

def get_fipe_value_by_info(client_brand, client_car_model, client_car_year):
    base_url = f"https://parallelum.com.br/fipe/api/v1/carros"


    brand = get_brand_code_by_name(brand_name = client_brand)
    brand_code = brand['codigo']

    model = get_model_code_by_model_and_brand(model_name = client_car_model,brand_code = brand_code)
    model_code = model['codigo']

    year = get_year_by_model_and_brand(model_code = model_code , brand_code = brand_code, client_car_year = client_car_year)
    year_code = year['codigo']
    print(year)
    fipe = requests.get(f'{base_url}/marcas/{brand_code}/modelos/{model_code}/anos/{year_code}').json()


    return fipe['Valor']

def get_brand_code_by_name(brand_name):
    base_url = f"https://parallelum.com.br/fipe/api/v1/carros"

    brands_json = requests.get(f"{base_url}/marcas").json()
    find_brand = ''
    find = False
    for brand in brands_json:
        if brand['nome'].lower() == brand_name.lower():
            find_brand = brand
            find = True

    if not find:
        raise ValueError('Marca {brand_name} não encontrada')
    
    return find_brand

def get_model_code_by_model_and_brand(model_name,brand_code):
    base_url = f"https://parallelum.com.br/fipe/api/v1/carros"

    models_json = requests.get(f'{base_url}/marcas/{brand_code}/modelos').json()
    models = models_json['modelos']
    find_model = ''
    find = False
    for model in models:
        if model['nome'].lower() == model_name.lower():
            find_model = model
            find = True

    if not find:
        raise ValueError('Modelo {model_name} não encontrada')
    
    return find_model

def get_year_by_model_and_brand(model_code, brand_code, client_car_year):
    base_url = f"https://parallelum.com.br/fipe/api/v1/carros"

    years_json = requests.get(f'{base_url}/marcas/{brand_code}/modelos/{model_code}/anos').json()
    find_year = ''
    find = False
    for year in years_json:
        year_json_name = year['nome'].split()[0]
        if year_json_name.lower() == client_car_year.lower():
            find_year = year
            find = True

    if not find:
        raise ValueError('Modelo {client_car_year} não encontrada')
    return find_year

print(get_fipe_value_by_info('VW - VolksWagen','JETTA Comfortline 1.4 TSI 16V 4p Aut.','2018'))