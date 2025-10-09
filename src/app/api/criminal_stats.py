import pandas as pd
import numpy as np
import osmnx as ox

def get_nearby_crimes_amount(street, city, radius=500):
    # Define a localização central
    vehicles_df = pd.read_csv('src/data/VeiculosSubtraidos_2017_2025.csv')
    print('File readed')
    local_graph = ox.graph.graph_from_address(street, radius, dist_type='bbox', network_type='all_public', simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)

    df_city = vehicles_df[vehicles_df['CIDADE'] == city.upper()]

    maior_lat, menor_lat, maior_lon, menor_lon = get_radius_coordinates(local_graph)

    local_df = pd.DataFrame()
    for index,row in df_city.iterrows():
        lat = row['LATITUDE']
        lon = row['LONGITUDE']
        if pd.notnull(lat) and lat != 0 and pd.notnull(lon) and lon != 0:
            if  ((lat <= maior_lat and lat >= menor_lat) and
                (lon <= maior_lon and lon >= menor_lon)):
                local_df = pd.concat([local_df, row.to_frame().T])
                
    return local_df.shape[0]

def classify_crime_amount(amount):
    if amount == 0:
        return "Muito Baixo"
    elif amount <= 500:
        return "Baixo"
    elif amount <= 100:
        return "Médio"
    elif amount <= 2000:
        return "Alto"
    else:
        return "Muito Alto"

def get_radius_coordinates(local_graph):
    maior_lat=-1000
    menor_lat=1
    for lat in local_graph.nodes('y'):
        if lat[1] > maior_lat:
            maior_lat = lat[1]
        if lat[1] < menor_lat:
            menor_lat = lat[1]

    maior_lon=-1000
    menor_lon=1
    for lon in local_graph.nodes('x'):
        if lon[1] > maior_lon:
            maior_lon = lon[1]
        if lon[1] < menor_lon:
            menor_lon = lon[1]
    return maior_lat, menor_lat, maior_lon, menor_lon

print(get_nearby_crimes_amount("Rua Falcao Filho", "Campinas", 500))
