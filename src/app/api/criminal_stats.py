import pandas as pd
import numpy as np
import osmnx as ox
import brazilcep
from geopy.geocoders import Nominatim
import sys
import os
import unicodedata
# nfkd_form = unicodedata.normalize('NFD', texto)
geolocator = Nominatim(user_agent="test_app")

def get_nearby_crimes_amount(zipcode, radius=500):
    vehicles_df = pd.read_csv('src/data/VeiculosSubtraidos_2017_2025.csv')

    has_lat_lon = is_cep_valid(zipcode)
    if has_lat_lon:
        location = get_place_by_cep(zipcode)
        coordinates = (location.latitude,location.longitude)

        print("Using Lat Lon")
        local_graph = ox.graph.graph_from_point(coordinates, radius, dist_type='bbox', network_type='all', 
                                                simplify=True, retain_all=False, truncate_by_edge=False, custom_filter=None)
    else:
        location = get_place_by_cep(zipcode)
        street = location['street']
        street = unicodedata.normalize('NFD', street).encode('ascii', 'ignore').decode("utf-8")
        print (street)
        print("Using Street")
        local_graph = ox.graph.graph_from_address(street, radius, dist_type='bbox', network_type='all_public', 
                                                  simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)

    city = get_city_by_cep(zipcode)
    print(city)
    city = unicodedata.normalize('NFD', city).encode('ascii', 'ignore').decode("utf-8")
    df_city = vehicles_df[vehicles_df['CIDADE'] == city.upper()]
    print(df_city)
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
    if amount > 100: 
        danger_level = amount // 100
    else: 
        if amount < 50:
            danger_level = 0
        else: 
            danger_level = 1
    if danger_level > 10:
        danger_level = 10
    return danger_level

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

def get_place_by_cep(zipcode):
    try:
        address = brazilcep.get_address_from_cep(zipcode)
        location = geolocator.geocode(address['street'] + ", " + address['city'] + " - " + address['district'])
        if location:
            return location
        else:
            return address
    except Exception as e:
        print(f"Erro ao reconhecer o CEP {zipcode} {e}")
        sys.exit(1)

def get_city_by_cep(zipcode):
    try:
        address = brazilcep.get_address_from_cep(zipcode)
        return address['city']
    except Exception as e:
        print(f"Erro ao reconhecer o CEP {zipcode} {e}")
        sys.exit(1)

def is_cep_valid(zipcode):
    try:
        address = brazilcep.get_address_from_cep(zipcode)
        location = geolocator.geocode(address['street'] + ", " + address['city'] + " - " + address['district'])
        if location:
            return True
        else:
            return False
    except Exception as e:
        print(f"Erro ao reconhecer o CEP {zipcode} {e}")
        sys.exit(1)
