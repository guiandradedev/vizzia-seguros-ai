import pandas as pd
import numpy as np
import osmnx as ox

def get_crime_amount(street, city, radius=500):
    # Define a localização central
    vehicles_df = pd.read_csv('src/app/data/VeiculosSubtraidos_2017_2025.csv')
    grafo_local = ox.graph.graph_from_address(street, radius, dist_type='bbox', network_type='all_public', simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)


    # Baixa os dados de crimes da OpenStreetMap na área especificada
    tags = {'amenity': 'police', 'crime': True}
    try:
        gdf = ox.geometries_from_point(center_point, tags=tags, dist=radius)
    except Exception as e:
        print(f"Erro ao baixar dados do OSM: {e}")
        return None

    if gdf.empty:
        print("Nenhum dado de crime encontrado na área especificada.")
        return None

    # Filtra apenas os dados relevantes (exemplo: tipos de crime)
    crime_types = gdf['crime'].value_counts().to_dict()

    # Converte para DataFrame para melhor visualização
    crime_stats_df = pd.DataFrame(list(crime_types.items()), columns=['Tipo de Crime', 'Contagem'])

    return crime_stats_df