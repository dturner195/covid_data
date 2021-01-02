
#Combine weekly and monthly unemployment data as granular as possible
import requests
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon

from datetime import datetime, timedelta
from pathlib import Path
path = Path(r'/Users/_DMT/jupyter/covid')
from CovidClass.CovidData import CovidData


data = CovidData()

# Pull COVID Cases/Deaths from USAfacts
data.get_covid()

# Pull unemployment data from BLS
data.get_bls()

# Load age demographic data and income
data.get_age()
data.get_percap_census_data()

# Pull latest weekly unemlpoyment claims
data.get_weekly()

# Get the mobility data from Descartes Labs
data.get_descartes()
   

# Merge the datasets
print(data.merge_data().head())
