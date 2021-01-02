
#Combine weekly and monthly unemployment data as granular as possible

import pandas as pd
import matplotlib.pyplot as plt

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
data.merge_data()

# Display national cases
data.df_covid.pivot_table(index = 'Date', 
                            values = 'Cases', 
                            aggfunc='sum')['2020-03-01':].diff().rolling(7).mean().plot()

plt.title('Total US cases')
plt.show()

# Show latest unemployment claims
data.df_weekly.pivot_table(index='Date', 
                           columns='State', 
                           values='Initial_claims').sum(axis=1).plot()
plt.ylabel('Claims in M')
plt.title('US total weekly unemployment claims')
plt.show()

# Plot the mobility data
states = ['Colorado', 'California', 'New York', 'Tennessee', 'Texas']

pd.pivot_table(data.df_descartes[data.df_descartes['State'].isin(states)],
               index='Date', columns=['State'],
               values='Mobility index')['2020-03-10':].resample('W').mean().plot()
plt.ylabel('Mobility index')
plt.title('Movement trends - selected states')
plt.show()
