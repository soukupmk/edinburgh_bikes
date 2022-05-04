import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import folium
import streamlit_folium
from streamlit_folium import folium_static
import utils
import datetime as dt
import os

st.write("""
# Edinburgh bikes
### In loving memory of Just Eat Cycles
**Author:** Marek Soukup
""")


# LOAD DATA
# @st.cache
# def load_data():
bikes_df = pd.read_csv(os.path.join('data', 'bikes_df.csv')).astype({'started_at': 'datetime64', 'ended_at': 'datetime64'})
weather_df = pd.read_csv(os.path.join('data', 'weather_df.csv'))
dist_df = pd.read_csv(os.path.join('data', 'dist_table.csv'), index_col='station1')
dist_df.columns.name = 'station2'

bwdf = utils.join_bikes_weather()

starts_df = pd.DataFrame(bikes_df[['start_station_name']].value_counts(), columns=['num_starts']).rename_axis('station_name')
ends_df = pd.DataFrame(bikes_df[['end_station_name']].value_counts(), columns=['num_ends']).rename_axis('station_name')
starts_ends_df = starts_df.join(ends_df, how='outer').fillna(0)
dates_all = pd.date_range(bikes_df['started_at'].dt.floor('D').min(), bikes_df['started_at'].dt.floor('D').max())

    # return bikes_df, weather_df, bwdf, starts_df, ends_df, starts_ends_df, dates_all

# bikes_df, weather_df, bwdf, starts_df, ends_df, starts_ends_df, dates_all = load_data()

st.write("""
## Identify the active stations and the inactive stations
*activity* = the time difference between 1.7.2021 an the very last record for a given station
""")
@st.cache
def find_active_inactive_stations():
    df = bikes_df[['start_station_name', 'started_at']]
    last_start = pd.Timestamp('2021-07-01')
    df = df.groupby('start_station_name').max()
    df = df['started_at'].apply(lambda x: last_start-x)

    d1 = df.astype('timedelta64[m]').sort_values().iloc[:10]
    d2 = df.astype('timedelta64[M]').sort_values().iloc[-10:]

    return d1, d2

d1, d2 = find_active_inactive_stations()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
fig.suptitle("The difference between 1.7.2021 and the last record for selected stations")
d1.plot.bar(ax=ax1, ylabel='minutes', title='active stations')
ax1.set_xticklabels(d1.index, rotation=45, ha='left')
d2.plot.bar(ax=ax2, ylabel='months', title='inactive stations')
for ax, d in zip([ax1, ax2], [d1, d2]):
    ax.set_xticklabels(d.index, rotation=45, ha='right')
st.pyplot(fig=fig)

st.write("""
## Identify the most frequently visited stations
""")

@st.cache
def identify_frequently_visited_stations():
    df = (starts_ends_df['num_starts'] + starts_ends_df['num_ends']).sort_values(ascending=False).reset_index().rename(columns={0: 'starts_plus_ends'})
    d = df.iloc[:20].set_index('station_name')
    return d

d = identify_frequently_visited_stations()
fig, ax = plt.subplots(figsize=(12, 4))
d.plot.bar(ax=ax, title='stations with the largest sum of starts and ends')
ax.set_xticklabels(d.index, rotation=45, ha='right')
st.pyplot(fig=fig)

st.write("""
## Identify stations with significant excess demand and excess supply
""")

@st.cache
def identify_excess_supply_demand():
    df = (starts_ends_df['num_ends'] - starts_ends_df['num_starts']).sort_values(ascending=False).reset_index().rename(columns={0: 'ends_minus_starts'})
    return df

df = identify_excess_supply_demand()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
fig.suptitle('The difference between starts and ends')
d1 = df.iloc[:10].set_index('station_name')
d1.plot.bar(ax=ax1, title='ends >> starts')
d2 = df.iloc[-10:].set_index('station_name')
d2.plot.bar(ax=ax2, title='ends << starts')

for ax, d in zip([ax1, ax2], [d1, d2]):
    ax.set_xticklabels(d.index, rotation=45, ha='right')

st.pyplot(fig=fig)

st.write("""
## Calculate the distances between stations
""")
st.write("Select your stations:")

c1, c2 = st.columns(2)
with c1:
    station1 = st.selectbox(label='Station 1', options=dist_df.index.tolist())
with c2:
    station2 = st.selectbox(label='Station 2', options=dist_df.columns.tolist())

distance = dist_df.loc[station1, station2]
st.write(" ")
st.write(f"The distance between {station1} and {station2} is **{round(float(distance), 2)}** kilometres.")

st.write("## What is the distribution of rental times? Find outliers, display a histogram")

q = st.slider("Select cut quantile", 0.9, 1., 0.97, 0.01)

durations = bikes_df[['duration']]
cut = np.quantile(durations.values, q)

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(durations[durations < cut], bins=50, density=True, edgecolor='k')
ax.set_xlabel('seconds')
st.pyplot(fig=fig)

st.write("## Display the time series of the demand for bicycles.")
df = pd.DataFrame(bikes_df['started_at'].dt.round('D').value_counts()).sort_index().rename(columns={'started_at': 'num_starts'}).rename_axis('date')

start_date, end_date = st.slider(
    'Select date range', df.index.min(), df.index.max(),
    value=[dt.datetime.date(df.index.min()), dt.datetime.date(df.index.max())]
)

fig, ax = plt.subplots(figsize=(14, 5))
df.loc[start_date: end_date].plot(ax=ax, grid=True)
st.pyplot(fig)

st.write("## Display the relationship between weather and the demand for bicycles")
col = st.radio("Pick one", ['temperature', 'wind', 'rain'])

fig, ax = plt.subplots()
bwdf.plot.scatter(col, 'num_rentals', ax=ax)
st.pyplot(fig=fig)

st.write("## Create a map of stations")
year = st.radio("Select year", [2018, 2019, 2020, 2021])

@st.cache
def make_map(year):
    df = bikes_df.loc[bikes_df.started_at.dt.year == year, ['start_station_name', 'start_station_latitude', 'start_station_longitude']].round(4).drop_duplicates()
    df.columns = ['station_name', 'lat', 'lon']
    df = df.set_index('station_name')
    return df

df = make_map(year)
m = folium.Map(location=df.mean(), zoom_start=12)
for idx, row in df.reset_index().iterrows():
    folium.Marker(row[['lat', 'lon']].values.tolist(),
                  tooltip=folium.Tooltip(row['station_name']),
                  icon=folium.Icon(icon='bicycle', prefix='fa')
                  ).add_to(m)
folium_static(m)

