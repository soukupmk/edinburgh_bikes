import pandas as pd
import streamlit as st
import os


@st.cache
def join_bikes_weather():
    bikes_df = pd.read_csv(os.path.join('data', 'bikes_df.csv')).astype({'started_at': 'datetime64', 'ended_at': 'datetime64'})
    weather_df = pd.read_csv(os.path.join('data', 'weather_df.csv'))

    bdf = bikes_df['started_at'].dt.round('H').value_counts().sort_index().reset_index()
    bdf.columns = ['time', 'num_rentals']
    bdf = (bdf.assign(date=bdf['time'].dt.round('D'), hh=bdf['time'].dt.hour)
           .set_index(['date', 'hh'])
           .sort_values('time')
           .drop('time', axis=1)
           )

    wdf = weather_df[['date', 'time', 'temp', 'wind', 'rain']]
    wdf['date'] = pd.to_datetime(wdf['date'])
    wdf['time'] = wdf['time'].apply(lambda x: x.split(':')[0]).apply(lambda x: int(x[1:]) if str(x)[0]=='0' else x)
    for col in ['temp', 'wind', 'rain']:
        wdf[col] = wdf[col].apply(lambda x: x.split(' ')[0])

    wdf['hh'] = wdf['time'].astype(int)
    wdf = wdf.set_index(['date', 'hh']).drop('time', axis=1).rename(columns={'temp': 'temperature'})

    bwdf = bdf.join(wdf).dropna().astype(float)

    return bwdf