import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('cleaned_train.csv')
songs = pd.read_csv('cleaned_songs.csv')
members = pd.read_csv('cleaned_members.csv', parse_dates=['registration_init_time', 'expiration_date'])
song_extra = pd.read_csv('cleaned_song_extra_info_new.csv')

# Calculate membership duration
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days

# Clean age outliers
members['bd'] = members['bd'].clip(lower=5, upper=105)

# Convert genre_ids to string type
songs['genre_ids'] = songs['genre_ids'].astype(str)

# Split genre_ids into first genre and handle 'nan' values
songs['primary_genre'] = songs['genre_ids'].str.split('|').str[0]
songs['primary_genre'] = songs['primary_genre'].replace('nan', np.nan)

# Convert primary_genre to categorical and add 'Unknown' category
songs['primary_genre'] = songs['primary_genre'].astype('category')
if 'Unknown' not in songs['primary_genre'].cat.categories:
    songs['primary_genre'] = songs['primary_genre'].cat.add_categories(['Unknown'])

# Merge dataframes
merged = train.merge(songs, on='song_id', how='left')
merged = merged.merge(members, on='msno', how='left', validate='many_to_one')
merged = merged.merge(song_extra, on='song_id', how='left')

# Replace string "nan" with np.nan
merged = merged.replace("nan", np.nan)

# Define fill values
fill_values = {
    'composer': 'Unknown',
    'lyricist': 'Unknown',
    'bd': members['bd'].median() if not members['bd'].isnull().all() else 27.0,
    'gender': 'Unknown',
    'membership_days': members['membership_days'].median() if not members['membership_days'].isnull().all() else 1627.0,
    'name': 'Unknown',
    'primary_genre': 'Unknown',
    'source_system_tab': 'Unknown',
    'source_screen_name': 'Unknown',
    'source_type': 'Unknown',
    'song_length': merged['song_length'].median(),
    'registered_via': members['registered_via'].median(),
    'registration_init_time': members['registration_init_time'].min(),
    'expiration_date': members['expiration_date'].max(),
    'genre_ids': 'Unknown',
    'artist_name': 'Unknown'
}

# Fill missing values
merged.fillna(value=fill_values, inplace=True)

# Verify no missing values
print(merged.isnull().sum())

# Save merged data
merged.to_csv('final_merged.csv', index=False)
