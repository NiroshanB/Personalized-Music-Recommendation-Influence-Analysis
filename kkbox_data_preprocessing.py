import pandas as pd
import numpy as np

#purpose: data cleaning (general ETL)

#load data with efficient dtypes to reduce memory
dtypes = {
    'msno': 'category',
    'song_id': 'category',
    'source_system_tab': 'category',
    'source_screen_name': 'category',
    'source_type': 'category',
    'target': 'int8' 
}
#had issue with running -> replace with path
train_data = pd.read_csv('train.csv',dtype=dtypes)
song_extra_info_data = pd.read_csv('song_extra_info.csv')
members_data = pd.read_csv('members.csv', parse_dates=['registration_init_time', 'expiration_date'])
songs_data = pd.read_csv('songs.csv', dtype={'genre_ids': 'category', 'language': 'category'})

#formatting data output (show all results)
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 200)  # Increase width
pd.set_option("display.max_colwidth", None)  # Prevent truncation of long text

#output of train data
print(train_data.head()) 
print(len(train_data))

#check for missing values
print(train_data.isnull().sum())

#percentage of missing fata (significance)
missing_values = train_data.isnull().sum()
total_values = len(train_data)
missing_percentage = (missing_values / total_values) * 100
print("missing percentages:\n", missing_percentage)

#drop rows with minimal missing data (less than 1%)
train_data.dropna(subset=['source_system_tab', 'source_type'], inplace=True)

unique_values = train_data['source_screen_name'].unique()
print("Unique values in source_screen_name:")
print(unique_values)

#fill in missing data with "Missing" category
train_data['source_screen_name'] = train_data['source_screen_name'].cat.add_categories(['Missing'])
train_data['source_screen_name'].fillna('Missing', inplace=True)

#check for missing values (should be 0%)
print("Remaining missing values:")
print(train_data.isnull().sum())

#remove duplicates
print(f"Original shape: {train_data.shape}")
train_data = train_data.drop_duplicates()
# assumes one record per user-song interaction 
train_data = train_data.drop_duplicates(subset=['msno', 'song_id']) 
print(f"New shape: {train_data.shape}") #check if there were duplicates 

# Clean train.csv and save intermediate output
train_data.to_csv('cleaned_train.csv', index=False)

# clean song_extra_info.csv (drop isrc)
song_extra_info_new = song_extra_info_data[['song_id', 'name']]
print(song_extra_info_new.head())
song_extra_info_new.to_csv('cleaned_song_extra_info_new.csv', index = False)

# clean members.csv (drop city and remove all rows that dont have a gender)
members_new = members_data[['msno', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date']]
members_new = members_new.dropna(subset=['gender']) 
members_new.to_csv('cleaned_members.csv', index = False)
print(members_new.head())


# clean songs.csv (drop language)
songs_new = songs_data[['song_id', "song_length", 'genre_ids', 'artist_name', 'composer', 'lyricist']]
songs_new.to_csv('cleaned_songs.csv', index = False)
print(songs_new.head())

# Load the cleaned dataframes
songs_df = pd.read_csv('cleaned_songs.csv')
train_df = pd.read_csv('cleaned_train.csv')
members_df = pd.read_csv('cleaned_members.csv')
song_extra_info_df = pd.read_csv('cleaned_song_extra_info_new.csv')

# Function to handle missing values
def handle_missing_values(df, column_fill_map):
    for column, fill_value in column_fill_map.items():
        if column in df.columns:
            df[column].fillna(fill_value, inplace=True)

# Handle missing values in cleaned_songs.csv
songs_fill_map = {
    'composer': 'Unknown',
    'lyricist': 'Unknown'
}
handle_missing_values(songs_df, songs_fill_map)
songs_df.to_csv('cleaned_songs.csv', index=False)  # Save updated dataframe

# Handle missing values in cleaned_train.csv
train_fill_map = {
    'source_screen_name': 'Missing'
}
handle_missing_values(train_df, train_fill_map)
train_df.to_csv('cleaned_train.csv', index=False)  # Save updated dataframe

# Handle missing values in cleaned_members.csv
members_fill_map = {
    'bd': members_df['bd'].median(),  # Fill missing birthdate with median
    'gender': 'Unknown'  # This should already be handled by the original script
}
handle_missing_values(members_df, members_fill_map)
members_df.to_csv('cleaned_members.csv', index=False)  # Save updated dataframe

# Handle missing values in cleaned_song_extra_info.csv
song_extra_info_fill_map = {
    'name': 'Unknown'  # Fill missing song names with "Unknown"
}
handle_missing_values(song_extra_info_df, song_extra_info_fill_map)
song_extra_info_df.to_csv('cleaned_song_extra_info_new.csv', index=False)  # Save updated dataframe

# Print confirmation of missing value handling
print("Missing values handled and saved in respective files.")
