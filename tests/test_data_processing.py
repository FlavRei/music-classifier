import pytest
import pandas as pd
from src.data_processing import drop_duplicates, transform_tags, merge_dataframes

@pytest.fixture
def sample_data():
    """Fixture to provide example DataFrames."""
    df_genre = pd.DataFrame({
        'TRACK_ID': ['track_0001', 'track_0002'],
        'ARTIST_ID': ['artist_01', 'artist_02'],
        'ALBUM_ID': ['album_01', 'album_02'],
        'PATH': ['path/to/track1', 'path/to/track2'],
        'DURATION': [210.0, 200.0],
        'TAGS': ['genre---rock genre---pop', 'genre---metal']
    })
    df_instrument = pd.DataFrame({
        'TRACK_ID': ['track_0001', 'track_0002'],
        'ARTIST_ID': ['artist_01', 'artist_02'],
        'ALBUM_ID': ['album_01', 'album_02'],
        'PATH': ['path/to/track1', 'path/to/track2'],
        'DURATION': [210.0, 200.0],
        'TAGS': ['instrument---guitar', 'instrument---drums']
    })
    df_moodtheme = pd.DataFrame({
        'TRACK_ID': ['track_0001', 'track_0002'],
        'ARTIST_ID': ['artist_01', 'artist_02'],
        'ALBUM_ID': ['album_01', 'album_02'],
        'PATH': ['path/to/track1', 'path/to/track2'],
        'DURATION': [210.0, 200.0],
        'TAGS': ['mood/theme---happy', 'mood/theme---sad']
    })
    df_meta = pd.DataFrame({
        'TRACK_ID': ['track_0001', 'track_0002'],
        'ARTIST_ID': ['artist_01', 'artist_02'],
        'ALBUM_ID': ['album_01', 'album_02'],
        'TRACK_NAME': ['Track 1', 'Track 2'],
        'ARTIST_NAME': ['Artist 1', 'Artist 2'],
        'ALBUM_NAME': ['Album 1', 'Album 2'],
        'RELEASEDATE': ['2021-01-01', '2021-02-01'],
        'URL': ['url/to/track1', 'url/to/track2']
    })
    return df_genre, df_instrument, df_moodtheme, df_meta

def test_drop_duplicates(sample_data):
    """Test removing duplicates."""
    df_genre, df_instrument, df_moodtheme, df_meta = sample_data

    df_genre = pd.concat([df_genre, df_genre.iloc[0:1]], ignore_index=True)
    df_genre, df_instrument, df_moodtheme, df_meta = drop_duplicates(df_genre, df_instrument, df_moodtheme, df_meta)

    assert df_genre.shape[0] == 2 

def test_transform_tags(sample_data):
    """Test transforming TAGS columns into lists."""
    df_genre, df_instrument, df_moodtheme, df_meta = sample_data

    df_genre, df_instrument, df_moodtheme = transform_tags(df_genre, df_instrument, df_moodtheme)

    assert df_genre['GENRES'].iloc[0] == ['rock', 'pop']
    assert df_instrument['INSTRUMENTS'].iloc[0] == ['guitar']
    assert df_moodtheme['MOOD_THEME'].iloc[0] == ['happy']

def test_merge_dataframes(sample_data):
    """Test merging DataFrames."""
    df_genre, df_instrument, df_moodtheme, df_meta = sample_data

    df_genre, df_instrument, df_moodtheme = transform_tags(df_genre, df_instrument, df_moodtheme)
    df_merged = merge_dataframes(df_meta, df_genre, df_instrument, df_moodtheme)

    assert df_merged.shape[0] == 2
    assert df_merged['GENRES'].iloc[0] == ['rock', 'pop']
    assert df_merged['INSTRUMENTS'].iloc[0] == ['guitar']
    assert df_merged['MOOD_THEME'].iloc[0] == ['happy']
    assert df_merged['PATH'].iloc[0] == 'path/to/track1'

