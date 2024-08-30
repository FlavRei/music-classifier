import pandas as pd

def load_data():
    """Load raw TSV files into DataFrames."""
    df_genre = pd.read_csv('data/raw/autotagging_genre.tsv', sep='\t')
    df_instrument = pd.read_csv('data/raw/autotagging_instrument.tsv', sep='\t')
    df_moodtheme = pd.read_csv('data/raw/autotagging_moodtheme.tsv', sep='\t')
    df_meta = pd.read_csv('data/raw/raw.meta.tsv', sep='\t')
    return df_genre, df_instrument, df_moodtheme, df_meta

def drop_duplicates(df_genre, df_instrument, df_moodtheme, df_meta):
    """Remove duplicates based on TRACK_ID."""
    df_genre = df_genre.drop_duplicates(subset=['TRACK_ID'])
    df_instrument = df_instrument.drop_duplicates(subset=['TRACK_ID'])
    df_moodtheme = df_moodtheme.drop_duplicates(subset=['TRACK_ID'])
    df_meta = df_meta.drop_duplicates(subset=['TRACK_ID'])
    return df_genre, df_instrument, df_moodtheme, df_meta

def transform_tags(df_genre, df_instrument, df_moodtheme):
    """Transform TAGS columns into tag lists."""
    
    def genre_tags_to_list(tags):
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split('genre---') if tag.strip()]
        return []

    def instrument_tags_to_list(tags):
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split('instrument---') if tag.strip()]
        return []

    def moodtheme_tags_to_list(tags):
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split('mood/theme---') if tag.strip()]
        return []

    df_genre['GENRES'] = df_genre['TAGS'].apply(genre_tags_to_list)
    df_instrument['INSTRUMENTS'] = df_instrument['TAGS'].apply(instrument_tags_to_list)
    df_moodtheme['MOOD_THEME'] = df_moodtheme['TAGS'].apply(moodtheme_tags_to_list)

    return df_genre, df_instrument, df_moodtheme

def merge_dataframes(df_meta, df_genre, df_instrument, df_moodtheme):
    """Merge the DataFrames to get a final DataFrame."""
    
    df_meta_selected = df_meta.drop(columns=['URL'])
    df_genre_selected = df_genre[['TRACK_ID', 'PATH', 'DURATION', 'GENRES']]
    df_instrument_selected = df_instrument[['TRACK_ID', 'INSTRUMENTS']]
    df_moodtheme_selected = df_moodtheme[['TRACK_ID', 'MOOD_THEME']]

    df_merged = pd.merge(df_meta_selected, df_genre_selected, on='TRACK_ID', how='left')
    df_merged = pd.merge(df_merged, df_instrument_selected, on='TRACK_ID', how='left')
    df_merged = pd.merge(df_merged, df_moodtheme_selected, on='TRACK_ID', how='left')

    df_merged = df_merged.dropna(subset=['PATH'])

    df_merged['INSTRUMENTS'] = df_merged['INSTRUMENTS'].apply(lambda x: x if isinstance(x, list) else ['unknown'])
    df_merged['MOOD_THEME'] = df_merged['MOOD_THEME'].apply(lambda x: x if isinstance(x, list) else ['unknown'])

    return df_merged

def save_cleaned_data(df_merged, output_path):
    """Save the cleaned DataFrame as CSV."""
    df_merged.to_csv(output_path, index=False)
    print(f'DataFrame saved as {output_path}')

def main():
    df_genre, df_instrument, df_moodtheme, df_meta = load_data()
    df_genre, df_instrument, df_moodtheme, df_meta = drop_duplicates(df_genre, df_instrument, df_moodtheme, df_meta)
    df_genre, df_instrument, df_moodtheme = transform_tags(df_genre, df_instrument, df_moodtheme)
    df_merged = merge_dataframes(df_meta, df_genre, df_instrument, df_moodtheme)

    output_path = 'data/processed/music.csv'
    save_cleaned_data(df_merged, output_path)

if __name__ == "__main__":
    main()
