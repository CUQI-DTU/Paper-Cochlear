""" Function for parsing CT data

Authors: Peter Bork, Barbara Mathiesen
July 2021
"""

import pandas as pd
import numpy as np
from os import listdir
from pathlib import Path

pythagoras = lambda x: np.sqrt(sum(i**2 for i in x))

def parse_all_CT(raw_CT_dir = '../data/raw/CT/', parsed_CT_dir = '../data/parsed/CT/'):
    
    ct_overview_and_positions = raw_CT_dir + 'CT positions and overview.xlsx'

    parse_distances(ct_overview_and_positions, 20, parsed_CT_dir)

    ct_csvs = [f for f in listdir(raw_CT_dir) if f[-4:] == '.csv']

    parse_list_of_CT_csvs(raw_CT_dir, ct_csvs, parsed_CT_dir)


def CT_has_been_parsed_already():
    parsed_data_dir = '../data/parsed/CT/'
    parsed_files = [parsed_data_dir + f for f in listdir(parsed_data_dir) if '.csv' in f]
    known_number_of_parsed_files = 20
    if len(parsed_files) == known_number_of_parsed_files:
        print('It seems CT files have already been parsed.')
        return True
    else:
        print('CT files seem to not have been parsed.')
        return False

def parse_csv(data_dir, filename, start_time=-1, time_step=1, nrows=None,
              remove_frames_before_time=-1):
    """ Parses CSV file from ITK snap to row-timed ROI intensities with SD. """

    #import data
    df = pd.read_csv(data_dir + filename)

    #define columns that is concentration and SD
    conc_cols = [colname for colname in df.columns
                 if ('FIESTA' not in colname) and ('mean' in colname)
                 or ('Volume' in colname) or ('Name' in colname)]
    std_cols = [colname for colname in df.columns
                 if ('stdev' in colname) or ('Name' in colname)]

    #make a df with the concentrations and one with the SDs
    df_column_filtered = df[conc_cols].copy()
    std_column_filtered = df[std_cols].copy()

    #Function that shorten the name of the columns
    filter_colname = lambda colname: colname.split('[')[-1].split(']')[0]

    #change the names of the colums to the shortened name
    col_rename_dict = {original_name: filter_colname(original_name)
                       for original_name in df_column_filtered.columns}
    std_col_rename_dict = {original_name: filter_colname(original_name)
                           for original_name in std_column_filtered.columns}

    #use the function to rename the columns
    df_col_renamed = df_column_filtered.rename(columns=col_rename_dict)
    std_col_renamed = std_column_filtered.rename(columns=std_col_rename_dict).drop(index=0)

    #make a dictionary with the volumes
    vol_dict = {label: vol
                for label, vol in zip(df_col_renamed['Label Name'], df_col_renamed['Volume (mm^3)'])
                if label != 'Clear Label'}

    #take out the volumes from the dataframe with intensities and SDs
    df_intensities = df_col_renamed.drop(columns='Volume (mm^3)', index=0)

    #transpose dataframes
    df_row_time = df_intensities.set_index('Label Name').T
    std_row_time = std_col_renamed.set_index('Label Name').T
    #include std in label name of the std columns
    std_row_time_std_col_names = std_row_time.rename(columns=lambda colname: colname + ' std')
    std_row_time_std_col_names=std_row_time_std_col_names.reset_index()
    std_row_time_std_col_names=std_row_time_std_col_names.drop(['index'],axis=1)

    df_row_time=df_row_time.reset_index()
    df_row_time=df_row_time.drop(['index'],axis=1)

    # Join the intensity and the std dataframe to one
    df_row_time = df_row_time.join(std_row_time_std_col_names)

    # create time column
    num_timesteps = df_row_time.shape[0]
    last_time = start_time + time_step * num_timesteps
    time_axis = np.arange(start=start_time, stop=last_time, step=time_step)

    #used to normalize data
    normalized_df = df_row_time.copy() #/ max_cochlear_intensity

    df_ready_for_analysis = normalized_df.copy()
    df_ready_for_analysis['time'] = time_axis + abs(time_axis[0])

    first_row = np.argmax(remove_frames_before_time < time_axis)
    df_ready_for_analysis = df_ready_for_analysis.iloc[first_row:nrows, :]

    return df_ready_for_analysis, vol_dict

def parse_list_of_CT_csvs(datadir, raw_filenames, parsed_CT_dir):
    cochlear_aqueduct_roi_vols = []
    cochlear_roi_vols = []

    for raw_filename in raw_filenames:
        df, vol_dict = parse_csv(datadir, raw_filename, start_time=-5, time_step=5,
                                 remove_frames_before_time=-10)

        tmp_cochlear_aqueduct_roi_vols = [v for (k, v) in vol_dict.items() if 'A' in k]
        tmp_cochlear_roi_vols = [v for (k, v) in vol_dict.items() if 'A' not in k]

        savepath = parsed_CT_dir + raw_filename.replace('.', '_parsed.').lower()
        df.to_csv(savepath, index=False)
        print(f'For file {raw_filename}, the volumes are:\n', '\n'.join([f'{k}: {v}' for (k, v) in vol_dict.items()]))

        if np.var(tmp_cochlear_aqueduct_roi_vols) > 10**(-8):
            print(f'Parsing {raw_filename} where we encounter unexpected variation in ROI volumes')
            print('Variation in cochlear aqueduct roi volumes is not zero: ', np.var(cochlear_aqueduct_roi_vols))
        if np.var(tmp_cochlear_roi_vols)  > 10**(-8):
            print(f'Parsing {raw_filename} where we encounter unexpected variation in ROI volumes')
            print('Variation in cochlear volumes is not zero: ', np.var(cochlear_roi_vols))

        cochlear_aqueduct_roi_vols += tmp_cochlear_aqueduct_roi_vols
        cochlear_roi_vols += tmp_cochlear_roi_vols

    if np.var(cochlear_aqueduct_roi_vols) + np.var(cochlear_roi_vols) > 10**(-10):
        print('Theres unexpected varation in ROI volumes; examine the volumes manually'
              '(print statement available above)')

def parse_distances(positions_excel_file, savedir, microns_per_pixel=20):
    sheet_dict = pd.read_excel(positions_excel_file, sheet_name=None)

    distances_sheet_dict = {k: add_micron_distances(v, microns_per_pixel)
                            for (k, v) in sheet_dict.items()
                            if 'view' not in k}

    for k in distances_sheet_dict.keys():
        savepath = savedir + k.lower() + '_distances' '.csv'
        distances_sheet_dict[k].to_csv(savepath, index=False)

def add_micron_distances(sheet_df, microns_per_pixel):
    df = sheet_df.sort_values(sheet_df.columns[0], ignore_index=True)

    df['x microns'] = df['x'] * microns_per_pixel
    df['y microns'] = df['y'] * microns_per_pixel
    df['z microns'] = df['z'] * microns_per_pixel

    df['distance microns'] = np.nan * np.zeros_like(df['x'])
    df.loc[0, 'distance microns'] = 0.0

    current_roi_class = ''
    for row_index in range(0, len(df)):
        row_roi = df.iloc[row_index, 0]
        if row_roi[:2] == current_roi_class:
            x_distance = abs(df.loc[row_index, 'x microns'] - df.loc[row_index - 1, 'x microns'])
            y_distance = abs(df.loc[row_index, 'y microns'] - df.loc[row_index - 1, 'y microns'])
            z_distance = abs(df.loc[row_index, 'z microns'] - df.loc[row_index - 1, 'z microns'])
            prior_acc_dist = df.loc[row_index-1, 'distance microns']
            df.loc[row_index, 'distance microns'] = prior_acc_dist + pythagoras([x_distance, y_distance, z_distance])
        else:
            current_roi_class = row_roi[:2]
            df.loc[row_index, 'distance microns'] = 0.0

    return df

def parseca1distances(rawdatadir, locsfile, parseddir, microns_per_pixel):
    posdf = pd.read_csv(rawdatadir + locsfile, header=0)
    distdf = add_micron_distances(posdf, microns_per_pixel)
    df = distdf.sort_values(distdf.columns[2], ignore_index=True)
    for row in range(1, len(df)):
        df.loc[row, 'distance microns'] = df.loc[row-1, 'distance microns'] + pythagoras(df.iloc[row, [4, 5, 6]] - df.iloc[row -1 , [4, 5, 6]])
        
    df.to_csv(parseddir + locsfile.replace("_locations_", "_").replace(".csv", "_distances.csv"))

def load_parsed_dfs(signal_files):
    result = {}

    for f in signal_files:
        name = f.split('_parsed')[0].split('/')[-1]
        signals_df = pd.read_csv(f)

        distances_csv = f.replace('_parsed', '_distances')

        if Path(distances_csv).is_file():
            distances_df = load_parsed_distances(distances_csv)
        else:
            print('Could not find distances in:\n', distances_csv)
            distances_df = None

        result[name] = {'CT signals': signals_df,
                        'distances': distances_df}

    return result

def load_parsed_distances(distances_csv):
    df = pd.read_csv(distances_csv)
    return df[[df.columns[0], 'distance microns']]
