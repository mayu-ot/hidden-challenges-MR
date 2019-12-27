# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_charade_sta', type=click.Path(exists=True))
@click.argument('input_charade_v1', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_charade_sta, input_charade_v1, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # load charade-STA file
    data = []
    with open(input_charade_sta, 'r') as f:
        for line in f:
            meta, desc = line.split('##')
            desc = desc.rstrip()
            v_id, s_sec, e_sec = meta.split()
            data.append((v_id, float(s_sec), float(e_sec), desc))
    
    df_sta = pd.DataFrame(data,
                          columns=['id', 'start (sec)', 'end (sec)', 'description']
                         )
    
    # load charade-v1 file
    df_v1 = pd.read_csv(input_charade_v1,
                       usecols=['id', 'length'])
    
    # get length of the input video
    merge_df = pd.merge(df_sta, df_v1, how='left', on='id')
    
    # save merged data
    merge_df.to_csv(output_filepath, index=False)

def save_vfeat_h5(split):
    logger = logging.getLogger(__name__)
    logger.info('save visual features in hdf5 format')
    
    vfeat_root = 'data/raw/Charades_v1_features_rgb/'
    df = pd.read_csv(f'data/processed/{split}.csv')

    with h5py.File(f"data/processed/{split}_vfeat.h5", "w") as hf:

        for video in tqdm(df['id'].unique()):
            dir_name = vfeat_root+f"{video}"
            f_files = os.listdir(dir_name)
            f_files.sort()
            v_feat = [np.loadtxt(f"{dir_name}/{f}") for f in f_files]
            v_feat = np.vstack(v_feat).astype('f')

            hf.create_dataset(video, data=v_feat)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
