# coding: utf-8
# dataset.py
#__last_updated__='2020_09_23'

"""
Module: barchase.data
Purpose: To retieve COVID-19 timeseries from
         JHU CSSEGIS data repo;

"""
from pathlib import Path
import numpy as np
import itertools
import pandas as pd
from datetime import datetime as dtm

from barchase import utils
# ....................................................................
round3 = utils.round3
DIR_IMG = utils.get_project_dirs()

TS_REPO = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
TS_DATA_PATH = 'csse_covid_19_data/csse_covid_19_time_series/'
TS_PREFIX = 'time_series_covid19_' 
TS_URL = F'{TS_REPO}{TS_DATA_PATH}{TS_PREFIX}'
TS_KIND = ['confirmed', 'deaths', 'recovered']
NO_RECOVERED_US = 'The JHU CSSE covid-19 repo does not have \
recovered data for the US.'
TS_GH = 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data'

global_new_names = {'Taiwan*':'Taiwan',
                    'Saint Vincent and the Grenadines':
                    'Saint Vincent & Grenadines'}


def all_barchase_dirs(img_dir=DIR_IMG):
    '''For use in notebooks.'''
    def ds_kinds(lsts):
        return list(itertools.product(*lsts))
    
    all_dirs = {}
    tups = ds_kinds([['confirmed', 'deaths', 'recovered'],
                     ['global', 'US']])
    tups.pop()
    for tup in tups:
        globl = (tup[1] == 'global')
        dirpath, _ = get_bar_chase_folder(img_dir=img_dir,
                                          ts_kind=tup[0],
                                          global_ts=globl)
        all_dirs[tup] = dirpath
    return all_dirs


def check_kind(ts_kind):
    if ts_kind is not None:
        ts_kind = ts_kind.lower()
        return ts_kind in TS_KIND
    print(F"'ts_kind' is None.")
    return False


def check_recovered_US(ts_kind, global_ts):
    '''As of 9/2020'''
    if not global_ts and ts_kind == 'recovered':
        print(NO_RECOVERED_US)
        return False
    return True


def get_bar_chase_folder(img_dir=DIR_IMG,
                         ts_kind='deaths',
                         global_ts=True):
    '''
    ts_kind (str): one of ['confirmed', 'deaths', 'recovered'].
    '''
    created = False
    suffix = '_global'
    
    if not check_kind(ts_kind):
        return
    ts_kind = ts_kind.lower()
    if not check_recovered_US(ts_kind, global_ts):
        return
    if not global_ts:
        suffix = '_US'
    
    dir_img = Path(img_dir).joinpath('barh_chase')
    chase_pics = dir_img.joinpath(ts_kind + suffix)
        
    if not chase_pics.exists():
        Path.mkdir(chase_pics)
        created = True
        
    return chase_pics, created


def get_most_recent_pic(pic_folder, kind='png'):
    '''
    Return (pic date, path) if found, else (None, None).
    :param kind (str): image file extension, e.g.: png, gif, jpeg...
    '''
    recent = None
    fname = None
    i = 0

    kind = kind.lower().replace('.','')
    j = len(kind) + 1
    
    # gif file name format: folder + '%Y_%m_%d' +'.gif'
    # date part start: len('2020_06_01') = 10
    if kind == 'gif':
        i = -(10 + j)
    
    # Note, WIP: st_size of empty folder may not be 0 on some OS.
    if pic_folder.stat().st_size:
        pics = sorted(pic_folder.glob('*.' + kind), reverse=True)
        for fname in pics:
            # get 1st file that complies with format
            try:
                recent = dtm.strptime(fname.name[i:-j],'%Y_%m_%d')
                break
            except ValueError:
                continue
                    
    return recent, fname


def get_ds_latest_date(ts_kind='deaths', global_ts=True):
    suffix = '_global.csv'
    date_idx = 4
    
    if not global_ts: #reset
        suffix = '_US.csv'
        date_idx = 12
        
    url = TS_URL + F'{ts_kind}{suffix}'
    header = pd.read_csv(url, nrows=0)
    if header is not None:
        return dtm.strptime(header.columns[-1],
                            '%m/%d/%y')
    return None


def get_gif(ts_kind='deaths', global_ts=True, img_dir=DIR_IMG):
    '''
    Return None or path to gif if found.
    '''
    if not check_kind(ts_kind):
        return
    ts_kind = ts_kind.lower()

    if not check_recovered_US(ts_kind, global_ts):
        return
    
    suffix = '_global.csv'
    date_idx = 4
    
    if not global_ts: #reset
        suffix = '_US.csv'
        date_idx = 12
        
    png_folder, new = get_bar_chase_folder(img_dir, ts_kind, global_ts)
    if new:
        return None

    # look for related gif: date, pic
    local_date, local_gif = get_most_recent_pic(png_folder, kind='gif')
    if local_date is None:
        return None
    
    # matching dates?
    latest_date = get_ds_latest_date(ts_kind, global_ts)
    if latest_date == local_date:
        return local_gif
    else:
        return None


def get_row_colors(df):
    '''
    Use latest time point to assign a color to all rows in df
    so that a row's bar color persist across the time series.
    The colormap is tab20, so it will cycle over the rows.
    df: a pandas.DataFrame whose index are strings and columns
    datetime or compatible with `df.columns.max()`.
    Return: a dict.
    '''
    from matplotlib import cm
    import matplotlib as mpl
    
    if 'row_color' in df.columns:
        return

    n = df.shape[0]
    t20 = [mpl.colors.to_hex(c) for c in cm.tab20(range(1, n+1))]
    # create color dict:
    idx = df[df.columns.max()].sort_values(ascending=False).index

    return dict(zip(idx, t20))


def assign_colors(df):
    '''
    df: a pandas.DataFrame whose index are strings.
    Return: df with added 'row_color' column.
    '''
    color_dict = get_row_colors(df)
    df['row_color'] = df.index.map(color_dict)
    return df


def aggreg_by(df, by='Country/Region', 
              drop_list=['Lat', 'Long']):
    '''
    Aggregate the values using the `by` column.
    :param df: pandas.DataFrame of raw data from CSSE.
    :return: pandas.DataFrame of totals without geolocation.
    '''
    df0 = df.copy()
    df0.drop(drop_list, axis=1, inplace=True, errors='ignore')
    df_agg = df0.groupby(by).sum()
    # Only keep non-0 cols:
    msk = df_agg.mean(axis=0)
    df_agg = df_agg.loc[:, msk[msk!=0].index]
    
    return df_agg


def cols_to_dt(df, from_idx=0):
    '''
    Convert column strings to dates.
    All df columns are date strings if from_idx=0,
    else select cols[from_idx:].
    '''   
    if from_idx is None:
        from_idx = 0

    cols = df.columns.to_list()
    dttype = 'datetime64[ns]'
    if from_idx == 0:
        df.columns = pd.Series(cols).astype(dttype).to_list()
        return df
    
    coldates = pd.Series(cols[from_idx:]).astype(dttype).to_list()
    df.columns = cols[:from_idx] + coldates
    return df


#.... Data & DataFrame processing  ........................
def get_JHU_CSSE_covid_ts(ts_kind='deaths',
                          global_ts=True,
                          img_dir=DIR_IMG):
    '''    
    Return a tuple: (df, image folder):
        df :: the time-series data from JHU CSSE repo as a
              pd dataframe augmented with a 'row_color' column;
        image folder :: the corresponding image folder.
    :param ts_kind (str): one of 'confirmed', 'deaths', 'recovered'.
    :param global_ts (bool): US timeseries if False.
    :param img_dir (Path): path to barchase pic folder: 
        ../images/barh_chase/[kind]/[global|US]/
    '''
    if not check_kind(ts_kind):
        return
    ts_kind = ts_kind.lower()
        
    if not check_recovered_US(ts_kind, global_ts):
        return
    
    suffix = '_global.csv'
    date_idx = 4
     
    if not global_ts:
        suffix = '_US.csv'
        date_idx = 12
        
    img_folder, new = get_bar_chase_folder(img_dir, ts_kind, global_ts)
    
    url = TS_URL + F'{ts_kind}{suffix}'
    df = pd.read_csv(url)
    
    if df is not None:
        if global_ts:
            col = 'Country/Region'
            df[col].replace(global_new_names, inplace=True)
            drops = ['Lat', 'Long']
            max_len = 26
        else:
            col = 'Province_State'
            drops = ['UID','iso2','iso3','code3','FIPS','Admin2',
                    'Country_Region','Lat','Long_','Combined_Key',
                     'Population']
            max_len = 14

        agg_df = aggreg_by(df, by=col, drop_list=drops)
        # after aggregation: cols :: all dates
        agg_df = cols_to_dt(agg_df)
        # assign_colors() adds a column, 'row_color':
        agg_df = assign_colors(agg_df)
        
        return agg_df, img_folder
    else:
        print(F'Could not return data. Check the site:\n{TS_GH}')
        return None, img_folder
        