# coding: utf-8
# plotting.py
#__last_updated__='2020_09_23'

"""
Module: covid19
Purpose: To incrementally process COVID-19 timeseries from
         JHU CSSEGIS data repo;
         Save a ranked barplot as a png figure for each
         time point;
         Create a gif from a folder's png collection.
"""
from pathlib import Path
from functools import partial    
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from matplotlib.animation import FuncAnimation, PillowWriter

from barchase import utils
from barchase import dataset
# .................................................................

round3 = utils.round3
DIR_IMG = utils.get_project_dirs()

mystyle = {'axes.titlesize': 20,
           'axes.labelsize': 16,
           #'axes.labelweight': 'bold',
           'xtick.labelsize': 14,
           'ytick.labelsize': 14,
           'xtick.minor.size': 0,
           'ytick.major.size': 0,
           'ytick.minor.size': 0}

plt.style.use(['seaborn-muted', mystyle])


def f_billions(x, pos):
    return '%1.1fB' % (x * 1e-9)

def f_millions(x, pos):
    return '%1.1fM' % (x * 1e-6)

def f_thousands(x, pos):
    return '%1.1fK' % (x * 1e-3)

def f_nums(x, pos):
    if x >= 10:
        s = F'{x:.1f}'
    else:
        s = F'{x:n}'
    return s

def large_num_format(tick_val, pos):
    """
    Return the formatted string for Billions, Millions, Thousands if
    it is that large, else, return locale formatted string.
    Notes:
    Precision = 1;
    For use with mpl.ticker.FuncFormatter.
    Works well when the range of value is very wide; See example:
    https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    """
    billions, millions, thousands = 1e9, 1e6, 1e3
  
    if tick_val >= billions:
        return f_billions(x, pos)
    if tick_val >= millions:
        return f_millions(x, pos)
    if tick_val >= thousands:
        return f_thousands(x, pos)
    # all others:
    return f_nums(x, pos)


def despine(which=['top','right']):
    '''
    which ([str])): 'left','top','right','bottom'.
    '''
    ax = plt.gca()
    for side in which:
        ax.spines[side].set_visible(False)


def get_pixel_size(text):
    '''
    Return (width, height) of text.
    >"As in the Adobe Font Metrics File Format Specification, all 
      dimensions are given in units of 1/1000 of the scale factor 
      (point size) of the font being used."
    '''
    from matplotlib.afm import AFM
    
    afm_path = Path(mpl.get_data_path(), 
                    'fonts', 'afm', 'ptmr8a.afm')
    with afm_path.open('rb') as fh:
        afm = AFM(fh)
    return afm.string_width_height(text)


def autolabel_bars(rects,
                   orient='h', frmt='{}'): 
    '''
    Attach the bar value to each bar in `rects`:
    Inside the bar (if room permits) if `orient='h'`,
    else on top.
    '''
    orient = orient.lower()
    ax = plt.gca()
    
    fontdict = dict(size=12, weight=600)
    if orient=='h':
        fontdict.update([('color', 'w')])
    else:
        fontdict.update([('color', 'k')])

    tiny_bar_offset = 55
    default_h_offset = -5
    pix_conv = 19 # 19px for 14pt # ad-hoc factor
    
    for rect in rects:       
        # preset values for each bar:
        txt = ''  # no label for 0 values
        h_offset = default_h_offset
        set_to_black = False
        
        if orient=='h':
            val = round3(rect.get_width())
            
            if val > 0:
                txt = frmt.format(val)
                # for positioning the string:
                txt_w = round3(get_pixel_size(txt)[0]/ pix_conv)
                xt_w = round3(rect.get_extents().x1)
 
                if xt_w <= (txt_w - h_offset*6):
                    h_offset = tiny_bar_offset
                    set_to_black = True
            else:
                h_offset = 0
            
            xy_loc = (val, rect.get_y() + rect.get_height() / 2)
            xy_txt = (h_offset, 0)
            h_align = 'right'
            v_align = 'center'
            
            if set_to_black:
                fontdict.update([('color', 'darkgrey')])
        else:
            # vertical bar: simpler!
            val = rect.get_height()
            if val > 0:
                txt = frmt.format(val)
            
            xy_loc = (rect.get_x() + rect.get_width() / 2, val)
            xy_txt = (0, 3)
            h_align = 'center'
            v_align = 'bottom'

        ax.annotate(txt,
                    xy=xy_loc,
                    xytext= xy_txt,
                    textcoords='offset points',
                    ha=h_align, va=v_align,
                    **fontdict)
        if set_to_black: # reset
                fontdict.update([('color', 'w')])
                
    return


def barh_topN(day, df=None,
               ts_kind='COVID-19 deaths',
               per_multiple='thousands',
               N=10,
               data_source='Data: Johns Hopkins CSSE',
               pad_len=26,
               save_as='date',
               save_dir=None,
               replace=False,
               fig_size=(10,5),
               show_fig=False):
    '''
    day (datetime): date column for filtering df.
    df (pd.DataFrame): df of time series [col name=timestamp] + [row_color]
    ts_kind (str): for x-axis label.
    data_source (str): to annote the plot.
    per_multiple (str): 'thousands', 'millions', 'billions'.
    n (int, 10): top n rows.
    save_as (str): figure name to save in save_dir;
                   Can be 'date': if so, the formatted day
                   ('%m_%d_%y') is the png file name.
    save_dir (pathlib.Path): directory for bar plot images.
    replace (bool): replace figure if it exists.
    '''
    # Initial checks to avoid processing:
    if df is None:
        return

    if save_dir is None and save_as is not None:
        raise ValueError('barh_topN :: Cannot save figure: save_dir is None.')
    
    if N is None:
        N = 10
        
    # cases:
    factor = 1
    if per_multiple is not None:
        per_multiple = per_multiple.lower()
        if per_multiple == 'thousands':
            factor = 1e-3
        elif per_multiple == 'millions':
            factor = 1e-6
        elif per_multiple == 'billions':
            factor = 1e-9

    top10 = df[[day,'row_color']].sort_values(by=day).tail(N)

    idx = top10.index
    
    xlab = ''
    if ts_kind:
        xlab = F'{ts_kind}'
    if factor != 1:
        xlab += F' ({per_multiple})'

    def y_names(x):
        if top10.loc[x, day] == 0:
            x = ' ' * pad_len
        return F'{x: >{pad_len}}'
    
    #with plt.rc_context(mystyle):
    fig, ax = plt.subplots(figsize=fig_size)

    bars = ax.barh(idx, top10[day]*factor,
                       color=top10['row_color'].values.tolist())

    lbls = [y_names(lbl) for lbl in idx.to_list()]
    plt.yticks(range(N), lbls)

    autolabel_bars(bars, frmt='{:,.3f}')

    # only show ticks>=0 & exclude last:
    ax.set_xticks(ax.get_xticks()[1:-1])

    formatter = FuncFormatter(f_nums)
    ax.xaxis.set_major_formatter(formatter)
    # since using style context, fontsize not applied?
    ax.set_xlabel(xlab)

    ax.grid(which='major', axis='x',
            ls='--', lw=.5, c='lightgrey')
    # needed bc zorder for grid not working, issue #5045:
    ax.set_axisbelow(True)

    despine(['left','top','right'])
    ax.spines['bottom'].set_color('darkgrey')

    # annotations:
    anno_kwargs = dict(xy=(0.7, 0.3),
                       xycoords='figure fraction',
                       fontsize=20)
    anno_kwargs['text'] = day.strftime('%b %d %Y')
    ax.annotate(**anno_kwargs, fontweight=600)
        
    if data_source:
        # update before reuse:
        anno_kwargs['text'] = F'{data_source}'
        anno_kwargs['xy'] = (0.7, 0.2)
        anno_kwargs['fontsize'] = 12
        ax.annotate(**anno_kwargs,
                    fontweight=500,
                    bbox=dict(fc='w', ec='w'))

    plt.tight_layout()

    if save_as is not None:
        if save_as == 'date':
            # formated for correct sorting:
            save_as = day.strftime('%Y_%m_%d') + '.png'
        else:
            if not Path(save_as).suffix:
                save_as += '.png'
        fname = save_dir.joinpath(save_as)
        if not fname.exists or replace:
            plt.savefig(fname)

    if not show_fig:
        plt.close()
    else:
        plt.show()

    return
    

def update_gif(kind='deaths',
               global_ts=True,
               x_label='COVID-19 global deaths',
               per_multiple='thousands',
               img_dir=DIR_IMG,
               replace_pics=True,
               plot_last_day=False):
    '''
    Wrapper function for incremental processing of png files and
    gif creation from timeseries data on JHU CSSE repo.
    Params: see dataset.get_JHU_CSSE_covid_ts(), and:
    replace_pics (bool, True): redo the bar chart and save if True
    plot_last_day (bool, False): Display the last day of the series.
    '''
    if img_dir is None:
        print('update_gif :: img_dir: Image folder path required.')
        return
    
    df, png_folder = dataset.get_JHU_CSSE_covid_ts(ts_kind=kind,
                                                   global_ts=global_ts,
                                                   img_dir=img_dir)
    # new_data? 
    if df is None:
        return None

    # Get the days (col names):
    days = df.columns[:-1].to_list()
    # for naming gif:
    most_recent = days[-1].strftime('%Y_%m_%d')
    gname = F'{png_folder.name}_{most_recent}.gif'
    # for padding y labels:
    max_len = 26 if global_ts else 14 

    if plot_last_day:
        barh_topN(days[-1], df=df, 
                    ts_kind=x_label,
                    pad_len=max_len,
                    per_multiple=per_multiple,
                    save_as=None,
                    show_fig=True)

    # returns the gif file path
    gif_file = get_pics_and_gif(days, barh_topN, df,
                                N=10,
                                gif_fname=gname,
                                save_dir=png_folder,
                                ts_kind=x_label,
                                per_multiple=per_multiple,
                                pad_len=max_len,
                                replace=replace_pics)
    return gif_file, df


def get_pics_and_gif(col_names, bar_func, df, 
                     gif_fname='png.gif',
                     replace=False,
                     **func_args):
    '''
    Defaults: 'deaths', 'thousands', n=10, save_as='date'.
    col_names = days (dt).
    Note: Here df holds the complete time series + row_color;
          -> unfiltered.
    '''
    if df is None:
        print('Dataframe is None, but required.')
        return
    save_dir = func_args.get('save_dir', None)
    ts_kind = func_args.get('ts_kind', 'COVID-19 deaths')
    per_multiple = func_args.get('per_multiple', 'thousands')
    n = func_args.get('N', 10)
    ds_source = func_args.get('data_source', 
                              'Data: Johns Hopkins CSSE')
    pad = func_args.get('pad_len', 26)
    figxy = func_args.get('fig_size',(10,5))

    # wrapper pre-set for saving fig using the remaining
    # parameters: date in `day`, replace:
    barh_wrapper = partial(bar_func,
                           #day::param
                           df=df,
                           ts_kind=ts_kind,
                           per_multiple=per_multiple,
                           N=n,
                           data_source=ds_source,
                           pad_len=pad,
                           fig_size=figxy,
                           save_dir=save_dir,
                           save_as='date',
                           #replace::param
                           show_fig=False)
                          
    print('Saving charts to png (if replace=True)...')
    plot_created = False
    for c in tqdm(col_names):                 
        if replace:
            barh_wrapper(c, replace=True)
        else:
            png = F"{c.strftime('%Y_%m_%d')}.png"
            png_save = Path(save_dir).joinpath(png)
            if not png_save.exists():
                barh_wrapper(c, replace=True)
                plot_created = plot_created or True

    gif_file = save_dir.joinpath(gif_fname)
    # Next: don't redo gif if not necessary:
    if replace or plot_created or not gif_file.exists():
        from PIL import Image as pil_Image
        print('Updating gif')
        pngs = sorted(save_dir.glob('*.png'))

        img, *imgs = [pil_Image.open(f) for f in tqdm(pngs)]
        img.save(fp=gif_file, format='GIF',
                 append_images=imgs, save_all=True,
                 duration=260, loop=1)
        print('Gif created!')

    return gif_file
