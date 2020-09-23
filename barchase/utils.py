# coding: utf-8

# utils.py

from pathlib import Path
from functools import partial
import numpy as np


round3 = partial(np.round, decimals=3)


def get_project_dirs(which=['images'],
                     nb_folder='notebooks',
                     use_parent=True):
    dir_lst = []
    if Path.cwd().name.startswith(nb_folder) or use_parent:
        dir_fn = Path.cwd().parent.joinpath
    else:
        dir_fn = Path.cwd().joinpath
        
    for d in which:
        DIR = dir_fn(d)
        if not DIR.exists():
            Path.mkdir(DIR)
        dir_lst.append(DIR)
    if len(which) == 1:
        return dir_lst[0]
    return dir_lst


def folder_info(fld, created=None):
    print(F'Folder: {fld}\nNew: {created}; Size: {fld.stat().st_size}')
    

def show_gif(fname):
    """
    Return a HTML <img> with source from fname if found.
    
    """
    if fname is None:
        return None
    
    import base64
    
    if Path(fname).exists:
        with open(fname, 'rb') as fd:
            b64 = base64.b64encode(fd.read()).decode('utf-8')
        return f'<img src="data:image/gif;base64,{b64}" alt="gif" />'
    else:
        return '<h4>File not found!</h4>'
    