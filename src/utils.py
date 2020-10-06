#!/usr/bin/env python3

import glob
import os.path as osp

# Some utils for sequential training
def find_most_recent_path(base):
    r""" Gets most recent subfile."""
    list_of_files = glob.glob(f"{base}/*")
    return max(list_of_files, key=osp.getctime)

def find_data_path(base, task_name):
    r""" Find subdirectory that matches task"""
    list_of_paths = glob.glob(f"{base}/*")
    list_of_files = [osp.split(s)[1] for s in list_of_paths]
    list_of_lower_files = [s.lower() for s in list_of_files]
    index = list_of_lower_files.index(task_name.lower())
    return list_of_paths[index]
