import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import shutil

from glob import glob

c_dir = './images/PelvisImagesDICOM/'

def get_dicom_fail_paths(): # run in the pelvic_fx_AI dir
    fail_paths = []
    for pt in glob(os.path.join(c_dir, '*')): #each patient
        lvl2 = glob(os.path.join(pt, '*'))

        for j in lvl2: #each accession
            lvl3 = glob(os.path.join(j, '*'))

            for k in lvl3: #each series
                lvl4 = glob(os.path.join(k, '*'))

                for l in lvl4: #each dicom
                    try:
                        img_array = pydicom.dcmread(l).pixel_array
                    except:
                        fail_paths.append(l)

    return fail_paths

def copy_dicom_dirs():
    fail_paths = []
    with open('./misc_old_files/fail_paths.txt') as f:
        for i in f:
            fail_paths.append(i.strip()) #also gets rid of newline at the end


    # to copy dicom into a temp working directory
    for i in fail_paths:
        dir_path = re.search(r'(.*)/(\d*\.\d*)$', i).group(1)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        shutil.copy(os.path.join('../', i), i)

