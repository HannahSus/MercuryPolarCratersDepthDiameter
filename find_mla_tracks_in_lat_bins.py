##
##
##
# Program: find_mla_tracks_in_lat_bins
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-11
#
# Purpose: To calculate density of MLA tracks in latitude bins per area in bin
#
# Required Inputs: MLA tracks
#
# Updates: 2021-08-31 - Clean and document codes
##
##
##

import numpy as np
import matplotlib.pyplot as plt
import struct
from astropy.io import fits
from data import read_mla_binary, read_sbmt_poly
from geometry import xyz_to_lon_lat
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import download_file
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator
#import cartopy.crs as ccrs
#import cartopy.feature
import matplotlib.path as mpath
import glob
import os
from matplotlib import path, patches
import os.path
from matplotlib import gridspec
import pandas
#
#


mla_npy_directory = '/Users/hsusorney/Research/Data/MESSENGER/mlabin/'

min_lat = 70
max_lat = 90
spacing = 0.5
radius = 2440

num_bins = int((max_lat-min_lat)/spacing)

num_mla = np.zeros([num_bins])

num_area = np.zeros([num_bins])



min_lat_bin = ((np.arange(num_bins)*spacing)+min_lat)

max_lat_bin = (min_lat_bin)+spacing

avg_lat_bin = (min_lat_bin)+(spacing/2)


#bin_area =

export_location = '../analysis/'


print('Running find_mla_tracks_in_lat_bins')

#Use this to identify and find tracks that intersect radar region and place in tracks folder


file_list = glob.glob(mla_npy_directory+'m*.xyzd')
file_list = np.array(file_list)
#Create array to store information about spacing
master_lon_lat = np.empty([2,1])
for n in range(0,len(file_list)):
    print("Reading track = ",str(n),' of ',str(len(file_list)))
    track = file_list[n]
    data = read_mla_binary(track)
    track_lon_lat = np.vstack([data[:,0],data[:,1]])
    master_lon_lat = np.hstack([master_lon_lat,track_lon_lat])


print('Calculating number of MLA points in each bin')
for i in range(0,num_bins):
    index_points = np.where((master_lon_lat[1,:] > min_lat_bin[i]) & (master_lon_lat[1,:] < max_lat_bin[i]))
    lats = master_lon_lat[1,index_points]
    num_mla[i] = len(lats.T)

    big_area = 2*np.pi*(radius**2)*(1-np.cos(np.deg2rad(90-min_lat_bin[i])))
    little_area = 2*np.pi*(radius**2)*(1-np.cos(np.deg2rad(90-max_lat_bin[i])))
    num_area[i] = big_area-little_area




################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)
################################################################################
###### # versus binned latitude ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_lat_bin,num_mla,'ko')

ax.set_ylabel('Number of MLA points')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'num_lat.pdf',format='pdf')
plt.close('all')


################################################################################
###### area versus binned latitude ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_lat_bin,num_area,'ko')

ax.set_ylabel('Area (km^2)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'num_area.pdf',format='pdf')
plt.close('all')


################################################################################
###### density versus binned latitude ####################################################
num_density = num_mla/num_area
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_lat_bin,num_density,'ko')

ax.set_ylabel('Density of MLA (#/km^2)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'num_density.pdf',format='pdf')
plt.close('all')


print("End Running find_mla_tracks_in_lat_bins")
