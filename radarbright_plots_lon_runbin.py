#! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: radarbright_plots_lon_runbins
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-03
#
# Purpose: To calculate the percentage of radar-bright craters in overlapping longitude bins
#

#
# Required Inputs: .csv of data
#
# Updates: 2021-08-31 - Clean and document code
#
#
##
##
##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator
################################################################################


export_location = '../analysis/'
longitude_bin_size = 15
max_diam = 10.0
min_diam = 5.0


################################################################################
file_nancy = '../SBMT_shapefiles/5to10nancy_name_name_lat_lon_diam_rb.csv'
data_source_nancy = '_nancy'
dd_data_nancy = np.loadtxt(file_nancy,dtype='str',delimiter=',')

diameter_nancy = dd_data_nancy[:,4].astype(np.float)


longitude_nancy = dd_data_nancy[:,3].astype(np.float)
latitude_nancy = dd_data_nancy[:,2].astype(np.float)
for k in range(0,len(longitude_nancy)):
    if longitude_nancy[k] > 180:
        longitude_nancy[k]=longitude_nancy[k]-360

radar_bright_nancy = dd_data_nancy[:,8]




################################################################################
###### finding radar-bright data _nancy ##########################################
index_radar_bright_nancy = np.where(radar_bright_nancy=='y')
longitude_radar_bright_nancy = longitude_nancy[index_radar_bright_nancy]
latitude_radar_bright_nancy = latitude_nancy[index_radar_bright_nancy]
diameter_radar_bright_nancy = diameter_nancy[index_radar_bright_nancy]

index_not_radar_bright_nancy = np.where(radar_bright_nancy!='y')
longitude_not_radar_bright_nancy = longitude_nancy[index_not_radar_bright_nancy]
latitude_not_radar_bright_nancy = latitude_nancy[index_not_radar_bright_nancy]
diameter_not_radar_bright_nancy = diameter_nancy[index_not_radar_bright_nancy]

################################################################################
###### binning data in longitude bins _nancy ##########################################
total_lon_bins_nancy = int(360/longitude_bin_size)
middle_bins_lon_nancy = (np.arange(total_lon_bins_nancy)*longitude_bin_size)+(longitude_bin_size/2.0)-(180+(longitude_bin_size/2.0))

count_dd_bin_nancy = np.empty(total_lon_bins_nancy)
count_dd_bin_radar_bright_nancy = np.empty(total_lon_bins_nancy)
count_dd_bin_not_radar_bright_nancy = np.empty(total_lon_bins_nancy)

for i in range(0,total_lon_bins_nancy):
    print(i*longitude_bin_size)
    print((i+1)*longitude_bin_size)

    index_lon_bin_nancy = np.where((longitude_nancy>(middle_bins_lon_nancy[i]-longitude_bin_size)) & (longitude_nancy<(middle_bins_lon_nancy[i]+longitude_bin_size)))
    count_dd_bin_nancy[i] = len(diameter_nancy[index_lon_bin_nancy])

    index_lon_bin_radar_bright_nancy = np.where((longitude_radar_bright_nancy>(middle_bins_lon_nancy[i]-longitude_bin_size)) & (longitude_radar_bright_nancy<(middle_bins_lon_nancy[i]+longitude_bin_size)))
    count_dd_bin_radar_bright_nancy[i] = len(diameter_radar_bright_nancy[index_lon_bin_radar_bright_nancy])

    index_lon_bin_not_radar_bright_nancy = np.where((longitude_not_radar_bright_nancy>(middle_bins_lon_nancy[i]-longitude_bin_size)) & (longitude_not_radar_bright_nancy<(middle_bins_lon_nancy [i]+longitude_bin_size)))
    count_dd_bin_not_radar_bright_nancy[i] = len(diameter_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy])
    print(count_dd_bin_not_radar_bright_nancy[i])



################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)




################################################################################
###### count d/D versus longitude -180 to 180 _nancy####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,count_dd_bin_radar_bright_nancy,'ko',label='Non-radar-bright craters')
ax.plot(middle_bins_lon_nancy,count_dd_bin_not_radar_bright_nancy,'bo',label='Radar-bright craters')

ax.set_xlim(-180,181)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Number of craters measured')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'count_sbmt_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')

################################################################################
###### percentage radar-bright versus longitude -180 to 180 _nancy####################################################
fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,((count_dd_bin_radar_bright_nancy/(count_dd_bin_not_radar_bright_nancy+count_dd_bin_radar_bright_nancy))*100),'ko',label='Percentage measured radar-bright')

ax.set_xlim(-180,181)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-180, 210, 45)))
ax.set_ylabel('% of craters that are radar-bright')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
plt.savefig(export_location+'percentage_sbmt_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')
