 #! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: dD_plots_secondaries
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-03
#
# Purpose: To check the depth/diameter of identified secondaries compared to the general depth/diameter population
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
from pylab import *
#from scipy import stats
################################################################################

export_location = '../analysis/'
mla_data = 'no'
nancy_data = 'yes'

max_diam = 10.0
min_diam = 5.0
################################################################################


file_mla = '../crater_measurements/polar_hannah.csv'
data_source = '_mla'
dd_data_mla = np.loadtxt(file_mla,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_mla[:,5].astype(np.float) < max_diam) & (dd_data_mla[:,5].astype(np.float) > min_diam))
dd_data_mla = dd_data_mla[index_diam,:]
dd_data_mla = dd_data_mla[0,:,:]

depth_mla = dd_data_mla[:,7].astype(np.float)
depth_error_mla = dd_data_mla[:,8].astype(np.float)

diameter_mla = dd_data_mla[:,5].astype(np.float)
diameter_error_mla = dd_data_mla[:,6].astype(np.float)

longitude_mla = dd_data_mla[:,2].astype(np.float)
latitude_mla = dd_data_mla[:,1].astype(np.float)

radar_bright_mla = dd_data_mla[:,3]

index_radar_bright_mla = np.where(radar_bright_mla=='Yes')

################################################################################

file = '../crater_measurements/depth_diameter_spreadsheet_nancy.csv'
data_source = '_nancy'
dd_data_nancy = np.loadtxt(file,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_nancy[:,22].astype(np.float) < max_diam) & (dd_data_nancy[:,22].astype(np.float) > min_diam))
dd_data_nancy = dd_data_nancy[index_diam,:]
dd_data_nancy = dd_data_nancy[0,:,:]

depth_nancy = dd_data_nancy[:,23].astype(np.float)
depth_error_nancy = np.sqrt((((depth_nancy-dd_data_nancy[:,6].astype(np.float))**2)+((depth_nancy-dd_data_nancy[:,10].astype(np.float))**2)+((depth_nancy-dd_data_nancy[:,14].astype(np.float))**2))/3)

diameter_nancy = dd_data_nancy[:,22].astype(np.float)
diameter_error_nancy = np.sqrt((((diameter_nancy-dd_data_nancy[:,15].astype(np.float))**2)+((diameter_nancy-dd_data_nancy[:,16].astype(np.float))**2)+((diameter_nancy-dd_data_nancy[:,17].astype(np.float))**2))/3)

longitude_nancy = dd_data_nancy[:,36].astype(np.float)
latitude_nancy = dd_data_nancy[:,35].astype(np.float)
for k in range(0,len(longitude_nancy)):
    if longitude_nancy[k] < 0:
        longitude_nancy[k]=360+longitude_nancy[k]

radar_bright_nancy = dd_data_nancy[:,1]
index_radar_bright_nancy = np.where(radar_bright_nancy=='Yes')

################################################################################

file = '../crater_measurements/depth_diameter_spreadsheet_nancy_secondaries.csv'
data_source = '_sec'
dd_data_sec = np.loadtxt(file,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_sec[:,22].astype(np.float) < max_diam) & (dd_data_sec[:,22].astype(np.float) > min_diam))
dd_data_sec = dd_data_sec[index_diam,:]
dd_data_sec = dd_data_sec[0,:,:]

depth_sec = dd_data_sec[:,23].astype(np.float)
depth_error_sec = np.sqrt((((depth_sec-dd_data_sec[:,6].astype(np.float))**2)+((depth_sec-dd_data_sec[:,10].astype(np.float))**2)+((depth_sec-dd_data_sec[:,14].astype(np.float))**2))/3)

diameter_sec = dd_data_sec[:,22].astype(np.float)
diameter_error_sec = np.sqrt((((diameter_sec-dd_data_sec[:,15].astype(np.float))**2)+((diameter_sec-dd_data_sec[:,16].astype(np.float))**2)+((diameter_sec-dd_data_sec[:,17].astype(np.float))**2))/3)

longitude_sec = dd_data_sec[:,36].astype(np.float)
latitude_sec = dd_data_sec[:,35].astype(np.float)
for k in range(0,len(longitude_sec)):
    if longitude_sec[k] < 0:
        longitude_sec[k]=360+longitude_sec[k]

radar_bright_sec = dd_data_sec[:,1]
index_radar_bright_sec = np.where(radar_bright_sec=='Yes')

################################################################################


file_2016 = '../crater_measurements/MLA_crater_synced_w_appendix_for_html.csv'
data_source = '_2016'
dd_data_2016 = np.loadtxt(file_2016,dtype='str',delimiter=',', usecols=range(6))
#dd_data_2016 = np.loadtxt(file_2016,dtype='float',delimiter=',', usecols=range(6))

index_diam = np.where((dd_data_2016[:,2].astype(np.float) < max_diam) & (dd_data_2016[:,2].astype(np.float) > min_diam))
dd_data_2016 = dd_data_2016[index_diam,:]
dd_data_2016 = dd_data_2016[0,:,:]

depth_2016 = dd_data_2016[:,4].astype(np.float)
depth_error_2016 = dd_data_2016[:,5].astype(np.float)

diameter_2016 = dd_data_2016[:,2].astype(np.float)
diameter_error_2016 = dd_data_2016[:,3].astype(np.float)

longitude_2016 = dd_data_2016[:,1].astype(np.float)
latitude_2016 = dd_data_2016[:,0].astype(np.float)



#######################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)

################################################################################
###### d/D of all craters _mla and _nancy######################################################
diam_line = np.array([0.11,50])
depth_line = 0.2*diam_line

mpl.rcParams['axes.linewidth'] = 1
fig = plt.figure()
ax = fig.add_subplot(111)
index_diam = np.where(diameter_mla < 10)


ax.scatter(longitude_2016,depth_2016/diameter_2016,c='grey',marker='v',label='Susorney et al., 2016',alpha=1)
ax.plot(longitude_mla[index_diam],depth_mla[index_diam]/diameter_mla[index_diam],'ko',label='This study ',alpha=1,markeredgewidth=1.0)
#ax.errorbar(diameter_mla[index_diam],depth_mla[index_diam], yerr=depth_error_mla[index_diam],xerr=diameter_error_mla[index_diam],fmt='ko',capsize=5,alpha=0.7)

ax.plot(longitude_nancy,depth_nancy/diameter_nancy,'ko',alpha=1,markeredgewidth=1.0)
#ax.errorbar(diameter_nancy,depth_nancy, yerr=depth_error_nancy,xerr=diameter_error_nancy,fmt='ko',capsize=5,alpha=0.7)

ax.plot(longitude_sec,depth_sec/diameter_sec,'r*',label='Prokofiev Definitive Secondaries',alpha=1,markeredgewidth=1.0,zorder=10, markersize=15)

#ax.errorbar(diameter_sec,depth_sec, yerr=depth_error_sec,xerr=diameter_error_sec,fmt='ro',capsize=5,alpha=1,barsabove=True,zorder=10)


#ax.plot(diam_line,depth_line,':ko')


ax.set_xlim(-1,121)
ax.set_ylim(0.05,0.27)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-0, 150, 30)))
ax.set_ylabel('depth/diameter ratio ')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'dD_all_mla_nancy_secondaries.pdf',format='pdf')
plt.close('all')
