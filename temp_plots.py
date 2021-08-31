 #! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: temp_plots
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-03
#
# Purpose: To plot the maxT, avgT and minMaxT versus latitude (and longitude)
#

#
# Required Inputs: .csv of data
#
# Updates: 2021-08-31 - Clean and document codes
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


radar_bright_nancy = dd_data_nancy[:,1]
index_radar_bright_nancy = np.where(radar_bright_nancy=='Yes')
not_index_radar_bright_nancy = np.where(radar_bright_nancy!='Yes')

avgT_nancy = dd_data_nancy[:,25].astype(np.float)

maxT_nancy = dd_data_nancy[:,28].astype(np.float)

Tdepth_nancy = dd_data_nancy[:,31].astype(np.float)

minmaxT_nancy = dd_data_nancy[:,27].astype(np.float)

################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)


################################################################################
###### avg T versus longitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_nancy[not_index_radar_bright_nancy],avgT_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(longitude_nancy[index_radar_bright_nancy],avgT_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-200,200],[110,110],':ko')

ax.set_xlim(-181,181)
ax.set_ylim(50,300)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-180, 210, 60)))
ax.set_ylabel('Average Temperature (K)')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'avgT_v_longitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy[not_index_radar_bright_nancy],avgT_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(latitude_nancy[index_radar_bright_nancy],avgT_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-50,200],[110,110],':ko')

ax.set_ylabel('Average Temperature (K)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})
ax.xaxis.set_major_locator(FixedLocator(np.arange(75, 90, 2)))
ax.set_ylim(50,300)
ax.set_xlim(75,85)

plt.tight_layout()
plt.savefig(export_location+'avgT_v_latitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')


################################################################################
###### avg T versus longitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_nancy[not_index_radar_bright_nancy],maxT_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(longitude_nancy[index_radar_bright_nancy],maxT_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-200,200],[110,110],':ko')

#ax.set_xlim(-35,121)
ax.set_xlim(-181,181)
ax.set_ylim(50,300)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-180, 210, 60)))
ax.set_ylabel('Maximum Temperature (K)')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'maxT_v_longitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy[not_index_radar_bright_nancy],maxT_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(latitude_nancy[index_radar_bright_nancy],maxT_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-50,200],[110,110],':ko')

ax.set_ylabel('Maximum Temperature (K)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})
ax.xaxis.set_major_locator(FixedLocator(np.arange(75, 90, 2)))
ax.set_ylim(50,300)
ax.set_xlim(75,85)

plt.tight_layout()
plt.savefig(export_location+'maxT_v_latitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')


################################################################################
###### avg T versus longitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_nancy[not_index_radar_bright_nancy],Tdepth_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(longitude_nancy[index_radar_bright_nancy],Tdepth_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-200,200],[110,110],':ko')

#ax.set_xlim(-35,121)
ax.set_xlim(-181,181)
ax.set_ylim(0,2)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-180, 210, 60)))
ax.set_ylabel('Depth below surface (m) ')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'Tdepth_v_longitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy[not_index_radar_bright_nancy],Tdepth_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(latitude_nancy[index_radar_bright_nancy],Tdepth_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')


ax.set_ylabel('Depth to Ice (m)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})
ax.xaxis.set_major_locator(FixedLocator(np.arange(75, 90, 2)))
ax.set_ylim(0,2)
ax.set_xlim(75,85)

plt.tight_layout()
plt.savefig(export_location+'Tdepth_v_latitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')


################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy[not_index_radar_bright_nancy],minmaxT_nancy[not_index_radar_bright_nancy],'ko',label='Non-radar-bright craters')
ax.plot(latitude_nancy[index_radar_bright_nancy],minmaxT_nancy[index_radar_bright_nancy],'b^',label='Radar-bright craters')

ax.plot([-50,200],[110,110],':ko')

ax.set_ylabel('Lowest Maximum Temperature (K)')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})
ax.xaxis.set_major_locator(FixedLocator(np.arange(75, 90, 2)))
ax.set_ylim(50,300)
ax.set_xlim(75,85)

plt.tight_layout()
plt.savefig(export_location+'minmaxT_v_latitude_radarbright_v_nonradarbright.pdf',format='pdf')
plt.close('all')
