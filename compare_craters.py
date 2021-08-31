 #! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: compare_craters
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-19
#
# Purpose: To compare crater depth/diameter measurements from individual MLA tracks and gridded Topography
#

#
# Required Inputs: .csv of data
#
# Updates: 2021-09-31:Document and Clean code
#
#
##
##
##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
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
index_nradar_bright_mla = np.where(radar_bright_mla!='Yes')

name_mla = dd_data_mla[:,0]

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
index_nradar_bright_nancy = np.where(radar_bright_nancy!='Yes')

name_nancy = dd_data_nancy[:,0]

################################################################################
###### finding radar-bright data _mla ##########################################
index_radar_bright_mla = np.where(radar_bright_mla=='Yes')
longitude_radar_bright_mla = longitude_mla[index_radar_bright_mla]
latitude_radar_bright_mla = latitude_mla[index_radar_bright_mla]
depth_radar_bright_mla = depth_mla[index_radar_bright_mla]
diameter_radar_bright_mla = diameter_mla[index_radar_bright_mla]

index_not_radar_bright_mla = np.where(radar_bright_mla!='Yes')
longitude_not_radar_bright_mla = longitude_mla[index_not_radar_bright_mla]
latitude_not_radar_bright_mla = latitude_mla[index_not_radar_bright_mla]
depth_not_radar_bright_mla = depth_mla[index_not_radar_bright_mla]
diameter_not_radar_bright_mla = diameter_mla[index_not_radar_bright_mla]

################################################################################
###### finding radar-bright data _nancy ##########################################
index_radar_bright_nancy = np.where(radar_bright_nancy=='Yes')
longitude_radar_bright_nancy = longitude_nancy[index_radar_bright_nancy]
latitude_radar_bright_nancy = latitude_nancy[index_radar_bright_nancy]
depth_radar_bright_nancy = depth_nancy[index_radar_bright_nancy]
diameter_radar_bright_nancy = diameter_nancy[index_radar_bright_nancy]

index_not_radar_bright_nancy = np.where(radar_bright_nancy!='Yes')
longitude_not_radar_bright_nancy = longitude_nancy[index_not_radar_bright_nancy]
latitude_not_radar_bright_nancy = latitude_nancy[index_not_radar_bright_nancy]
depth_not_radar_bright_nancy = depth_nancy[index_not_radar_bright_nancy]
diameter_not_radar_bright_nancy = diameter_nancy[index_not_radar_bright_nancy]


################################################################################
###### Search for matches ######################################################
depth_diff = np.zeros(1)
diam_diff = np.zeros(1)
diam_avg = np.zeros(1)
lat_diff = np.zeros(1)
lon_diff = np.zeros(1)
dd_diff = np.zeros(1)
radar_bright_diff = np.zeros(1)
name_diff =np.zeros(1)
dd_mla = np.zeros(1)
dd_nancy = np.zeros(1)
for b in range(0,len(name_mla)):
    for c in range(0,len(name_nancy)):
        if name_mla[b]==name_nancy[c]:
            print(name_mla[b],name_nancy[c])
            name_diff = np.vstack([name_diff,name_mla[b]])
            depth_diff = np.vstack([depth_diff,depth_mla[b]-depth_nancy[c]])
            diam_diff = np.vstack([diam_diff,diameter_mla[b]-diameter_nancy[c]])
            diam_avg = np.vstack([diam_avg,np.average([diameter_mla[b],diameter_nancy[c]])])
            #lat_diff = np.vstack([lat_diff,np.average([latitude_mla[b],latitude_nancy[c]])])
            lat_diff = np.vstack([lat_diff,latitude_mla[b]])
            #lon_diff = np.vstack([lon_diff,np.average([longitude_mla[b],longitude_nancy[c]])])
            lon_diff = np.vstack([lon_diff,longitude_mla[b]])
            dd_diff = np.vstack([dd_diff,(depth_mla[b]/diameter_mla[b])-(depth_nancy[c]/diameter_nancy[c])])
            dd_mla = np.vstack([dd_mla,(depth_mla[b]/diameter_mla[b])])
            dd_nancy = np.vstack([dd_nancy,(depth_nancy[c]/diameter_nancy[c])])
            radar_bright_diff = np.vstack([radar_bright_diff,radar_bright_nancy[c]])


name_diff = name_diff[1:,0]
depth_diff = depth_diff[1:,0]
diam_diff = diam_diff[1:,0]
diam_avg = diam_avg[1:,0]
lat_diff = lat_diff[1:,0]
lon_diff = lon_diff[1:,0]
dd_diff = dd_diff[1:,0]
radar_bright_diff = radar_bright_diff[1:,0]

################################################################################
###### Matplotlib formatting ######################################################
index_rb_all = np.where(radar_bright_diff == 'Yes')
index_nrb_all = np.where(radar_bright_diff!='Yes')

loc_sz_rb_all = np.vstack([lon_diff[index_rb_all],lat_diff[index_rb_all],diam_avg[index_rb_all]])
loc_sz_nrb_all = np.vstack([lon_diff[index_nrb_all],lat_diff[index_nrb_all],diam_avg[index_nrb_all]])
np.savetxt(export_location+'loc_sz_rb_all.csv',loc_sz_rb_all.T,delimiter=',',fmt='%f')
np.savetxt(export_location+'loc_sz_rb_all.csv',loc_sz_nrb_all.T,delimiter=',',fmt='%f')

loc_sz_rb_mla = np.vstack([longitude_mla[index_radar_bright_mla],latitude_mla[index_radar_bright_mla],diameter_mla[index_radar_bright_mla]])
loc_sz_nrb_mla = np.vstack([longitude_mla[index_nradar_bright_mla],latitude_mla[index_nradar_bright_mla],diameter_mla[index_nradar_bright_mla]])
loc_sz_rb_nancy = np.vstack([longitude_nancy[index_radar_bright_nancy],latitude_nancy[index_radar_bright_nancy],diameter_nancy[index_radar_bright_nancy]])
loc_sz_nrb_nancy = np.vstack([longitude_nancy[index_nradar_bright_nancy],latitude_nancy[index_nradar_bright_nancy],diameter_nancy[index_nradar_bright_nancy]])

index_name_rb_mla = np.isin(name_mla[index_radar_bright_mla], name_diff,invert=True)
index_name_nrb_mla = np.isin(name_mla[index_nradar_bright_mla], name_diff,invert=True)

index_name_rb_nancy = np.isin(name_nancy[index_radar_bright_nancy], name_diff,invert=True)
index_name_nrb_nancy = np.isin(name_nancy[index_nradar_bright_nancy], name_diff,invert=True)

np.savetxt(export_location+'loc_sz_rb_mla.csv',loc_sz_rb_mla[:,index_name_rb_mla].T,delimiter=',',fmt='%f')
np.savetxt(export_location+'loc_sz_nrb_mla.csv',loc_sz_nrb_mla[:,index_name_nrb_mla].T,delimiter=',',fmt='%f')

np.savetxt(export_location+'loc_sz_rb_nancy.csv',loc_sz_rb_nancy[:,index_name_rb_nancy].T,delimiter=',',fmt='%f')
np.savetxt(export_location+'loc_sz_nrb_nancy.csv',loc_sz_nrb_nancy[:,index_name_nrb_nancy].T,delimiter=',',fmt='%f')

################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)
################################################################################
###### difference in depth/diameter ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lat_diff,dd_diff,'ko',label='All craters')

ax.set_ylabel('Difference in depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

#ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_difference_v_latitude.pdf',format='pdf')
plt.close('all')


################################################################################
###### difference in depth/diameter ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lon_diff,dd_diff,'ko',label='All craters')

ax.set_ylabel('Difference in depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

#ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_difference_v_longitude.pdf',format='pdf')
plt.close('all')


################################################################################
###### difference in depth/diameter ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(diam_avg,dd_diff,'ko',label='All craters')

ax.set_ylabel('Difference in depth/diameter')
ax.set_xlabel('Diameter (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

#ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_difference_v_diameter.pdf',format='pdf')
plt.close('all')


################################################################################
###### difference in depth/diameter histogram ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(dd_diff, bins=np.linspace(-0.05,0.05, num=21),weights=np.ones(len(dd_diff)) / len(dd_diff), facecolor='b')


#ax.set_ylabel('Difference in depth/diameter')
from matplotlib.ticker import PercentFormatter
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel('Difference in depth/diameter')
ax.set_ylabel('Percentage')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

#ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_difference_histogram.pdf',format='pdf')
plt.close('all')
