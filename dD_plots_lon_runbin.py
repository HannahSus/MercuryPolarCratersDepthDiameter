#! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: dD_plots_lon_runbin
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-03
#
# Purpose: To compare depth/diameter measurements in overlapping longitude bins
#   Used in study

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

file_mla = '../crater_measurements/polar_hannah.csv'
data_source_mla = '_mla'
dd_data_mla = np.loadtxt(file_mla,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_mla[:,5].astype(np.float) < max_diam) & (dd_data_mla[:,5].astype(np.float) > min_diam))
dd_data_mla = dd_data_mla[index_diam,:]
dd_data_mla = dd_data_mla[0,:,:]

depth_mla = dd_data_mla[:,7].astype(np.float)
diameter_mla = dd_data_mla[:,5].astype(np.float)

longitude_mla = dd_data_mla[:,2].astype(np.float)
latitude_mla = dd_data_mla[:,1].astype(np.float)

radar_bright_mla = dd_data_mla[:,3]
index_radar_bright_mla = np.where(radar_bright_mla=='Yes')

for k in range(0,len(longitude_mla)):
    if longitude_mla[k] > 180:
        longitude_mla[k]=longitude_mla[k]-360

################################################################################
file_nancy = '../crater_measurements/depth_diameter_spreadsheet_nancy.csv'
data_source_nancy = '_nancy'
dd_data_nancy = np.loadtxt(file_nancy,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_nancy[:,22].astype(np.float) < max_diam) & (dd_data_nancy[:,22].astype(np.float) > min_diam))
dd_data_nancy = dd_data_nancy[index_diam,:]
dd_data_nancy = dd_data_nancy[0,:,:]



depth_nancy = dd_data_nancy[:,23].astype(np.float)
depth_error_nancy = dd_data_nancy[:,8].astype(np.float)

diameter_nancy = dd_data_nancy[:,22].astype(np.float)
diameter_error_nancy = dd_data_nancy[:,6].astype(np.float)

longitude_nancy = dd_data_nancy[:,36].astype(np.float)
latitude_nancy = dd_data_nancy[:,35].astype(np.float)
for k in range(0,len(longitude_nancy)):
    if longitude_nancy[k] > 180:
        longitude_nancy[k]=longitude_nancy[k]-360

radar_bright_nancy = dd_data_nancy[:,1]
index_radar_bright_nancy = np.where(radar_bright_nancy=='Yes')

################################################################################

file = '../crater_measurements/Rubanenko_mercury_data.csv'
dd_data_rub = np.loadtxt(file,dtype='str',delimiter=',',skiprows=1)

index_diam = np.where((dd_data_rub[:,3].astype(np.float)/1000.0 < max_diam) & (dd_data_rub[:,3].astype(np.float)/1000.0 > min_diam))
dd_data_rub = dd_data_rub[index_diam,:]
dd_data_rub = dd_data_rub[0,:,:]

depth_rub = dd_data_rub[:,2].astype(np.float)/1000.0
diameter_rub = dd_data_rub[:,3].astype(np.float)/1000.0
longitude_rub = dd_data_rub[:,1].astype(np.float)
latitude_rub = dd_data_rub[:,0].astype(np.float)

for k in range(0,len(longitude_rub)):
    if longitude_rub[k] > 180:
        longitude_rub[k]=longitude_rub[k]-360

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
###### binning data in longitude bins _mla ##########################################
total_lon_bins_mla = int(360/longitude_bin_size)
middle_bins_lon_mla = (np.arange(total_lon_bins_mla)*longitude_bin_size)+(longitude_bin_size/2.0)-(180+(longitude_bin_size/2.0))

mean_dd_bin_mla = np.empty(total_lon_bins_mla)
mean_dd_bin_radar_bright_mla = np.empty(total_lon_bins_mla)
mean_dd_bin_not_radar_bright_mla = np.empty(total_lon_bins_mla)
mean_dd_bin_rub = np.empty(total_lon_bins_mla)

median_dd_bin_mla = np.empty(total_lon_bins_mla)
median_dd_bin_radar_bright_mla = np.empty(total_lon_bins_mla)
median_dd_bin_not_radar_bright_mla = np.empty(total_lon_bins_mla)
median_dd_bin_rub = np.empty(total_lon_bins_mla)

std_dd_bin_mla = np.empty(total_lon_bins_mla)
std_dd_bin_radar_bright_mla = np.empty(total_lon_bins_mla)
std_dd_bin_not_radar_bright_mla = np.empty(total_lon_bins_mla)
std_dd_bin_rub = np.empty(total_lon_bins_mla)

count_dd_bin_mla = np.empty(total_lon_bins_mla)
count_dd_bin_radar_bright_mla = np.empty(total_lon_bins_mla)
count_dd_bin_not_radar_bright_mla = np.empty(total_lon_bins_mla)
count_dd_bin_rub = np.empty(total_lon_bins_mla)

for i in range(0,total_lon_bins_mla):
    index_lon_bin_mla = np.where((longitude_mla>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_mla<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_mla[i] = np.mean(depth_mla[index_lon_bin_mla]/diameter_mla[index_lon_bin_mla])
    median_dd_bin_mla[i] = np.median(depth_mla[index_lon_bin_mla]/diameter_mla[index_lon_bin_mla])
    std_dd_bin_mla[i] = np.std(depth_mla[index_lon_bin_mla]/diameter_mla[index_lon_bin_mla])
    count_dd_bin_mla[i] = len(depth_mla[index_lon_bin_mla]/diameter_mla[index_lon_bin_mla])

    index_lon_bin_radar_bright_mla = np.where((longitude_radar_bright_mla>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_radar_bright_mla<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_radar_bright_mla[i] = np.mean(depth_radar_bright_mla[index_lon_bin_radar_bright_mla]/diameter_radar_bright_mla[index_lon_bin_radar_bright_mla])
    median_dd_bin_radar_bright_mla[i] = np.median(depth_radar_bright_mla[index_lon_bin_radar_bright_mla]/diameter_radar_bright_mla[index_lon_bin_radar_bright_mla])
    std_dd_bin_radar_bright_mla[i] = np.std(depth_radar_bright_mla[index_lon_bin_radar_bright_mla]/diameter_radar_bright_mla[index_lon_bin_radar_bright_mla])
    count_dd_bin_radar_bright_mla[i] = len(depth_radar_bright_mla[index_lon_bin_radar_bright_mla]/diameter_radar_bright_mla[index_lon_bin_radar_bright_mla])

    index_lon_bin_not_radar_bright_mla = np.where((longitude_not_radar_bright_mla>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_not_radar_bright_mla<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_not_radar_bright_mla[i] = np.mean(depth_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla]/diameter_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla])
    median_dd_bin_not_radar_bright_mla[i] = np.median(depth_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla]/diameter_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla])
    std_dd_bin_not_radar_bright_mla[i] = np.std(depth_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla]/diameter_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla])
    count_dd_bin_not_radar_bright_mla[i] = len(depth_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla]/diameter_not_radar_bright_mla[index_lon_bin_not_radar_bright_mla])

    index_lon_bin_rub = np.where((longitude_rub>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_rub<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_rub[i] = np.mean(depth_rub[index_lon_bin_mla]/diameter_rub[index_lon_bin_mla])
    median_dd_bin_rub[i] = np.median(depth_rub[index_lon_bin_mla]/diameter_rub[index_lon_bin_mla])
    std_dd_bin_rub[i] = np.std(depth_rub[index_lon_bin_mla]/diameter_rub[index_lon_bin_mla])
    count_dd_bin_rub[i] = len(depth_rub[index_lon_bin_mla]/diameter_rub[index_lon_bin_mla])

################################################################################
###### binning data in longitude bins _nancy ##########################################
total_lon_bins_nancy = int(360/longitude_bin_size)
middle_bins_lon_nancy = (np.arange(total_lon_bins_mla)*longitude_bin_size)+(longitude_bin_size/2.0)-(180+(longitude_bin_size/2.0))

mean_dd_bin_nancy = np.empty(total_lon_bins_nancy)
mean_dd_bin_radar_bright_nancy = np.empty(total_lon_bins_nancy)
mean_dd_bin_not_radar_bright_nancy = np.empty(total_lon_bins_nancy)

median_dd_bin_nancy = np.empty(total_lon_bins_nancy)
median_dd_bin_radar_bright_nancy = np.empty(total_lon_bins_nancy)
median_dd_bin_not_radar_bright_nancy = np.empty(total_lon_bins_nancy)

std_dd_bin_nancy = np.empty(total_lon_bins_nancy)
std_dd_bin_radar_bright_nancy = np.empty(total_lon_bins_nancy)
std_dd_bin_not_radar_bright_nancy = np.empty(total_lon_bins_nancy)

count_dd_bin_nancy = np.empty(total_lon_bins_nancy)
count_dd_bin_radar_bright_nancy = np.empty(total_lon_bins_nancy)
count_dd_bin_not_radar_bright_nancy = np.empty(total_lon_bins_nancy)

for i in range(0,total_lon_bins_nancy):
    print(i*longitude_bin_size)
    print((i+1)*longitude_bin_size)

    index_lon_bin_nancy = np.where((longitude_nancy>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_nancy<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_nancy[i] = np.mean(depth_nancy[index_lon_bin_nancy]/diameter_nancy[index_lon_bin_nancy])
    median_dd_bin_nancy[i] = np.median(depth_nancy[index_lon_bin_nancy]/diameter_nancy[index_lon_bin_nancy])
    std_dd_bin_nancy[i] = np.std(depth_nancy[index_lon_bin_nancy]/diameter_nancy[index_lon_bin_nancy])
    count_dd_bin_nancy[i] = len(depth_nancy[index_lon_bin_nancy]/diameter_nancy[index_lon_bin_nancy])

    index_lon_bin_radar_bright_nancy = np.where((longitude_radar_bright_nancy>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_radar_bright_nancy<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_radar_bright_nancy[i] = np.mean(depth_radar_bright_nancy[index_lon_bin_radar_bright_nancy]/diameter_radar_bright_nancy[index_lon_bin_radar_bright_nancy])
    median_dd_bin_radar_bright_nancy[i] = np.median(depth_radar_bright_nancy[index_lon_bin_radar_bright_nancy]/diameter_radar_bright_nancy[index_lon_bin_radar_bright_nancy])
    std_dd_bin_radar_bright_nancy[i] = np.std(depth_radar_bright_nancy[index_lon_bin_radar_bright_nancy]/diameter_radar_bright_nancy[index_lon_bin_radar_bright_nancy])
    count_dd_bin_radar_bright_nancy[i] = len(depth_radar_bright_nancy[index_lon_bin_radar_bright_nancy]/diameter_radar_bright_nancy[index_lon_bin_radar_bright_nancy])

    index_lon_bin_not_radar_bright_nancy = np.where((longitude_not_radar_bright_nancy>(middle_bins_lon_mla[i]-longitude_bin_size)) & (longitude_not_radar_bright_nancy<(middle_bins_lon_mla[i]+longitude_bin_size)))
    mean_dd_bin_not_radar_bright_nancy[i] = np.mean(depth_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy]/diameter_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy])
    median_dd_bin_not_radar_bright_nancy[i] = np.median(depth_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy]/diameter_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy])
    std_dd_bin_not_radar_bright_nancy[i] = np.std(depth_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy]/diameter_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy])
    count_dd_bin_not_radar_bright_nancy[i] = len(depth_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy]/diameter_not_radar_bright_nancy[index_lon_bin_not_radar_bright_nancy])
    print(count_dd_bin_not_radar_bright_nancy[i])
    print(count_dd_bin_not_radar_bright_mla[i])

################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)

###### mean d/D versus binned longitude -180 to 180 _mla####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,mean_dd_bin_mla,'ko',label='All craters')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_mla, yerr=std_dd_bin_mla,fmt='ko',capsize=5)

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### mean d/D versus longitude -180 to 180 _mla ###################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,mean_dd_bin_radar_bright_mla,'ko',label='MLA Non-radar-bright craters')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_radar_bright_mla, yerr=std_dd_bin_radar_bright_mla,fmt='ko',capsize=5)
ax.plot(middle_bins_lon_mla,mean_dd_bin_not_radar_bright_mla,'b^',label='MLA Radar-bright craters')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_not_radar_bright_mla, yerr=std_dd_bin_not_radar_bright_mla,fmt='b^',capsize=5)

ax.plot([-180,180],[0.2,0.2],':ko')




ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':14})
ax.text(0, 0.202, 'depth=0.2Diameter',size=12)

ax.set_ylim(0.05,0.25)
ax.set_xlim(-40,130)

ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))

plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### median d/D versus longitude -180 to 180 _mla ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,median_dd_bin_radar_bright_mla,'ko',label='Non-radar-bright craters')
ax.errorbar(middle_bins_lon_mla,median_dd_bin_radar_bright_mla, yerr=std_dd_bin_radar_bright_mla,fmt='ko',capsize=5)
ax.plot(middle_bins_lon_mla,median_dd_bin_not_radar_bright_mla,'b^',label='Radar-bright craters')
ax.errorbar(middle_bins_lon_mla,median_dd_bin_not_radar_bright_mla, yerr=std_dd_bin_not_radar_bright_mla,fmt='b^',capsize=5)

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Median depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'mediandD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### count d/D versus longitude -180 to 180 _mla####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,count_dd_bin_radar_bright_mla,'ko',label='Non-radar-bright craters')
ax.plot(middle_bins_lon_mla,count_dd_bin_not_radar_bright_mla,'bo',label='Radar-bright craters')

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Number of craters measured')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'countD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### percentage radar-bright versus longitude -180 to 180 _mla####################################################
fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,((count_dd_bin_radar_bright_mla/(count_dd_bin_not_radar_bright_mla+count_dd_bin_radar_bright_mla))*100),'ko',label='Percentage measured radar-bright')

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('% of measured craters that are radar-bright')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
plt.savefig(export_location+'percentage_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')



###### mean d/D versus binned longitude -180 to 180 _nancy####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,mean_dd_bin_nancy,'ko',label='All craters')
ax.errorbar(middle_bins_lon_nancy,mean_dd_bin_nancy, yerr=std_dd_bin_nancy,fmt='ko',capsize=5)

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### mean d/D versus longitude -180 to 180 _nancy ###################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,mean_dd_bin_radar_bright_nancy,'ro',label='Gridded Non-radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_nancy,mean_dd_bin_radar_bright_nancy, yerr=std_dd_bin_radar_bright_nancy,fmt='ro',capsize=5,alpha=0.5)
ax.plot(middle_bins_lon_nancy,mean_dd_bin_not_radar_bright_nancy,'m^',label='Gridded Radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_nancy,mean_dd_bin_not_radar_bright_nancy, yerr=std_dd_bin_not_radar_bright_nancy,fmt='m^',capsize=5,alpha=0.5)
ax.plot([-180,180],[0.2,0.2],':ko')




ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':14})
ax.text(0, 0.202, 'depth=0.2Diameter',size=12)

ax.set_ylim(0.05,0.25)
ax.set_xlim(-40,130)

ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))

plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### median d/D versus longitude -180 to 180 _nancy ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,median_dd_bin_radar_bright_nancy,'ko',label='Non-radar-bright craters')
ax.errorbar(middle_bins_lon_nancy,median_dd_bin_radar_bright_nancy, yerr=std_dd_bin_radar_bright_nancy,fmt='ko',capsize=5)
ax.plot(middle_bins_lon_nancy,median_dd_bin_not_radar_bright_nancy,'bo',label='Radar-bright craters')
ax.errorbar(middle_bins_lon_nancy,median_dd_bin_not_radar_bright_nancy, yerr=std_dd_bin_not_radar_bright_nancy,fmt='bo',capsize=5)

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Median depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'mediandD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### count d/D versus longitude -180 to 180 _nancy####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,count_dd_bin_radar_bright_nancy,'ko',label='Non-radar-bright craters')
ax.plot(middle_bins_lon_nancy,count_dd_bin_not_radar_bright_nancy,'bo',label='Radar-bright craters')

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Number of craters measured')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'countD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### percentage radar-bright versus longitude -180 to 180 _nancy####################################################
fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_nancy,((count_dd_bin_radar_bright_nancy/(count_dd_bin_not_radar_bright_nancy+count_dd_bin_radar_bright_nancy))*100),'ko',label='Percentage measured radar-bright')

ax.set_xlim(-31,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('% of measured craters that are radar-bright')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
plt.savefig(export_location+'percentage_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### mean d/D versus longitude -180 to 180 _mla _nancy###################################################
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(middle_bins_lon_nancy,mean_dd_bin_radar_bright_nancy,'ro',label='Gridded Non-radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_nancy,mean_dd_bin_radar_bright_nancy, yerr=std_dd_bin_radar_bright_nancy,fmt='ro',capsize=5,alpha=0.5)
ax.plot(middle_bins_lon_nancy,mean_dd_bin_not_radar_bright_nancy,'mo',label='Gridded Radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_nancy,mean_dd_bin_not_radar_bright_nancy, yerr=std_dd_bin_not_radar_bright_nancy,fmt='mo',capsize=5,alpha=0.5)

ax.plot(middle_bins_lon_mla,mean_dd_bin_radar_bright_mla,'ko',label='MLA Non-radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_radar_bright_mla, yerr=std_dd_bin_radar_bright_mla,fmt='ko',capsize=5,alpha=0.5)
ax.plot(middle_bins_lon_mla,mean_dd_bin_not_radar_bright_mla,'bo',label='MLA Radar-bright craters',alpha=0.5)
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_not_radar_bright_mla, yerr=std_dd_bin_not_radar_bright_mla,fmt='bo',capsize=5,alpha=0.5)


ax.plot([-180,180],[0.2,0.2],':ko')




ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':14})
ax.text(0, 0.202, 'depth=0.2Diameter',size=12)

ax.set_ylim(0.05,0.25)
ax.set_xlim(-40,130)

ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_binned_radarbright_v_nonradarbright_mla_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### mean d/D versus longitude -180 to 180 _mla _nancy###################################################


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla[0:9],mean_dd_bin_mla[0:9],'ko',label='MLA track topography')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_mla, yerr=std_dd_bin_mla,fmt='ko',capsize=5)

ax.plot(middle_bins_lon_mla,mean_dd_bin_nancy,'ro',label='Gridded topography')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_nancy, yerr=std_dd_bin_mla,fmt='ro',capsize=5)


ax.plot(middle_bins_lon_mla,mean_dd_bin_rub,'bo',label='Rubanenko et al., 2019')
ax.errorbar(middle_bins_lon_mla,mean_dd_bin_rub, yerr=std_dd_bin_rub,fmt='bo',capsize=5)
ax.set_ylim(0.06,0.2)
ax.set_xlim(-35,121)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'meandD_v_runbinned_longitude_v2_mla_nancy_rub.pdf',format='pdf')
plt.close('all')


################################################################################
###### median d/D versus longitude -180 to 180 _mla _nancy###################################################


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(middle_bins_lon_mla,median_dd_bin_mla,'ko',label='MLA track topography')
ax.errorbar(middle_bins_lon_mla,median_dd_bin_mla, yerr=std_dd_bin_mla,fmt='ko',capsize=5)

ax.plot(middle_bins_lon_mla,median_dd_bin_nancy,'ro',label='Gridded topography')
ax.errorbar(middle_bins_lon_mla,median_dd_bin_nancy, yerr=std_dd_bin_mla,fmt='ro',capsize=5)


ax.plot(middle_bins_lon_mla,median_dd_bin_rub,'bo',label='Rubanenko et al., 2019')
ax.errorbar(middle_bins_lon_mla,median_dd_bin_rub, yerr=std_dd_bin_rub,fmt='bo',capsize=5)

ax.set_xlim(-35,121)
ax.set_ylim(0.06,0.2)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-30, 150, 30)))
ax.set_ylabel('Mean depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'mediandD_v_runbinned_longitude_mla_nancy_rub.pdf',format='pdf')
plt.close('all')
