 #! /Users/susorhc1/anaconda/bin/python
##
##
##
# Program: dD_plots
# Author: Hannah C.M. Susorney
# Date Created: 2020-03-03
#
# Purpose: To make general depth/diameter plots for the whole crater populations
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
index_noradar_bright_mla = np.where(radar_bright_mla!='Yes')

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
index_noradar_bright_nancy = np.where(radar_bright_nancy!='Yes')

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
################################################################################
###### Matplotlib formatting ######################################################
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)
################################################################################
###### d/D of all craters _mla ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(diameter_mla,depth_mla,'ko',label='All craters in study')
ax.errorbar(diameter_mla,depth_mla, yerr=depth_error_mla,xerr=diameter_error_mla,fmt='ko',capsize=5)

ax.set_ylim(0.1,3)
ax.set_xlim(1,40)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_all_mla.pdf',format='pdf')
plt.close('all')
################################################################################
###### d/D by radar bright via not radar-bright _mla ################################
fig = plt.figure()
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)
ax = fig.add_subplot(111)
ax.plot(diameter_mla,depth_mla,'ko',label='Non-radar-bright craters')
ax.errorbar(diameter_mla,depth_mla, yerr=depth_error_mla,xerr=diameter_error_mla,fmt='ko',capsize=5)
ax.plot(diameter_mla[index_radar_bright_mla],depth_mla[index_radar_bright_mla],'bo',label='Radar-bright craters')
ax.errorbar(diameter_mla[index_radar_bright_mla],depth_mla[index_radar_bright_mla], yerr=depth_error_mla[index_radar_bright_mla],xerr=diameter_error_mla[index_radar_bright_mla],fmt='bo',capsize=5)


ax.set_ylim(0.1,3)
ax.set_xlim(1,40)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_radarbright_v_non_radar_bright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus longitude _mla ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_mla,depth_mla/diameter_mla,'ko',label='All craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_longitude_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus longitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_mla,depth_mla/diameter_mla,'ko',label='Non-radar-bright craters')
ax.plot(longitude_mla[index_radar_bright_mla],depth_mla[index_radar_bright_mla]/diameter_mla[index_radar_bright_mla],'bo',label='Radar-bright craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_longitude_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude _mla ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='All craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _mla ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='Non-radar-bright craters')
ax.plot(latitude_mla[index_radar_bright_mla],depth_mla[index_radar_bright_mla]/diameter_mla[index_radar_bright_mla],'bo',label='Radar-bright craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_radarbright_v_nonradarbright_mla.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D of all craters _nancy ######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(diameter_nancy,depth_nancy,'ko',label='All craters in study')
ax.errorbar(diameter_nancy,depth_nancy, yerr=depth_error_nancy,xerr=diameter_error_nancy,fmt='ko',capsize=5)

ax.set_ylim(0.1,3)
ax.set_xlim(1,40)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_all_nancy.pdf',format='pdf')
plt.close('all')
################################################################################
###### d/D by radar bright via not radar-bright _nancy ################################
fig = plt.figure()
tfont = {'family' : 'Times New Roman',
         'size'   : 18}
mpl.rc('font',**tfont)
ax = fig.add_subplot(111)
ax.plot(diameter_nancy,depth_nancy,'ko',label='Non-radar-bright craters')
ax.errorbar(diameter_nancy,depth_nancy, yerr=depth_error_nancy,xerr=diameter_error_nancy,fmt='ko',capsize=5)
ax.plot(diameter_nancy[index_radar_bright_nancy],depth_nancy[index_radar_bright_nancy],'bo',label='Radar-bright craters')
ax.errorbar(diameter_nancy[index_radar_bright_nancy],depth_nancy[index_radar_bright_nancy], yerr=depth_error_nancy[index_radar_bright_nancy],xerr=diameter_error_nancy[index_radar_bright_nancy],fmt='bo',capsize=5)


ax.set_ylim(0.1,3)
ax.set_xlim(1,40)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_radarbright_v_non_radar_bright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus longitude _nancy ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_nancy,depth_nancy/diameter_nancy,'ko',label='All craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_longitude_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus longitude radar-bright v. not radar-bright _nancy ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(longitude_nancy,depth_nancy/diameter_nancy,'ko',label='Non-radar-bright craters')
ax.plot(longitude_nancy[index_radar_bright_nancy],depth_nancy[index_radar_bright_nancy]/diameter_nancy[index_radar_bright_nancy],'bo',label='Radar-bright craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Longitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_longitude_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude _nancy ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy,depth_nancy/diameter_nancy,'ko',label='All craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_nancy.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude radar-bright v. not radar-bright _nancy ###################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(latitude_nancy,depth_nancy/diameter_nancy,'ko',label='Non-radar-bright craters')
ax.plot(latitude_nancy[index_radar_bright_nancy],depth_nancy[index_radar_bright_nancy]/diameter_nancy[index_radar_bright_nancy],'bo',label='Radar-bright craters')

ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_radarbright_v_nonradarbright_nancy.pdf',format='pdf')
plt.close('all')
################################################################################
###### d/D of all craters _mla and _nancy######################################################
diam_line = np.array([0.11,50])
depth_line = 0.2*diam_line

mpl.rcParams['axes.linewidth'] = 1
fig = plt.figure()
ax = fig.add_subplot(111)
index_diam = np.where(diameter_mla < 10)
ax.plot(diameter_mla[index_diam],depth_mla[index_diam],'ko',label='Measured by individual MLA tracks ',alpha=0.7,markeredgewidth=0.0)
ax.errorbar(diameter_mla[index_diam],depth_mla[index_diam], yerr=depth_error_mla[index_diam],xerr=diameter_error_mla[index_diam],fmt='ko',capsize=5,alpha=0.7)
ax.plot(diameter_nancy,depth_nancy,'ro',label='Measured by gridded topography',alpha=0.5,markeredgewidth=0.0)
ax.errorbar(diameter_nancy,depth_nancy, yerr=depth_error_nancy,xerr=diameter_error_nancy,fmt='ro',capsize=5,alpha=0.5)

ax.plot(diam_line,depth_line,':ko')


ax.set_ylim(0.1,3)
ax.set_xlim(1,20)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.tick_params('both', length=10, width=1.5, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.text(1.25, 0.28, 'depth=0.2Diameter',rotation=32,size=10)

ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'dD_all_mla_nancy.pdf',format='pdf')
plt.close('all')


################################################################################
###### d/D of all craters _mla and _nancy_rub ######################################################
diam_line = np.array([0.11,50])
depth_line = 0.2*diam_line

mpl.rcParams['axes.linewidth'] = 1
fig = plt.figure()
ax = fig.add_subplot(111)
index_diam = np.where(diameter_mla < 10)
ax.plot(diameter_rub,depth_rub,'bo',label='Rubanenko et al., 2019',alpha=0.5)
ax.plot(diameter_mla[index_diam],depth_mla[index_diam],'ko',label='Measured by individual MLA Tracks ',alpha=0.7,markeredgewidth=0.0)
ax.errorbar(diameter_mla[index_diam],depth_mla[index_diam], yerr=depth_error_mla[index_diam],xerr=diameter_error_mla[index_diam],fmt='ko',capsize=5,alpha=0.7)
ax.plot(diameter_nancy,depth_nancy,'ro',label='Measured by gridded topography',alpha=0.5,markeredgewidth=0.0)
ax.errorbar(diameter_nancy,depth_nancy, yerr=depth_error_nancy,xerr=diameter_error_nancy,fmt='ro',capsize=5,alpha=0.5)

ax.plot(diam_line,depth_line,':ko')


ax.set_ylim(0.1,3)
ax.set_xlim(1,20)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Diameter (km)')
ax.set_ylabel('Depth (km)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.tick_params('both', length=10, width=1.5, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.text(1.2, 0.45, 'depth=0.2Diameter',rotation=32,size=10)

ax.legend(prop={'size':12})

plt.tight_layout()
plt.savefig(export_location+'dD_all_mla_nancy_rub.pdf',format='pdf')
plt.close('all')
################################################################################
###### d/D versus latitude _mla _rub ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)

m,b = polyfit(latitude_rub, depth_rub/diameter_rub, 1)
lat_for_line = np.array([75,85])

ax.plot(lat_for_line, m*lat_for_line+b, '-r',label='Rubanenko et al., 2019')

#ax.plot(latitude_rub,depth_rub/diameter_rub,'ro',label='Rubanenko et al., 2019', alpha=0.5)
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='All craters')

h,c = polyfit(latitude_mla, depth_mla/diameter_mla, 1)

#ax.plot(lat_for_line, h*lat_for_line+c, '-k')


ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_mla_rub_line.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude _mla _rub ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)

m,b = polyfit(latitude_rub, depth_rub/diameter_rub, 1)
lat_for_line = np.array([75,85])

#ax.plot(lat_for_line, m*lat_for_line+b, '-r',label='Rubanenko et al., 2019')

ax.plot(latitude_rub,depth_rub/diameter_rub,'ro',label='Rubanenko et al., 2019', alpha=0.5)
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='All craters')

h,c = polyfit(latitude_mla, depth_mla/diameter_mla, 1)

#ax.plot(lat_for_line, h*lat_for_line+c, '-k')


ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_mla_rub.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude _mla _rub ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)

lat_for_line = np.array([75,85])


#ax.plot(latitude_rub,depth_rub/diameter_rub,'bo',label='Rubanenko et al., 2019')
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='MLA Topography')
ax.plot(latitude_nancy,depth_nancy/diameter_nancy,'ro',label='Gridded Topography')


m,b = polyfit(latitude_rub, depth_rub/diameter_rub, 1)
p = np.polyfit(latitude_rub, depth_rub/diameter_rub, 1)

#slope, intercept, r_value, p_value, std_err = stats.linregress(latitude_rub, depth_rub/diameter_rub)
#print("R-squared: %f" % r_value**2)
ax.plot(lat_for_line, m*lat_for_line+b, '-b',label='Rubanenko et al., 2019')

h,c = polyfit(latitude_nancy, depth_nancy/diameter_nancy, 1)
ax.plot(lat_for_line, h*lat_for_line+c, '-r')

#h,c = polyfit(latitude_nancy[index_radar_bright_nancy], depth_nancy[index_radar_bright_nancy]/diameter_nancy[index_radar_bright_nancy], 1)
#ax.plot(lat_for_line, h*lat_for_line+c, ':r')

#h,c = polyfit(latitude_nancy[~np.asarray(index_radar_bright_nancy)[0,:]], depth_nancy[~np.asarray(index_radar_bright_nancy)[0,:]]/diameter_nancy[~np.asarray(index_radar_bright_nancy)[0,:]], 1)
#ax.plot(lat_for_line, h*lat_for_line+c, '--r')

#slope, intercept, r_value, p_value, std_err = stats.linregress(latitude_nancy, depth_nancy/diameter_nancy)
#print("R-squared: %f" % r_value**2)


w,d = polyfit(latitude_mla, depth_mla/diameter_mla, 1)
ax.plot(lat_for_line, w*lat_for_line+d, '-k')

#w,d = polyfit(latitude_mla[index_radar_bright_mla], depth_mla[index_radar_bright_mla]/diameter_mla[index_radar_bright_mla], 1)
#ax.plot(lat_for_line, w*lat_for_line+d, ':k')

#w,d = polyfit(latitude_mla[~np.asarray(index_radar_bright_mla)[0,:]], depth_mla[~np.asarray(index_radar_bright_mla)[0,:]]/diameter_mla[~np.asarray(index_radar_bright_mla)[0,:]], 1)
#ax.plot(lat_for_line, w*lat_for_line+d, '--k')



#slope, intercept, r_value, p_value, std_err = stats.linregress(latitude_mla, depth_mla/diameter_mla)
#print("R-squared: %f" % r_value**2)
ax.set_ylim(0.05,0.29)
ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.xaxis.set_major_locator(FixedLocator(np.arange(75, 90, 2)))

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_mla_nancy_rub_line.pdf',format='pdf')
plt.close('all')

################################################################################
###### d/D versus latitude _mla _rub ####################################################
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(latitude_rub,depth_rub/diameter_rub,'bo',label='Rubanenko et al., 2019')
ax.plot(latitude_mla,depth_mla/diameter_mla,'ko',label='MLA Topography')
ax.plot(latitude_nancy,depth_nancy/diameter_nancy,'ro',label='Gridded Topography')




ax.set_ylabel('depth/diameter')
ax.set_xlabel('Latitude (degrees)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax.legend(prop={'size':15})

plt.tight_layout()
plt.savefig(export_location+'dD_v_latitude_mla_nancy_rub.pdf',format='pdf')
plt.close('all')

dD_all = np.hstack([[(depth_nancy/diameter_nancy)],[(depth_mla/diameter_mla)]])
dD_radar_bright = np.hstack([[(depth_nancy[index_radar_bright_nancy]/diameter_nancy[index_radar_bright_nancy])],[(depth_mla[index_radar_bright_mla]/diameter_mla[index_radar_bright_mla])]])

dD_noradar_bright = np.hstack([[(depth_nancy[index_noradar_bright_nancy]/diameter_nancy[index_noradar_bright_nancy])],[(depth_mla[index_noradar_bright_mla]/diameter_mla[index_noradar_bright_mla])]])

print('dD all = '+str(np.mean(dD_all))+'+/-'+str(np.std(dD_all)))

print('dD rb = '+str(np.mean(dD_radar_bright))+'+/-'+str(np.std(dD_radar_bright)))

print('dD no rb = '+str(np.mean(dD_noradar_bright))+'+/-'+str(np.std(dD_noradar_bright)))
