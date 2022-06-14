"""
STIC inversion example

In this example, we prepare a dataset consisting of Ca II K, Ca II 8543 and Fe I 6301/6302
that was recorded with CRISP and CHROMIS.

The assumption is that the data has been calibrated to absolute CGS intensity units.
All dependencies should be available in stic/pythontools/py2/

Modifications:
                  2017-03-27, JdlCR: Created!

"""
import numpy as np
import matplotlib.pyplot as pl
import sparsetools as sp
import mfits as mf
#import prepare_data as S
import satlas
import crisp as fpi
import chromis as cr
from netCDF4 import Dataset as nc
import pandas as pd
import imtools as imt
import matplotlib.patches as patches
import glob
import scipy.io as io


#-----------------------------------------------------------------------------------------
# SOME DEFINITIONS
#-----------------------------------------------------------------------------------------

def findgrid(w, dw, extra=8):
    w1=np.round(w*1000).astype('int32')
    dw1 = int(dw*1000)

    w2 = w/dw
    w2 = np.round(w2)
    
    idx = np.zeros(w.size, dtype='int32')

    np0 = (w2[-1] - w2[0])+1 + extra
    wn = (np.arange(np0, dtype='float64') - np0//2)*dw  #-extra//2*dw

    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(wn-w[ii]))
    
    return wn, idx

#-----------------------------------------------------------------------------------------

def getCont(lam):
    s = satlas.satlas()
    x,y,c = s.getatlas(lam-0.1,lam+0.1, cgs=True)
    return np.median(c)

#-----------------------------------------------------------------------------------------

def writeInstProf(oname, var, pref=None):
    ncfile1 = nc(oname,'w', format='NETCDF4')
    ncfile1.createDimension('wav',var.size)
    par1 = ncfile1.createVariable('iprof','f8',('wav'))
    par1[:] = var


    if(len(pref) == 3):
        ncfile1.createDimension('np',len(pref))
        par2 = ncfile1.createVariable('pref','f8',('np'))
        par2[:] = np.float32(pref)

    ncfile1.close()
    
#-----------------------------------------------------------------------------------------

def calibrate_data(wav, d, filename, cw = 0.0):
    cc = mf.readfits(filename)
    d /= cc[0]
    wav -= cc[1] - cw
    
    return 1


#
# MAIN PROGRAM
#
if __name__ == "__main__":
	#----select pixels---
	x0,y0 = 300,300
	dx, dy = 1, 1
	#-------------------

	print('')
	print('creating files for pixels, x0, y0:',x0,y0)
	print('pixel range dx, dy:',dx,dy)
	print('')

	#read data ([wav, StokesI, StokesQ, StokesU, StokesV])
	# read time for each frame

	#dat = io.readsav('/scratch/rahul/stic/mast/IMCUBE_LEV2.sav')
	
	#offset in data counts [CGS units] and wavelength offset
	'''
	factor = [ 9.01951863e-10, -5.36763645e-02]

	ca_int = dat['imagecube']*factor[0]
	wc8 = dat['wl']+factor[1]
	lam, px,py = ca_int.shape
	'''
	
	factor = [ 7.18381822e-10, 5.989988e-02]

	dd = io.readsav('Ca_8542_A_line_profile_deltalambda.sav')
	data = dd['ica']
	wave = dd['dellam']+8542.09

	ca_int = data*factor[0]
	wc8 = wave+factor[1]

	
	lam = len(ca_int)
	px,py=1,1


	# Get the continuum level for all lines using the FTS atlas
	cw = 8542
	cont = getCont(cw)

	ca8 = np.zeros((lam,px,py,4), dtype=np.float64)
	ca8[:,0,0,0] = ca_int/cont
	
	#insert random noise in Stokes Q, U, and V
	#for i in range(px):
	#	for j in range(py):
	#		for k in range(1,4): ca8[:,i,j,k] = np.random.normal(0,0.0005,lam)
	
	ca8 = np.transpose(ca8,[1,2,0,3])


	# These observations are not acquired in a regular wavelength grid
	# Find a finer grid that contains all wavelength points for each line
	# This grid should be at least the FWHM of the instrumental profile


	# 6301 and 6302, each in a different region. The CRISP FWHM is
	# ~30 mA at 6302/
	# We try to find a grid of ~10 mA, that should be able to fit all points
	# The original grid is multiple of ~40 mA. wfe1 is the new grid and ife1
	# is the location of the observed points in the new grid

	#wfe1, ife = findgrid(wfe, 0.005, extra=0)

	# 8542, the observations where performed in a grid multiple of ~85 mA.
	# The FWHM of CRISP at 8542 is ~55 mA so 1/2 of the original grid should do.

	#wc8, ic8 = findgrid(wca[:], 0.025, extra=8)

	# Ca II K, the observations are recorded in a grid of 3 km/s + 2 external points
	# These profiles are in theory critically sampled, but we need to add extra points
	# for the outer points and for the continuum (last point in the array).

	#wckf, ick = findgrid(wck[0:len(wck)-1], (wck[10]-wck[9]), extra=8)


	#
	# Now we create a container for each spectral region in STIC format
	# We will add the fine grid, but all points that were not observed will
	# be given weight zero, so they don't contribute to the inversion.
	# 

	nf = 1
	ca_8 = sp.profile(nt = nf, nx=dy, ny=dx, ns=4, nw=wc8.size)

	ca_8.wav[:] = wc8[:]	#insert observed wavelength

	#mean_i = np.mean(ca8[0:400,0:400,:,0],axis=(0,1))

	#insert observed profiles
	#ca_8.dat[0,:,:,:,:] = ca8[x0:x0+dx,y0:y0+dy, :,:]
	ca_8.dat[0,:,:,:,:] = ca8[0,0, :,:]

	# weights 
	ca_8.weights[:,:] = 1.e16	# Very high value means weight zero
	ca_8.weights[:,0] = 0.001
	#ca_8.weights[:,1] = 0.01	
	#ca_8.weights[:,2] = 0.01
	#ca_8.weights[:,3] = 0.01	


	sp_all = ca_8

	#obsfile = str('observed_' + str(x0) + '_' + str(y0) + '_' + str(dx) +'_'+ str(dy))+'.nc'
	obsfile = 'observed_hirdesh.nc'

	#write the observed profiles 
	sp_all.write(obsfile)

	#
	# Now print the regions for the config file
	#
	lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
	print(" ")
	print("Regions information for the input file:" )
	#print(lab.format(ca_k.wav[0], ca_k.wav[1]-ca_k.wav[0], ca_k.wav.size-1, cont[0], 'fpi, 3934.nc'))
	#print(lab.format(4000., ca_k.wav[1]-ca_k.wav[0], 1, cont[0], 'none, none'))
	print(lab.format(ca_8.wav[0], ca_8.wav[1]-ca_8.wav[0], ca_8.wav.size, cont, 'none, none'))
	#print(lab.format(fe_1.wav[0], fe_1.wav[1]-fe_1.wav[0], fe_1.wav.size, cont[2], 'fpi, 6173.nc'))
	#print(lab.format(wfe2[0], wfe2[1]-wfe2[0], wfe2.size, cont[2], 'fpi, 6302.nc'))
	print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
	print(" ")

	#
	# Generate Instrumental profiles and save them
	#

	# Ca II 8542
	dw =  ca_8.wav[1]-ca_8.wav[0]
	ntw= 41
	f=fpi.crisp(8542.0)
	tw = (np.arange(ntw)-ntw//2)*dw 
	tr = f.dual_fpi(tw, erh = -0.025)
	tr /= tr.sum()
	writeInstProf('../mast/8542.nc', tr,  [8542.091, 9.0, 2.0])
	
	#
    # Init the input model, all quantities in CGS units!
    #

    # First create a tau scale
    
	taumin = -7.8
	taumax= 1.0
	dtau = 0.14
	ntau = int((taumax-taumin)/dtau) + 1
	tau = np.arange(ntau, dtype='float64')/(ntau-1.0) * (taumax-taumin) + taumin

	# Now create a smooth temperature profile
	temp = np.interp(tau, np.asarray([-8.0, -6.0, -4.0, -2.0 , 0.8]), np.asarray([70000., 8000., 4000., 4800., 7000.]))
	# Fill in the model
	m = sp.model(nx=dx, ny=dy, nt=nf, ndep=ntau)
	m.ltau[:,0:dy,0:dx,:] = tau
	m.temp[:,0:dy,0:dx,:] = temp

	 # The inversion only needs to know the gas pressure at the upper boundary. FALC has Pgas[top] ~ 0.3, but
	 # this value is for quiet-Sun. Active regions can have up to Pgas[top] = 10.
	 
	m.pgas[:,0:dy,0:dx,:] = 1.0    

	# Fill in initial B field and velovity (optional)
	m.vturb[:,0:dy,0:dx,:] = 1.e5
	m.vlos[:,0:dy,0:dx,:] = 0.5e5 # cm/s
	m.Bln[:,0:dy,0:dx,:] = 0.
	m.Bho[:,0:dy,0:dx,:] = 0. 
	m.azi[:,0:dy,0:dx,:] = 80. * 3.14159 / 180.

	# Write to HD
	#modfile = str('inmod_' + str(x0) + '_' + str(y0) + '_' + str(dx) +'_'+ str(dy))+'.nc'
	modfile = 'inmod_hirdesh.nc'

	m.write(modfile)

