import sparsetools as sp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate
from ipdb import set_trace as stop

var_names = ['Temp','vlos','vturb','Blos','Bhor','Bazi']
stokes_label = ['I','Q','U','V']

# Load model: Q,F,N,P2; S,C,R,L
#diri = '/scratch/rahul/stic/rfdata/'
#m = sp.model(diri+'rf_mod330_190_42_1_2.nc')

fmod = 'atmosout_hirdesh2.nc'
fsyn = 'synthetic_hirdesh2.nc'

m = sp.model(fmod)
print('Original', m.ltau.shape)
nt, nx, ny, ndep = m.ltau.shape

doit = True # do all the calculations
quick = False # for quick tests
plot_option = False
nodes = [10,7,5,3,3,2] # number of nodes in each variable
use_nodes = False
pert_var = [100.,1e5,1e5,100.,100.,0.1]
tasks = [[0,0],[1,0],[2,0],[3,3],[4,1],[5,1]] # [0,0] means temp to StokesI, [4,1] means Bhor to StokesQ
# tasks = [[2,0]] # [0,0] means temp to StokesI, [4,1] means Bhor to StokesQ
wy, wx = 0, 0 # pixel position
mode = 'left' # derivatives: right, left or centered
interpolate_atmos = 'linear' # linear/bezier
reformat = True # model is interpolated to the expected grid

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for kk in range(len(tasks)):
    var_index = tasks[kk][0]
    stokes_index = tasks[kk][1]

    name_rf = 'rf_hirdesh2_{}{}.npy'.format(var_names[var_index],stokes_label[stokes_index])
    pert = pert_var[var_index]

    model_grid = 0.9
    nt, nx, ny,ndep = m.temp.shape
    #nt = 1; nx = 1; ny = 1

    def intpltau(newtau, oldtau, var):
        fX = interpolate.interp1d(oldtau, var)
        return fX(newtau)
        
    def equidist(tau, n):
        if(n == 1): return(0)
        res = np.zeros(n, dtype='int32', order='c')
        mi = tau.min(); ma = tau.max()
        res[0] = 0; res[-1] = tau.size-1
        for ii in range(1, n-1):
            mmin = 1.e10; idx = 0
            xloc = (ma-mi)*ii / (n-1.0) + mi
            for kk in range(1,tau.size-1):
                val = np.abs(tau[kk] - xloc)
                if(mmin > val):
                    mmin = val; idx = kk
            res[ii] = idx
        return(res)
    

    def calculate_location(numnodes,tau_scale):
        if numnodes == 1:
            return [len(tau_scale)//2]
        if numnodes == 2:
            return [0,len(tau_scale)-1]
        if numnodes >= 3:
            return equidist(tau_scale, numnodes)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if doit is True:

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if quick is False:


            ntemp_location = calculate_location(nodes[0],m.ltau[0,0,0,:])
            nvlos_location = calculate_location(nodes[1],m.ltau[0,0,0,:])
            nvturb_location = calculate_location(nodes[2],m.ltau[0,0,0,:])
            nBln_location = calculate_location(nodes[3],m.ltau[0,0,0,:])
            nBho_location = calculate_location(nodes[4],m.ltau[0,0,0,:])
            nazi_location = calculate_location(nodes[5],m.ltau[0,0,0,:])

            newtau = m.ltau[0,0,0,:]
            m2 = sp.model(nx=nx, ny=ny, ndep=len(newtau),nt=1)
            # Fill in variables
            for tt in range(m2.nt):
                for yy in range(m2.ny):
                    for xx in range(m2.nx):
                        m2.ltau[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,:], m.ltau[0,0,0,:])
                        m2.z[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,:], m.z[tt,xx,yy,:])
                        m2.temp[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,ntemp_location], m.temp[tt,xx,yy,ntemp_location])
                        m2.vlos[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,nvlos_location], m.vlos[tt,xx,yy,nvlos_location])
                        m2.vturb[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,nvturb_location], m.vturb[tt,xx,yy,nvturb_location])
                        m2.pgas[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,:], m.pgas[tt,xx,yy,:])
                        m2.nne[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,:], m.nne[tt,xx,yy,:])
                        m2.Bln[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,nBln_location], m.Bln[tt,xx,yy,nBln_location])
                        m2.Bho[tt,yy,xx,:] = intpltau(newtau, m.ltau[0,0,0,nBho_location], m.Bho[tt,xx,yy,nBho_location])
                        m2.azi[tt,yy,xx,:] = np.interp(newtau, m.ltau[0,0,0,nazi_location], m.azi[tt,xx,yy,nazi_location])

            # print(m2.ltau[0,0,0,:])

            '''m2.ltau[:] = m.ltau[:]
            m2.z[:] = m.z[:]
            m2.temp[:] = m.temp[:]
            m2.vlos[:] = m.vlos[:]
            m2.vturb[:] = m.vturb[:]
            m2.pgas[:] = m.pgas[:]
            m2.nne[:] = m.nne[:]
            m2.Bln[:] = m.Bln[:]
            m2.Bho[:] = m.Bho[:]
            m2.azi[:] = m.azi[:]'''
            m2.write('modelin_2.nc')
            m2.write('modelin_X.nc')
            dep = len(m.ltau[0,0,0,:])
            tauu = m.ltau[0,0,0,:]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Original profile from the unperturbed model:
        os.system('/scratch/rahul/new/stic/src/STiC.x > log_stic.txt')
        original = sp.profile('syntheticX.nc')
        original = original.dat[:,:,:, :, stokes_index]
        os.system('cp syntheticX.nc synthetic_cycle0.nc')
        profile_array = []

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if use_nodes is False: # Perturbing each optical depth point
            for ii in range(dep):
                mX = sp.model('modelin_2.nc') # Adding the perturbation to the original model
                var = [mX.temp, mX.vlos, mX.vturb, mX.Bln, mX.Bho, mX.azi]
                tauX = mX.ltau[0,0,0,:]

                var[var_index][:,:,:,ii] = var[var_index][:,:,:,ii] + pert
                mX.write('modelin_X.nc')
                os.system('/scratch/rahul/new/stic/src/STiC.x > log_stic.txt')
                p = sp.profile('syntheticX.nc')
                profile_array.append(p.dat[:,:,:, :, stokes_index]) # t,x,y,lam,ss


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if use_nodes is True: # Perturbing node location



            numnodes = nodes[var_index]
            node_location = calculate_location(numnodes,tauu)
            print(node_location)
            for inode in range(numnodes):
                inode = node_location[inode]
                mX = sp.model('modelin_2.nc') # Adding the perturbation to the original model
                var = [mX.temp, mX.vlos, mX.vturb, mX.Bln, mX.Bho, mX.azi]
                tauX = mX.ltau[0,0,0,:]

                var[var_index][:,:,:,inode] = var[var_index][:,:,:,inode] + pert
                for tt in range(m2.nt):
                    for yy in range(m2.ny):
                        for xx in range(m2.nx):                
                            if interpolate_atmos == 'linear':
                                var[var_index][tt,yy,xx,:] = np.interp(tauu, tauX[node_location], var[var_index][tt,yy,xx,node_location])
                if interpolate_atmos == 'bezier':
                    import mathtools as mt
                    var[var_index][:,:,:,:] = mt.bezier3(tauX[node_location], var[var_index][:,:,:,node_location], tauu)
               
                mX.write('modelin_X.nc')
                # print(mX.vlos)

                os.system('/scratch/rahul/new/stic/src/STiC.x> log_stic.txt')
                p = sp.profile('syntheticX.nc')
                profile_array.append(p.dat[:, :, :, :, stokes_index]) # t,x,y,lam,ss

            tauu = tauu[node_location]



        profile_array = np.array(profile_array)
        rf = (profile_array-original)/pert

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if rf.shape[0] < dep:
        #     mrf = np.zeros((dep,rf.shape[1]),dtype=np.float32)
        #     for ii in range(rf.shape[1]):
        #         mrf[:,ii] = np.interp(tauu, tauX[node_location], rf[:,ii])
        #     rf = mrf[:]

        np.save(name_rf,[rf,tauu,p.nw,original])
        rf, mltau, pnw, orig = rf,tauu,p.nw,original

    if doit is False:
        rf, mltau, pnw, orig = np.load(name_rf)
        pass
        # Checking the file
        


