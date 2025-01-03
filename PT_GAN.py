import numpy as np
from subprocess import Popen, DEVNULL, STDOUT, check_call
import scipy.signal as signal
import shutil
from sem2d_read_fault import sem2d_read_fault
from scipy.interpolate import griddata
from mpi4py import MPI
from PTMCMCSampler import PTMCMCSampler
import joblib
import keras

# pip install git+https://github.com/JanPremus/PTMCMCSampler.git@master

class SeismoLikelihood(object):
    
    def lnlikefn(self,ParInv):
        
        #Run the forward calculation
        global par
      
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()
        ichain = rank
        if GANdim > 1:
            parGAN = np.reshape(ParInv[0:GANdim], [1,GANdim])
            dynGAN = GANscaler.inverse_transform(GANmodel.predict(parGAN).reshape(1,-1))
            tt=np.reshape(ParInv[GANdim:],[Nfile, Npar-np.abs(GANBorder[0]-GANBorder[1])])  #operate in folder given by the number of process
            par[:,0:np.abs(ParBorder[0]-GANBorder[0])]=tt[:,0:np.abs(ParBorder[0]-GANBorder[0])]
            par[:,np.abs(ParBorder[0]-GANBorder[0]):np.abs(ParBorder[0]-GANBorder[1])]=np.reshape(dynGAN, [Nfile, np.abs(GANBorder[0]-GANBorder[1])])
            par[:,np.abs(ParBorder[0]-GANBorder[1]):np.abs(ParBorder[0]-ParBorder[1])]=tt[:,np.abs(ParBorder[0]-GANBorder[0]):]
        else:
            par=np.reshape(ParInv,[Nfile, Npar])
        #Discard if model out of bounds 
        misfit = 0
        misfit_fault = 0.
        misfit_seis=0.
        misfit_GPS=0.
        
        #Copy new parameters into the files
        for i in range(Nfile):

            FileID = open('%s/%i/%s'%(name, ichain, FileName[i]), 'w')
            #prestress is set as a fraction of 1e5 
                
            if i==0: #Stress drop is set as fraction of 1e6
                parf[i,ParBorder[0]:ParBorder[1],1]= (par[0,:]+par[1,:])*1.e6/Sn + mu_d            
            if i==1:  #strenght excess is set as a fraction of 1e6  
                parf[i,ParBorder[0]:ParBorder[1],1] = -(par[0,:]*1.e6+Sn*mu_d)
            if i==2: #Fracture energy is set as fraction of 1e6
                parf[i,ParBorder[0]:ParBorder[1],1]= 2*par[2,:]*1.e6/(Sn*parf[0,ParBorder[0]:ParBorder[1],1]-Sn*mu_d)
                
            np.savetxt(FileID, parf[i,:,:])
            FileID.close()
            
        #Run the code in ichain folder
        args='./sem2dpack.out'
        process=Popen(args, cwd='%s/%i' %(name,ichain), shell=False, stdout=DEVNULL)
        process.wait()    
        result_path = '%s/%i/' %(name,ichain)
        fault_name  = "Flt05"
        data = sem2d_read_fault(result_path,fault_name)
        
        CreateMtilde (name, ichain, data, seis_depth)
        
        #Calculate synthetic data and their misfit
        sxDI = 0.
        szDI = 0.
        misfit_seisI = 0.
        misfitF_seisI = 0.
        norm_seisI = 0.
        
        if NseisE!=0:
            sxDE, szDE, syDE = seistime3(NseisE, sxDEt,szDEt,syDEt,dtDE,tminE,TseisE,dtE,tming)
            misfit_seisE, misfitF_seisE, sxE, szE, syE = evalseisE(name,ichain,NseisE,sxDE,szDE,syDE,dtE,sigmaE,TseisE,seisE_shard,WindowE)
        else:
            sxDE= 0. 
            szDE= 0.
            syDE= 0.
            misfit_seisE = 0.
            misfitF_seisE = 0.
            norm_seisE = 0.
        if NGPS!=0:
            misfit_GPS, misfitF_GPS = evalGPS(name, ichain, NGPS, gxD, gzD, gyD, sigmaG,GPS_shard)
        else:
            misfit_GPS=0.
            misfitF_GPS=0.
            norm_GPS=0.
        
        misfit_fault = evalfault(name,ichain,data,seis_depth)
        misfit =-np.min(misfit_seisI)-np.min(misfit_seisE)-misfitF_GPS+misfit_fault
        print(-np.min(misfitF_seisE),-np.min(misfit_seisE),-misfit_GPS, misfit_fault)
            
        return misfit, 1.
    
    def lnpriorfn(self, ParInv):#, Pmin, Pmax, Sn, Nfile, Npar):
        global par
        if GANdim > 1:
            parGAN = np.reshape(ParInv[0:GANdim], [1,GANdim])
            dynGAN =GANscaler.inverse_transform( GANmodel.predict(parGAN).reshape(1,-1))            
            tt=np.reshape(ParInv[GANdim:],[Nfile, Npar-np.abs(GANBorder[0]-GANBorder[1])])  
            par[:,0:np.abs(ParBorder[0]-GANBorder[0])]=tt[:,0:np.abs(ParBorder[0]-GANBorder[0])]
            par[:,np.abs(ParBorder[0]-GANBorder[0]):np.abs(ParBorder[0]-GANBorder[1])]=np.reshape(dynGAN, [Nfile, np.abs(GANBorder[0]-GANBorder[1])])
            par[:,np.abs(ParBorder[0]-GANBorder[1]):np.abs(ParBorder[0]-ParBorder[1])]=tt[:,np.abs(ParBorder[0]-GANBorder[0]):]
        else:
            par=np.reshape(ParInv,[Nfile, Npar])
        lp = 0
        for i in range(Nfile):
            for j in range(Npar):
                if par[i,j]<Pmin[i] or par[i,j]>Pmax[i]:
                    lp = -np.inf
                    print('breakPar', par[i,j], Pmin[i], Pmax[i])
                    break      
        
        for i in range(GANdim):
            if parGAN[0,i]<0. or parGAN[0,i]>1.:
                lp = -np.inf
                print('breakGAN', parGAN[0,i], 0., 1.)
                break         
        
        #Discard if model nucleates outside of nucleation area - muS*Sn<Tt
        #for j in range(Npar):   
        #   # print((-par[1,j],par[2,j]*Sn))
        #    if par[1,j]<0.:
        #        lp = -np.inf
        #        print('breakN', par[1,j], par[0,j])
        #        break  
            
        return lp

def evalmisfit3(Nseis, Nt, sx, sz, sy, sxD, szD, syD, sigma,dt,time,shard):
    Nshift=5
    maxtshift=2
    Ntshift=np.int32(maxtshift/dt)
    dtshiftN=5
    dtshift=np.int32(Ntshift/dtshiftN)
    misfitT=np.zeros([2*dtshiftN+1])
    #normT=np.zeros([2*dtshiftN+1])
    start = np.zeros(Nseis)
    for i in range (Nseis):
        if np.max(sx[i,:])>0.01:
            xstart=np.argwhere(np.abs(sx[i,:])>0.01)[0]-12
        else:
            xstart=1
        if np.max(sz[i,:])>0.01:
            zstart=np.argwhere(np.abs(sz[i,:])>0.01)[0]-12
        else:
            zstart=1
        if np.max(sy[i,:])>0.01:
            ystart=np.argwhere(np.abs(sy[i,:])>0.01)[0]-12
        else:
            ystart=1
            
        start[i] = np.minimum(xstart,zstart)
        start[i] = np.minimum(start[i],ystart)
        start[i] = np.minimum(start[i], sxD.shape[1]-Nt[i])
        if start[i]<0:
            start[i]=0  
        if start[i]+Nt[i]>len(sx[1,:]):
            start[i]=len(sx[1,:])-Nt[i]     
    
    start = (np.ceil((start))).astype(int)
    #Evaluate misfit with basic timeshift (timshiftB)
    for i in range (Nseis):

        l2x = np.sum(np.square(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        l2z = np.sum(np.square(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        l2y = np.sum(np.square(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        
        misfitT[dtshiftN] = misfitT[dtshiftN]+shard[i]*(l2x+l2z+l2y)/sigma[i]**2
        #normT[dtshiftN] = normT[dtshiftN]+(np.sum(np.square(sxD[i,start[i]+Nshift:start[i]+Nshift+Nt]))+np.sum(np.square(szD[i,start[i]+Nshift:start[i]+Nshift+Nt]))+np.sum(np.square(syD[i,start[i]+Nshift:start[i]+Nshift+Nt])))/sigma[i]**2

    #Evaluate misfit with other timeshifts (timshiftB+-maxtshift)
    for j in range (dtshiftN):
        #print(l2x+l2z)
        for i in range (Nseis):
            l2x = np.sum(np.square(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            l2z = np.sum(np.square(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            l2y = np.sum(np.square(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            misfitT[dtshiftN+1+j] = misfitT[dtshiftN+1+j]+shard[i]*(l2x+l2z+l2y)/sigma[i]**2
            #normT[dtshiftN+1+j] = normT[dtshiftN+1+j]+ (np.sum(np.square(sxD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift])) + np.sum(np.square(szD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift])) + np.sum(np.square(syD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift]))) /sigma[i]**2
            
            l2x = np.sum(np.square(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            l2z = np.sum(np.square(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            l2y = np.sum(np.square(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            
            misfitT[dtshiftN-1-j] = misfitT[dtshiftN-1-j]+shard[i]*(l2x+l2z+l2y)/sigma[i]**2
            #normT[dtshiftN-1-j] = normT[dtshiftN-1-j]+ (np.sum(np.square(sxD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])) + np.sum(np.square(szD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])) + np.sum(np.square(syD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])))/sigma[i]**2

    return misfitT


def evalmisfit3L1(Nseis, Nt, sx, sz, sy, sxD, szD, syD, sigma,dt,time,shard):
    Nshift=5
    maxtshift=2
    Ntshift=np.int32(maxtshift/dt)
    dtshiftN=5
    dtshift=np.int32(Ntshift/dtshiftN)
    misfitT=np.zeros([2*dtshiftN+1])
    #normT=np.zeros([2*dtshiftN+1])
    start = np.zeros(Nseis)
    for i in range (Nseis):
        if np.max(sx[i,:])>0.01:
            xstart=np.argwhere(np.abs(sx[i,:])>0.01)[0]-12
        else:
            xstart=1
        if np.max(sz[i,:])>0.01:
            zstart=np.argwhere(np.abs(sz[i,:])>0.01)[0]-12
        else:
            zstart=1
        if np.max(sy[i,:])>0.01:
            ystart=np.argwhere(np.abs(sy[i,:])>0.01)[0]-12
        else:
            ystart=1
            
        start[i] = np.minimum(xstart,zstart)
        start[i] = np.minimum(start[i],ystart)
        start[i] = np.minimum(start[i], sxD.shape[1]-Nt[i])
        if start[i]<0:
            start[i]=0  
        if start[i]+Nt[i]>len(sx[1,:]):
            start[i]=len(sx[1,:])-Nt[i]     
    
    start = (np.ceil((start))).astype(int)
    #Evaluate misfit with basic timeshift (timshiftB)
    for i in range (Nseis):

        l2x = np.sum(np.abs(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        l2z = np.sum(np.abs(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        l2y = np.sum(np.abs(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift:start[i]+Nshift+Nt[i]]))
        
        misfitT[dtshiftN] = misfitT[dtshiftN]+shard[i]*(l2x+l2z+l2y)/sigma[i]
        #normT[dtshiftN] = normT[dtshiftN]+(np.sum(np.square(sxD[i,start[i]+Nshift:start[i]+Nshift+Nt]))+np.sum(np.square(szD[i,start[i]+Nshift:start[i]+Nshift+Nt]))+np.sum(np.square(syD[i,start[i]+Nshift:start[i]+Nshift+Nt])))/sigma[i]**2

    #Evaluate misfit with other timeshifts (timshiftB+-maxtshift)
    for j in range (dtshiftN):
        #print(l2x+l2z)
        for i in range (Nseis):
            l2x = np.sum(np.abs(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            l2z = np.sum(np.abs(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            l2y = np.sum(np.abs(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt[i]+(j+1)*dtshift]))
            misfitT[dtshiftN+1+j] = misfitT[dtshiftN+1+j]+shard[i]*(l2x+l2z+l2y)/sigma[i]
            #normT[dtshiftN+1+j] = normT[dtshiftN+1+j]+ (np.sum(np.square(sxD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift])) + np.sum(np.square(szD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift])) + np.sum(np.square(syD[i,start[i]+Nshift+(j+1)*dtshift:start[i]+Nshift+Nt+(j+1)*dtshift]))) /sigma[i]**2
            
            l2x = np.sum(np.abs(sx[i,start[i]+0:start[i]+Nt[i]]-sxD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            l2z = np.sum(np.abs(sz[i,start[i]+0:start[i]+Nt[i]]-szD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            l2y = np.sum(np.abs(sy[i,start[i]+0:start[i]+Nt[i]]-syD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt[i]-(j+1)*dtshift]))
            
            misfitT[dtshiftN-1-j] = misfitT[dtshiftN-1-j]+shard[i]*(l2x+l2z+l2y)/sigma[i]
            #normT[dtshiftN-1-j] = normT[dtshiftN-1-j]+ (np.sum(np.square(sxD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])) + np.sum(np.square(szD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])) + np.sum(np.square(syD[i,start[i]+Nshift-(j+1)*dtshift:start[i]+Nshift+Nt-(j+1)*dtshift])))/sigma[i]**2

    return misfitT


def CreateMtilde (name, ichain, data, seis_depth):

    grid_size = 1000.0  #Horizontal spatial discretization 
    div_factor = 100   #Time discretization in Sem2dpack steps
    nt = data['nt']
    nodes_x = data['x']
    v       = data['v']        
    X_lower  = np.min(nodes_x)
    X_upper  = np.max(nodes_x)
    X_dim    = int((X_upper-X_lower)/grid_size+1)
    grid_x = np.mgrid[X_lower:X_upper:X_dim*1j]
    sr1 = np.zeros([X_dim, nt-2])
    Nstep = np.int32((nt)/div_factor)
 
    for i in range(Nstep):
        sr1[:,i]= griddata(nodes_x,  (np.sum(v[div_factor*(i-1):div_factor*i,:],0)/div_factor), grid_x, method='linear')
    
    Ndepth = np.int32(seis_depth/grid_size)
    depths = np.arange(Ndepth)*0.5*np.pi/(Ndepth)
    depth_cos = np.cos(depths)
    depth_cos = depth_cos
    sr2 = np.zeros([Ndepth,X_dim,Nstep])
    FileID = open('%s/%i/%s'%(name, ichain, 'mtildeX.dat'), 'w')
    for i in range(Nstep):
        for j in range(X_dim):    
            sr2[:,j,i]=sr1[j,i]*depth_cos
    
    for k in range(Ndepth):
        for j in range(X_dim):
            np.savetxt(FileID, sr2[k,j,:])
    FileID.close()     
    

def evalseisE(name,ichain, NseisE, sxD, szD, syD, dtE, sigmaE, Tseis, shard, Window):

    #time vector for external seismograms
    Nt = np.int32(Window/dtE)
    timeE=np.arange(np.int32(Tseis/dtE))*dtE
    
    args='./fd3d_seisall'
    process=Popen(args, cwd='%s/%i' %(name,ichain), shell=False, stdout=DEVNULL)
    process.wait()  
    
    Fil = open('%s/%i/svseisErik.dat' %(name,ichain), 'r')
    temp = np.loadtxt(Fil)
    sxE = np.transpose(temp[:,1:NseisE+1])
    Fil.close()
    Fil = open('%s/%i/svseisNrik.dat' %(name,ichain), 'r')
    temp =  np.loadtxt(Fil)
    szE = np.transpose(temp[:,1:NseisE+1])
    Fil.close() 
    Fil = open('%s/%i/svseisZrik.dat' %(name,ichain), 'r')
    temp =  np.loadtxt(Fil)
    syE = np.transpose(temp[:,1:NseisE+1])
    Fil.close()
    
    misfitT = evalmisfit3(NseisE, Nt, sxE, szE, syE, sxD, szD, syD, sigmaE,dtE, timeE, shard)
    shardO=np.ones(len(shard))
    misfitL1 = evalmisfit3L1(NseisE, Nt, sxE, szE, syE, sxD, szD, syD, sigmaE,dtE, timeE, shardO)

    return misfitT, misfitL1, sxE, szE, syE

def evalGPS(name, ichain, NGPS, gxD, gzD, gyD, sigmaG, shard):
    
    misfitT=0.
    misfitF=0.
    normT=0.
    #calculate the gps displacements
    args='./fd3d_gpsall'
    process=Popen(args, cwd='%s/%i' %(name,ichain), shell=False, stdout=DEVNULL)
    process.wait()      
    
    Fil = open('%s/%i/GPSdist.dat' %(name,ichain), 'r')
    temp = np.loadtxt(Fil)
    gx=temp[:,0]
    gz=temp[:,1] 
    gy=temp[:,2]
    Fil.close()
    
    #calculate misfit
    for i in range(NGPS):
        lx=np.abs(gx[i]-gxD[i])
        lz=np.abs(gz[i]-gzD[i])
        ly=np.abs(gy[i]-gyD[i])
        
        misfitT = misfitT + shard[i]*(lx + lz + ly)/sigmaG[i]
        misfitF = misfitF + (lx**2 + lz**2 + ly**2)/sigmaG[i]**2
        #normT = normT + (gxD[i]**2 + gzD[i]**2 + gyD[i]**2)/sigmaG[i]
    
    return misfitT, misfitF

def evalfault(name, ichain, data, seis_depth):
     
    M_exp = 6.5 #expected minimum magnitude
    mom_exp = 10**(1.5*(M_exp+6.07))
    misfit = 0.
    nx = data['nx']
    nt = data['nt']
    dt = data['dt']
    nodes_x = data['x']
    d       = data['d']
    Slip     = np.zeros((nx))
    for x in range(nx):
        Slip[x] = d[-1,x]
    grid_size = 100.0
    
    X_lower  = np.min(nodes_x)
    X_upper  = np.max(nodes_x)
    X_dim    = int((X_upper-X_lower)/grid_size+1)
    grid_x = np.mgrid[X_lower:X_upper:X_dim*1j]
    grid_Slip= griddata(nodes_x,  Slip, grid_x, method='linear')

    Slip_sum = np.absolute(np.sum(grid_Slip))*grid_size*seis_depth
     
    Fil = open('%s/%i/Cs_sem2d.tab' %(name,ichain), 'r')
    vs=np.loadtxt(Fil)  
    Fil.close()
    Fil = open('%s/%i/Rho_sem2d.tab' %(name,ichain), 'r')
    rho=np.loadtxt(Fil)  
    Fil.close()
    mu = vs[1,1]*vs[1,1]*rho[1,1]
    mom = mu*Slip_sum
    misfit=-0.5*(100./(0.1+(mom/mom_exp)**2))
         
    return misfit          

def seistime3(Nseis,sxDt,szDt,syDt,dtD,tmin, Tseis, dt, tming):
     
    #time grid for synthetic data
    time=np.arange(np.int32(Tseis/dt))*dt
    
    #interpolate real data to synthetic time grid
    sxD = np.empty([Nseis, len(time)])
    szD = np.empty([Nseis, len(time)])
    syD = np.empty([Nseis, len(time)])
    for i in range (Nseis):
        timeD = np.arange(len(sxDt[i,:]))*dtD[i]+tmin[i]-tming
        sxD[i,:] = np.interp(time, timeD, sxDt[i,:])
        szD[i,:] = np.interp(time, timeD, szDt[i,:])   
        syD[i,:] = np.interp(time, timeD, syDt[i,:])   
    
    return sxD, szD, syD

class LogNormJump(object):
    
    def __init__(self, step):
        """Make log normal steps in parameters"""
        self.step = step
        
    def jump(self, x, it, beta):
        """ 
        Function prototype must read in parameter vector x,
        sampler iteration number it, and inverse temperature beta
        """
        
        # log of forward-backward jump probability
        lqxy = 0
        
        # uniformly drawn numbers
        #temp = np.random.normal(loc=0.,scale=1., size=len(x))
        
        q = np.zeros(ndim)
        #Make steps
        for i in range (ndim):
            #print(i, self.step[i])
            q[i] = x[i]*np.random.lognormal(mean=0.,sigma=self.step[i])
            
            
            #np.exp(temp[i]*self.step[i])
            
            lqxy = lqxy+(np.log(q[i])-np.log(x[i]))*beta
        
        return q, lqxy



#=======================================================================================================
#                                           MAIN CODE
#=======================================================================================================
  
#General information
Nstep = 25000 #Number of inversion steps
NseisE=12  #Number of stations calculated through AXITRA (fd3d_seisall)
NGPS = 12   #Number of GPS stations

seisI_shard=[]  #Internal seismograms included in the data shard. 1.. Included, 0.. Not included
seisE_shard=np.ones(NseisE) #External seismograms included in the data shard.  1.. Included, 0.. Not included
GPS_shard=np.ones(NGPS) #GPSs included in the data shard.  1.. Included, 0.. Not included

#Seismogram information
dtE=100./256.         #time sampling of AXITRA seismograms
TseisI = 0.     #Time length of sem2dpack seismograms
TseisE = 100.   #Time length of AXITRA seismograms
tming=0.        #beginning time of the real seismograms   1.675646227112452030e+09+50 for the Turkey earthquake
seisE_input = 1.  #0... binary file, 1.. text file from axitra

#Dynamic parameters information
FileName = [ 'MuS.tab', 'TtS.tab','DcS.tab']    #Names of the Sem2dpack dynamic parameter files
Nfile = 3   #Number of dynamic parameters/files
Npar = 9    #Number of independent discrete parameters (total number of inverted parameters is Npar*Nfile)
NparTot = 13 #Total number of parameters in the files
GANdim = 3  #Number of latent dimensions
GANstep=0.01

GANBorder = [4, 9]
ParBorder =  [2, 11]

Sn = 60e6 #Normal stress
mu_d = 0.4
seis_depth = 1.e4 #Seismogenic depth

Pmin= [0.5, 0.5, 0.5]   #Minimum values of dyn. parameters #prestress is set as a fraction of normal stress  
Pmax= [30., 30., 30.]   #Maximum values of dyn. parameters
Pstep=[0.5e-2, 0.5e-2, 0.5e-2]    #Initial steps of the dyn. parameters

#==========================================
#Initialize and read seismograms
#==========================================
#Data for Seis2dpack seismograms

sxDIt=0.
szDIt=0. 
sigmaI=0.
dtDI=0.
tminI=0.

#Data for AXITRA seismograms 
if (NseisE!=0):
    sigmaE = np.ones([NseisE])*0.15 #Seismogram error
    WindowE = np.array([30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
               30., 30.])

    if seisE_input == 0:
        #Read real data for externally computed seismograms
        image = open("base/distXE.dat", "rb")
        dat = np.fromfile(image, dtype=np.float32)
        trace1=dat.reshape(NseisE, int(len(dat)/NseisE))
        image.close()
        image = open("base/distZE.dat", "rb")
        dat = np.fromfile(image, dtype=np.float32)
        trace2=dat.reshape(NseisE, int(len(dat)/NseisE))
        image.close()
        image = open("base/distYE.dat", "rb")
        dat = np.fromfile(image, dtype=np.float32)
        trace3=dat.reshape(NseisE, int(len(dat)/NseisE))
        image.close()

        sxDEt = trace1/100 #from cm to m
        szDEt = trace2/100
        syDEt = trace2/100

        FileID = open('base/dtE.txt', 'r')
        dtDE=np.loadtxt(FileID)
        FileID.close()

        FileID = open('base/tminE.txt', 'r')
        tminE=np.loadtxt(FileID)
        FileID.close()
        
    if seisE_input == 1:
        Fil = open('base/distXE.txt', 'r')
        temp = np.loadtxt(Fil)
        sxDEt = np.transpose(temp[:,1:NseisE+1])
        Fil.close()
        Fil = open('base/distZE.txt', 'r')
        temp =  np.loadtxt(Fil)
        szDEt = np.transpose(temp[:,1:NseisE+1])
        Fil.close() 
        Fil = open('base/distYE.txt', 'r')
        temp =  np.loadtxt(Fil)
        syDEt = np.transpose(temp[:,1:NseisE+1])
        Fil.close()
        tminE=np.zeros(NseisE)
        dtDE=np.ones(NseisE)*dtE
        
else:
    sxDEt=0.
    szDEt=0. 
    syDEt=0.
    sigmaE=0.
    dtDE=0.
    tminE=0.

#==========================================
#Initialize and read GPS
#==========================================
if (NGPS!=0):
    sigmaG = np.ones([NGPS])*0.04 #GPS error
    Fil = open('base/GPSdat.txt', 'r')
    temp = np.loadtxt(Fil)
    Fil.close()
    gxD=temp[:,0]
    gzD=temp[:,1] 
    gyD=temp[:,2]
    
else:
    gxD=0.
    gzD=0. 
    gyD=0.
    sigmaG=0.

#==========================================
#Read prior generator and scaler
#==========================================
if GANdim>0:
    GANscaler = joblib.load('scaler.save')
    GANmodel = keras.models.load_model('./generator', compile=False)
    GANmodel.summary()

#==========================================
#Initialize dynamic parameters
#==========================================   

par=np.zeros([Nfile, Npar])
parf=np.zeros([Nfile, NparTot, 2])
ParInv=np.zeros([Npar*Nfile])

#Read initial model and data structure

for i in range(Nfile):

    FileID = open('base/%s'%FileName[i], 'r')
    parf[i,:,:]=np.loadtxt(FileID)
    FileID.close()    
    
for i in range(Nfile):
    if i==0: #Stress drop is set as fraction of 1e6
        par[i,:]= (-parf[1,ParBorder[0]:ParBorder[1],1] - Sn*mu_d)/1.e6
    if i==1:  #strenght excess is set as a fraction of 1e6  
        par[i,:]=   (parf[1,ParBorder[0]:ParBorder[1],1] + Sn*parf[0,ParBorder[0]:ParBorder[1],1])/1.e6    
    if i==2: #Fracture energy is set as fraction of 1e6
        par[i,:]= (Sn*parf[0,ParBorder[0]:ParBorder[1],1] - Sn*mu_d)*parf[2,ParBorder[0]:ParBorder[1],1]*0.5/1.e6

parGAN = np.ones(GANdim)*[0.5430392653950311121491, 0.6474634968977529547729, 0.4248961798479920504157]
ndim = GANdim + Nfile*(Npar-np.abs(GANBorder[0]-GANBorder[1])) #number of inverted parameters

tt = par[:,0:GANBorder[0]-ParBorder[0]]
ttt=np.append(tt, par[:,GANBorder[1]-ParBorder[0]:ParBorder[1]-ParBorder[0]], axis=1)
ParInv=np.append(parGAN, np.reshape(ttt,[Nfile*(np.abs(ParBorder[0]-GANBorder[0]) + np.abs(ParBorder[1]-GANBorder[1]))]))

p0 = ParInv  #Initial model is taken from the base folder
print(p0.size, ndim)

#==========================================
#Initialize and run the dynamic inversion
#==========================================

#Set up initial covariance matrix with Pstep on the diagonal 
cov=np.eye(ndim)
cov[0:GANdim]=cov[0:GANdim]*GANstep
cov[GANdim+0:GANdim+Npar]=cov[GANdim+0:GANdim+Npar]*Pstep[0]
cov[GANdim+Npar:GANdim+2*Npar]=cov[GANdim+Npar:GANdim+2*Npar]*Pstep[1]
cov[GANdim+2*Npar:GANdim+3*Npar]=cov[GANdim+2*Npar:GANdim+3*Npar]*Pstep[2]

#Populate the folder for forward simulation 
src_dir="base"
#size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
ichain = rank
dest_dir = ("%s/%i" %(name,ichain))
shutil.copytree(src_dir, dest_dir)

scl=SeismoLikelihood()

sampler = PTMCMCSampler.PTSampler(ndim, scl.lnlikefn, scl.lnpriorfn, np.copy(cov),outDir='./chains', verbose=True)
step=np.diag(cov)
newjump = LogNormJump(step)
sampler.addProposalToCycle(newjump.jump, 70)

sampler.sample(p0, Nstep, burn=1000, thin=1, covUpdate=100, isave = 1, Tmin = 1, Tskip=10, Tmax = 20, writeHotChains=True,
               SCAMweight=0, AMweight=0, DEweight=30, NUTSweight=0, HMCweight=0, MALAweight=0)
