# -*- coding: utf-8 -*-
"""
Hard cylinder, numerical 

@author: Dan
"""

from diff_matrix import fd_coeff

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

import math

import matplotlib.pyplot as plt


import matlab.engine
from ppolar import ppcolor

def mul(array):
    return np.reshape(array, (1,array.size))
    


#%%


ome = 9
c=1
k =ome/c
lam=np.pi
lamF = 2*np.pi/ome
xi = 3*(2*np.pi/k)+1
a = 1

Nr_p = 35
ppw = 16.5
Nr_f=round(ppw*(xi-a)/lamF)

mt = 5
mr_fp = 5
ar = 1
almax = 1

if xi != a:
    delr_fp = (xi-1)/(Nr_f-1)
    
    if Nr_p*delr_fp<lamF:
        print("PML too short. Correcting")
        Nr_p = math.ceil(lamF/delr_fp)
        
    Lp=Nr_p*delr_fp
    Nt=round(ppw*((xi+Lp)*2*np.pi)/lamF)

    N=int(Nt*(Nr_p+Nr_f))


 
    #r_f = np.arange(xi,a,-1*delr_fp)
    r_f = np.linspace(xi,a,Nr_f,endpoint=True)
    

    r_p = np.arange(xi+delr_fp*Nr_p,xi+delr_fp,-1*delr_fp)

    
    
    
    
    Nr_fp=Nr_f+Nr_p
    Np=Nr_p*Nt
    Nf=Nr_f*Nt
    Nfp=(Nr_fp)*Nt
    
    delt = 2*np.pi/Nt
    r_fp = np.concatenate((r_p,r_f))
    
    
    
    t = np.linspace(0,2*np.pi-delt,Nt,endpoint=True)
   
    
    rr_f, tt_f = np.meshgrid(r_f,t)
    rr_p,tt_p = np.meshgrid(r_p,t)



    
    tt_fp = np.concatenate((np.column_stack(tt_p).flatten(),np.column_stack(tt_f).flatten()))
    rr_fp = np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_f).flatten()))
    
    
    
    Lp=(delr_fp*Nr_p)
    
    It = sp.identity(Nt)
    D1t = sp.csc_array((Nt,Nt))
    D2t = sp.csc_array((Nt,Nt))
    
    for n in range(1,Nt+1):
        
        t1,t2 = (n-(mt-1)/2-1),(n+(mt-1)/2-1)
        t3 = np.arange((n-(mt-1)/2-1),(n+(mt-1)/2-1)+1)
        
        
        i = (np.mod(t3,Nt)+1).astype(int)
        
        t_i = t[i-1]
        
        
     
        
        for m in np.arange((mt-1)/2,0,-1).astype(int):
            if t_i[m]-t_i[m-1]<0:
                t_i[m-1] = t_i[m-1] - 2*np.pi
        
                
        for m in range(int((mt+1)/2),mt+1):
            if t_i[m-1]-t_i[m-2]<0:
                t_i[m-1] = t_i[m-1] + 2*np.pi
        
        W = fd_coeff(t[n-1], t_i, 2)
        D1t[n-1,i-1] = W[:,1]
        D2t[n-1,i-1] = W[:,2]
    
    
    
    Ir_fp = sp.identity(Nr_f+Nr_p)
    Zt=sp.csc_array((Nt,Nt))
    I_fp=sp.identity((Nr_f+Nr_p)*Nt)
    Z_fp=sp.csc_array( ( Nt*(Nr_f+Nr_p),Nt*(Nr_f+Nr_p) ) )
    
    if len(rr_f) != 0:
        
        mxir1_fp=sp.spdiags((1/rr_fp),0,((Nr_f+Nr_p)*Nt,(Nr_f+Nr_p)*Nt))
        mxir2_fp=sp.spdiags((1/rr_fp)**2,0,((Nr_f+Nr_p)*Nt,(Nr_f+Nr_p)*Nt))
        
    else:
        mxir1_fp=[]
    
    
    
    D1r_fp=sp.csc_array((Nr_fp,Nr_fp))
    D2r_fp=sp.csc_array((Nr_fp,Nr_fp))
    
    
    

    
    for n in range(1,Nr_fp+1):
        if n-((mr_fp-1)/2)<1:
            #Forward differencing
            i = np.arange(1,mr_fp+1).astype(int)
            
        elif n+((mr_fp-1)/2)>Nr_fp:
            #Backward differencing
            i = np.arange((Nr_fp-mr_fp+1),Nr_fp+1).astype(int)
            
        else:
            #Central differencing
            i = np.arange( (n-(mr_fp-1)/2) ,(n+(mr_fp-1)/2)+1).astype(int)
        
        W=fd_coeff(r_fp[n-1],r_fp[i-1],2)
        D1r_fp[n-1,i-1]=W[:,1]
        D2r_fp[n-1,i-1]=W[:,2]
    
    

    
    
    
    D1R_fp=sp.kron(D1r_fp,It)
    D2R_fp=sp.kron(D2r_fp,It)
    D1T_fp=sp.kron(Ir_fp,D1t) 
    D2T_fp=sp.kron(Ir_fp,D2t)
    
    mxr1mxi_fp=sp.spdiags(rr_fp-xi,0,(Nr_fp)*Nt,(Nr_fp)*Nt)
    mxir1_fp=sp.spdiags(1/rr_fp,0,(Nr_fp)*Nt,(Nr_fp)*Nt)
    mxir2_fp=sp.spdiags(1/rr_fp**2,0,(Nr_fp)*Nt,(Nr_fp)*Nt)         
    
        
    alpha=sp.spdiags(almax*(rr_fp-xi)**2/Lp**2,0,Nfp,Nfp)
    alphap=sp.spdiags(2*almax*(rr_fp-xi)/Lp**2,0,Nfp,Nfp) 
    

    mxden_fp=sp.spdiags(1/(ar+1j*alpha.data*ome),0,Nfp,Nfp)
    
    
    Dop_f=((D2R_fp+mxir1_fp*D1R_fp+mxir2_fp*D2T_fp)+(ome**2)*I_fp)
    
    Dop_p=(mxden_fp**2)*D2R_fp+(mxir1_fp*(mxden_fp**2)-1j*alphap*(mxden_fp**3)*ome)*D1R_fp+mxir2_fp*(mxden_fp**2)*D2T_fp+(ome**2)*I_fp   
    
    
   
    Dop_fp=sp.vstack((Dop_p[0:Nt*Nr_p,:],Dop_f[Nt*Nr_p:Nt*(Nr_p+Nr_f),:])).tocsc()
 
   
    bcE_p=np.where(np.column_stack(rr_p).flatten()==xi+delr_fp*Nr_p)
    bcO_f=np.where(np.column_stack(rr_f).flatten()==1)
    BC1Eph=D1R_fp.tocsc()
    BC2Oph=D1R_fp.tocsc()
    Dop_fp[bcE_p[0],:]=BC1Eph[bcE_p[0],:]
    Dop_fp[bcO_f[0]+Nr_p*Nt,:]=BC2Oph[bcO_f[0]+Nr_p*Nt,:]  
    
    
    pp_inc=np.exp(1j*k*rr_f*(np.cos(lam-np.pi)*np.cos(tt_f)+np.sin(lam-np.pi)*np.sin(tt_f)))
    pp_inc_r=1j*k*(np.cos(lam-np.pi)*np.cos(tt_f)+np.sin(lam-np.pi)*np.sin(tt_f))*pp_inc
  


    F = np.zeros(Nfp)*0j
    
  
    inds = Nt*(Nr_f-1)+np.arange(1,Nt+1)-1
    F[Np+Nt*(Nr_f-1)+np.arange(1,Nt+1)-1] = -1*pp_inc_r[np.unravel_index(inds, pp_inc_r.shape, 'F')]      
            
    
    
    
    u = spl.spsolve(Dop_fp,F)
    
    uu_pml = np.reshape(u[0:Np],(Nt,Nr_p),order='F')
    
    pp_sca=np.reshape(u[Np:Nfp],(Nt,Nr_f),order='F')
    
    pp=pp_sca+pp_inc
    
    #%%

   
    
    ppcolor(rr_f,tt_f,np.abs(pp))
    


    
    
