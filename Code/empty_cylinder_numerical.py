
# -*- coding: utf-8 -*-
"""
@author: Dan

Elastic-Fluid-PML

Solve in-plane elastic problem in R1<r_dim<R2, and an acoustic fluid
within R2<r<R3. Subject to stress-free boundary conditions at r=R1,
force F around the boundary r=R2, and 1. no shear at r=R1 within the
elastic, 2. continuity of normal displacement between elastic and fluid
and 3. continuity of normal stress

See equations of motion in polar coordinates (no potentials).mw for eqns
within the elastic. Equations and BCs within the fluid:
del^2.(phi)+(cT^2/cF^2)*omega^2*phi=0 (u=i.grad(phi)/omega).
tau_rr=(i/omega)(rhF/rhS)(cF^2/cT^2)del^2(phi), u_r=i.(d(phi)/dr)/omega.


P.S: code is aranged in cells like a Jupiter notebook
"""

import math
import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

from ppolar import ppcolor
import matplotlib.pyplot as plt

from diff_matrix import fd_coeff


def fn_areainverse(area):
    tol=1e-5
    error = np.inf
    
    xi_A=0
    xi_B=1
    
    if area>0.5:
        type_fill='full'
        
        while error>tol:
            
            xi=(xi_A+xi_B)/2
        
            trial_area=fn_shapearea(xi,1,type_fill)/np.pi
        
            if trial_area>area:
                xi_B=xi
            else:
                xi_A=xi
       
            error=abs(trial_area-area)
        
        return xi,type_fill
      
        
    elif area<0.5:
        type_fill='half'
        while error>tol:
            
            
            xi=(xi_A+xi_B)/2
            
            trial_area=fn_shapearea(xi,1,type_fill)/np.pi
        
            if trial_area>area:
                
                xi_A=xi
            else:
                
                xi_B=xi
          
            error=abs(trial_area-area)
            
        return xi,type_fill
    
    
    elif area==0.5:
          print('Answer is a half')
          type_fill='half';
          return
       
def fn_shapearea(xi,R1_,type_fill):
    
    if type_fill == 'half':
      shape_area=(np.arccos(xi)*R1_**2)-(R1_**2*xi*np.sqrt(1-xi**2))
      return shape_area
      
    elif type_fill == 'full':
      shape_area=(np.pi-np.arccos(xi))*R1_**2+(R1_**2*xi*np.sqrt(1-xi**2))
      return shape_area
  
    else:
      print('Shape type unrecognised: breaking')
      return

def fn_farfield(R_max,p_sca,p_sca_r,t_bp,k,plot_correction,theta0=False):
    
    mt=5;
  
    #Interpolate input onto grid with fixed spacing

    Nt_fix=1000;
    t_fix=np.linspace(t_bp[0],t_bp[-1],Nt_fix,endpoint=True)
    
    p_sca_fix=np.zeros(Nt_fix)*0j
    p_sca_r_fix=np.zeros(Nt_fix)*0j

    for n in range(1,Nt_fix+1):
        # Central difference (arb spacing) within the (inner) domain
        I=np.argsort(np.abs((np.mod(t_fix[n-1],2*np.pi)-np.mod(t_bp,2*np.pi))))
        
        i=np.sort(I[np.arange(0,mt)]).astype(int)
        tin_s=t_bp[i-1]
        
        W=fd_coeff(np.mod(t_fix[n-1],2*np.pi),np.mod(tin_s,2*np.pi),0)
        #print(np.mod(t_fix[n-1],2*np.pi),np.mod(tin_s,2*np.pi))
        
        p_sca_fix[n-1]=np.sum(W*p_sca[i-1])
        p_sca_r_fix[n-1]=np.sum(W*p_sca_r[i-1])
  

    if theta0==False:
        #THIS INTEGRATES TO FIND FAR FIELD FOR ALL THETA
        D_sca=np.zeros(Nt_fix)*0j
        for n in range(1,Nt_fix+1):
            t0=t_fix[n-1]
            cc0=np.cos(t_fix)*np.cos(t0);
            ss0=np.sin(t_fix)*np.sin(t0);
            #Notate approximate Green's function by G~F.exp(ikr0)/sqrt(r0), for r0->infty.
            F=-1j/4*np.sqrt(2/np.pi/k)*np.exp(-1j*np.pi/4)*np.exp(-1j*k*(cc0+ss0))
            F_r=-1j/4*np.sqrt(2/np.pi/k)*np.exp(-1j*np.pi/4)*(-1j*k*(cc0+ss0))*np.exp(-1j*k*(cc0+ss0))
            I=(F*p_sca_r_fix-p_sca_fix*F_r)
            D_sca[n-1]=2*np.pi*np.mean(I)
      
            
        print(np.shape(D_sca))
        plt.figure()
        plt.polar(np.append(t_fix,t_fix[0])+plot_correction,np.abs(np.append(D_sca,D_sca[0])))
         

    else:
        #theta0=t0+plot_correction, and t0 is what is required
        t0=theta0-plot_correction
        cc0=np.cos(t_fix)*np.cos(t0)
        ss0=np.sin(t_fix)*np.sin(t0)
        #Notate approximate Green's function by G~F.exp(ikr0)/sqrt(r0), for r0->infty.
        F=-1j/4*np.sqrt(2/np.pi/k)*np.exp(-1j*np.pi/4)*np.exp(-1j*k*(cc0+ss0))
        F_r=-1j/4*np.sqrt(2/np.pi/k)*np.exp(-1j*np.pi/4)*(-1j*k*(cc0+ss0))*np.exp(-1j*k*(cc0+ss0))
        I=(F*p_sca_r_fix-p_sca_fix*F_r)
        D_sca=2*np.pi*np.mean(I)
        t_fix=[]
        
    return D_sca,t_fix
 

#%% PARAMETERS FORM MATERIAL PROPERTIES

#f: Filling fluid, s: Elastic solid, b: Background fluid, p: PML

material_b='water at 0°C'
cF_b=1421 
rh_b=999.8


material_s='iron'
cL_s=5960
cT_s=3260
rh_s=7700

material_f='glycerine at 20°C'
cF_f=1860
rh_f=1258


R2=1
R1=0.9
ome=1.5*(R2*1e3/cT_s*2*np.pi)

xi=fn_areainverse(1/3)
psi=np.pi/3 #Incident angle of plane wave using global theta scale


doenergy=0
dodisplacement=0
dochecks=1
dogrid=0
dopml=0
dofluid=1

mr_f=5
mt_f=5
mr_s=5
mt_s=5
mt_bp=5
mr_bp=5

pt_density=15e3

R1_=R1/R2

gam=cT_s/cL_s
kap=cL_s/cT_s

ar=1

#Nt must be even...
lamS_=(2*np.pi/ome)
Nr_s=max(round(10*(1-R1_)/lamS_),6)


#These are unused when a mesh algorithm is required
Nr_f=100


#%% OUTER MESH PML

print('Run PML outer mesh sub-engine')
pml_multiplier=3
domainlength_multiplier=3
ppw=16.5

lamB=(2*np.pi*cF_b/cT_s/ome)*R2 #Wavelength in fluid
lamB_=lamB/R2
almax=pml_multiplier*ome


R3_ = domainlength_multiplier*2*np.pi/(cT_s*ome/cF_b)+1


print('Fluid domain is {} wavelength long'.format(domainlength_multiplier) )


Nr_b=math.ceil((R3_-1)/lamB_*ppw)
Nr_p=math.ceil(1.5*lamB_*(Nr_b-1)/(R3_-1))

Nt_b=math.ceil(ppw*(2*np.pi*R3_/lamB_))
Nt_b=Nt_b-np.mod(Nt_b,4)

Nt_s=Nt_b

print('almax= {}*ome'.format(pml_multiplier))


#%% COORDINATE GRIDDING 


angle_correction=np.pi/2 #theta=theta_global+angle_correction
plot_correction=0

delr_s=(1-R1_)/(Nr_s-1)
delr_bp=(R3_-1)/(Nr_b-1)
delt=2*np.pi/Nt_b

if (Nr_p*delr_bp)<lamB_:
  print('Lp<lamB, correcting Nr_p: Nr_p= {}'.format(round(lamB/delr_bp))  )
  Nr_p=round(lamB/delr_bp)

Nr_bp=Nr_b+Nr_p
Nbp=(Nr_bp)*Nt_b
Lp=Nr_p*delr_bp

Ns=Nr_s*Nt_s 

Np=Nr_p*Nt_b




r_s = np.linspace(1,R1_,Nr_s,endpoint=True)
r_b = np.linspace(R3_, 1,Nr_b,endpoint=True)
r_p = np.linspace(R3_+delr_bp*Nr_p, R3_+delr_bp,Nr_p,endpoint=True)


r_bp=np.hstack((r_p,r_b))

#t=np.arange(0,2*np.pi-delt,delt)

t=np.linspace(0,2*np.pi-delt,Nt_b,endpoint=True)

rr_s,tt_s=np.meshgrid(r_s,t)
rr_b,tt_b=np.meshgrid(r_b,t)
rr_p,tt_p=np.meshgrid(r_p,t)


tt_bp=np.concatenate((np.column_stack(tt_p).flatten(),np.column_stack(tt_b).flatten()))
rr_bp=np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten()))


#%% THETA DIFFERENTIATION MATRICIES - SUITABLE FOR SOLID AND LIQUIDS


It=sp.identity(Nt_b,format='csc')
D1t=sp.csc_array((Nt_b,Nt_b))
D2t=sp.csc_array((Nt_b,Nt_b))


for n in range(1,Nt_b+1):
  #Central difference (arb spacing) within the (inner) domain
  t1,t2 = (n-(mt_s-1)/2-1),(n+(mt_s-1)/2-1)
  t3 = np.arange((n-(mt_s-1)/2-1),(n+(mt_s-1)/2-1)+1)
  
  i = (np.mod(t3,Nt_b)+1).astype(int)

  t_i=t[i-1]
  #Wrap theta around
  for m in np.arange((mt_s-1)/2,0,-1).astype(int):
      if t_i[m]-t_i[m-1]<0:
          t_i[m-1] = t_i[m-1] - 2*np.pi
      
  for m in range(int((mt_s+1)/2),mt_s+1):
      if t_i[m-1]-t_i[m-2]<0:
          t_i[m-1] = t_i[m-1] + 2*np.pi
  
  W = fd_coeff(t[n-1], t_i, 2)
  D1t[n-1,i-1] = W[:,1]
  D2t[n-1,i-1] = W[:,2]


#%% RADIAL DIFFERENTIATION MATRICIES FOR SOLID ONLY

Ir_s=sp.identity(Nr_s,format='csc')
I_s=sp.identity(Nr_s*Nt_b,format='csc')
Z_s=sp.csc_array((Nt_b*Nr_s,Nt_b*Nr_s))
er_s=np.ones(Nr_s)


mxir1_s=sp.spdiags(1/np.column_stack(rr_s).flatten(),0,Nr_s*Nt_b,Nr_s*Nt_b)
mxir2_s=sp.spdiags(1/np.column_stack(rr_s).flatten()**2,0,Nr_s*Nt_b,Nr_s*Nt_b)
D1r_s=sp.csc_array((Nr_s,Nr_s))
D2r_s=sp.csc_array((Nr_s,Nr_s))


for n in range(1,Nr_s+1):
  if n-((mr_s-1)/2)<1:
    #Forward differencing
    i=np.arange(1,mr_s+1).astype(int)
  elif n+((mr_s-1)/2)>Nr_s:
    #Backward differencing
    i=np.arange((Nr_s-mr_s+1),Nr_s+1).astype(int)
  else:
    #Central differencing
    i=np.arange((n-(mr_s-1)/2),(n+(mr_s-1)/2)+1).astype(int)
  
  #Put the weights into the matrix
  W=fd_coeff(r_s[n-1],r_s[i-1],2)
  D1r_s[n-1,i-1]=W[:,1]
  D2r_s[n-1,i-1]=W[:,2]



D1R_s=sp.kron(D1r_s,It,format='csc')
D2R_s=sp.kron(D2r_s,It,format='csc')
D1T_s=sp.kron(Ir_s,D1t,format='csc')
D2T_s=sp.kron(Ir_s,D2t,format='csc')
DRT_s=sp.kron(D1r_s,D1t,format='csc')



#%% RADIAL DIFFERENTIATION MATRICIES FOR FLUIDS (PML AND BACKGROUND FLUID)

Ir_bp=sp.identity(Nr_b+Nr_p,format='csc')
Zt=sp.csc_array((Nt_b,Nt_b))
I_bp=sp.identity((Nr_b+Nr_p)*Nt_b,format='csc')
Z_bp=sp.csc_array( ( (Nt_b*(Nr_b+Nr_p),Nt_b*(Nr_b+Nr_p)) ) )
                  
mxir1_bp=sp.spdiags(1/np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten())),0,(Nr_b+Nr_p)*Nt_b,(Nr_b+Nr_p)*Nt_b)
mxir2_bp=sp.spdiags(1/np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten()))**2,0,(Nr_b+Nr_p)*Nt_b,(Nr_b+Nr_p)*Nt_b)
D1r_bp=sp.csc_array((Nr_bp,Nr_bp))
D2r_bp=sp.csc_array((Nr_bp,Nr_bp))



for n in range(1,Nr_bp+1):
  if n-((mr_bp-1)/2)<1:
    #Forward differencing
    i=np.arange(1,mr_bp+1).astype(int)
  elif n+((mr_bp-1)/2)>Nr_bp:
    #Backward differencing
    i=np.arange((Nr_bp-mr_bp+1),Nr_bp+1).astype(int)
  else:
    #Central differencing
    i=np.arange((n-(mr_bp-1)/2),(n+(mr_bp-1)/2)+1).astype(int)
 
  #Put the weights into the matrix
  W=fd_coeff(r_bp[n-1],r_bp[i-1],2)
  D1r_bp[n-1,i-1]=W[:,1]
  D2r_bp[n-1,i-1]=W[:,2]

D1R_bp=sp.kron(D1r_bp,It,format='csc')
D2R_bp=sp.kron(D2r_bp,It,format='csc')
D1T_bp=sp.kron(Ir_bp,D1t,format='csc')
D2T_bp=sp.kron(Ir_bp,D2t,format='csc')



#%% SETUP EQUATIONS W/I MATERIALS + COMBINE MATRICIES INTO SINGLE ONE

pp_inc=np.exp(1j*ome*(cT_s/cF_b)*rr_b*(np.cos(psi+angle_correction+np.pi)*np.cos(tt_b)+ \
           np.sin(psi+angle_correction+np.pi)*np.sin(tt_b)))
pp_inc_r=(1j*ome*(cT_s/cF_b)*(np.cos(psi+angle_correction+np.pi)*np.cos(tt_b)+ \
          np.sin(psi+angle_correction+np.pi)*np.sin(tt_b)))*pp_inc 


iph_inc=-(rh_s/rh_b/ome)*pp_inc
iph_inc_r=-(rh_s/rh_b/ome)*pp_inc_r
uur_inc_b=(1/ome)*iph_inc_r
uut_inc_b=(1/ome)*(np.diag(1/r_b) @ (D1t @ iph_inc).T ).T


#EQUATIONS WITHIN THE SOLID
Dop1ur_s=(kap**2)*D2R_s+(kap**2)*mxir1_s*D1R_s-(kap**2)*mxir2_s+mxir2_s*D2T_s+(ome**2)*I_s
Dop1ut_s=((kap**2)-1)*mxir1_s*DRT_s-((kap**2)+1)*mxir2_s*D1T_s
Dop2ur_s=(kap**2)*mxir2_s*D1T_s+((kap**2)-1)*mxir1_s*DRT_s+mxir2_s*D1T_s
Dop2ut_s=(kap**2)*mxir2_s*D2T_s-mxir2_s+mxir1_s*D1R_s+D2R_s+(ome**2)*I_s

Dop_s = sp.vstack((sp.hstack((Dop1ur_s,Dop1ut_s)),sp.hstack((Dop2ur_s,Dop2ut_s))))


#EQUATIONS WITHIN THE FLUID
mxr1mxi_bp=sp.spdiags(np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten()))-R3_ ,0,(Nr_bp)*Nt_b,(Nr_bp)*Nt_b)  
mxir1_bp=sp.spdiags(1/np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten())),0,(Nr_bp)*Nt_b,(Nr_bp)*Nt_b)
mxir2_bp=sp.spdiags(1/np.concatenate((np.column_stack(rr_p).flatten(),np.column_stack(rr_b).flatten()))**2,0,(Nr_bp)*Nt_b,(Nr_bp)*Nt_b)

alpha=sp.spdiags(almax*(rr_bp-R3_)**2/Lp**2,0,Nbp,Nbp)
alphap=sp.spdiags(2*almax*(rr_bp-R3_)/Lp**2,0,Nbp,Nbp)

mxden_bp=sp.spdiags(1/(ar+1j*alpha.diagonal()*cF_b/ome/cT_s),0,Nbp,Nbp)
Dop_b=((D2R_bp+mxir1_bp*D1R_bp+mxir2_bp*D2T_bp)+((cT_s*ome/cF_b)**2)*I_bp)
Dop_p=(mxden_bp**2)*D2R_bp+(mxir1_bp*mxden_bp**2-1j*alphap*(mxden_bp**3)*cF_b/ome/cT_s)* \
      D1R_bp+mxir2_bp*(mxden_bp**2)*D2T_bp+((cT_s*ome/cF_b)**2)*I_bp

Dop_bp=sp.vstack( (  Dop_p[0:Nt_b*Nr_p,:] , Dop_b[Nt_b*Nr_p:Nt_b*(Nr_p+Nr_b),:]  ),format='csc' )



#EQUATIONS COMBINED INTO A SINGLE MATRIX
Z_sbp=sp.csc_array((2*Nt_b*Nr_s,Nt_b*(Nr_bp)))
Z_bps=sp.csc_array((Nt_b*(Nr_bp),2*Nt_b*Nr_s))
Dop=sp.vstack((sp.hstack((Dop_bp,Z_bps)),sp.hstack((Z_sbp,Dop_s))),format='csc')



#%% INSERT BOUNDARY CONDITIONS INTO SINGLE MATRIX + SETUP RHS

#BOUNDARY CONDITIONS INDICIES X_y (condition on X (O:outer,I:inner,E:edge
#(of fluid domain), in y (s:solid,f:fluid))

bcE_p=np.where(np.column_stack(rr_p).flatten()==R3_+delr_bp*Nr_p)
bcO_b=np.where(np.column_stack(rr_b).flatten()==1)
bcO_s=np.where(np.column_stack(rr_s).flatten()==1)
bcI_s=np.where(np.column_stack(rr_s).flatten()==R1_)

#BCVWxy Boundary condition V on surface W (E:r=R3_+Np.delr,O:1,I:R1_) on
#xy (ur,ut,ph for phi)
#(1). d(phi)/dr=0 at r=R3_+delr_bp*Nr_p [BC at end of PML]
BC1Eph=D1R_bp
BC1Eur=Z_bps[:,0:Nr_s*Nt_b]   
BC1Eut=Z_bps[:,0:Nr_s*Nt_b]
#(2). Condition (dphi/dr)-ome.ur.=0 at r=1
BC2Oph=-D1R_bp
BC2Our=ome*I_s
BC2Out=Z_bps[:,0:Nr_s*Nt_b]
#(3). Condition (rh_b/rh_S).ome.phi+tau_rr=0 at r=1
BC3Oph=(rh_b/rh_s)*ome*I_bp
BC3Our=((kap**2)*D1R_s+((kap**2)-2)*mxir1_s)
BC3Out=((kap**2)-2)*mxir1_s*D1T_s
#(4). Condition tau_rt=0 at r=1
BC4Oph=Z_sbp[:,0:Nt_b*(Nr_bp)]
BC4Our=mxir1_s*D1T_s
BC4Out=(D1R_s-mxir1_s)
#(5). tau_rr=0 at r=R1_
BC5Iph=Z_sbp
BC5Iur=((kap**2)*D1R_s+((kap**2)-2)*mxir1_s)
BC5Iut=(((kap**2)-2)*mxir1_s*D1T_s)
#(6). tau_rt=0 at r=R1_
BC6Iph=Z_sbp
BC6Iur=(mxir1_s*D1T_s)
BC6Iut=(D1R_s-mxir1_s)


RHS=np.zeros((2*Nr_s+(Nr_b+Nr_p))*Nt_b)*0j

Dop[bcE_p[0],:]=sp.hstack((BC1Eph[bcE_p[0],:],BC1Eur[bcE_p[0],:],BC1Eut[bcE_p[0],:]),format='csc')


Dop[bcO_b[0]+Nr_p*Nt_b,:]=sp.hstack((BC2Oph[bcO_b[0]+Nr_p*Nt_b,:],BC2Our[bcO_s[0],:],BC2Out[bcO_s[0],:]),format='csc')


inds = bcO_b[0]
RHS[bcO_b[0]+Nr_p*Nt_b]=iph_inc_r[np.unravel_index(inds, pp_inc_r.shape, 'F')]


Dop[bcO_s[0]+Nt_b*(Nr_p+Nr_b),:]=sp.hstack((BC3Oph[bcO_b[0]+Nr_p*Nt_b,:],BC3Our[bcO_s[0],:],BC3Out[bcO_s[0],:]))


inds = bcO_b[0]
RHS[bcO_s[0]+Nt_b*(Nr_p+Nr_b)]=-pp_inc[np.unravel_index(inds, pp_inc.shape, 'F')]

Dop[bcO_s[0]+Nt_b*(Nr_p+Nr_b+Nr_s),:]=sp.hstack((BC4Oph[bcO_s[0],:],BC4Our[bcO_s[0],:],BC4Out[bcO_s[0],:]))

Dop[bcI_s[0]+Nt_b*(Nr_p+Nr_b),:]=sp.hstack((BC5Iph[bcI_s[0],:],BC5Iur[bcI_s[0],:],BC5Iut[bcI_s[0],:]))


Dop[bcI_s[0]+Nt_b*(Nr_p+Nr_b+Nr_s),:]=sp.hstack((BC6Iph[bcI_s[0],:], BC6Iur[bcI_s[0],:], BC6Iut[bcI_s[0],:]))


#%% INVERT AND SOLVE

uu=spl.spsolve(Dop,RHS)

iph_sca_bp=np.reshape(uu[0:Nt_b*(Nr_p+Nr_b)],(Nt_b,(Nr_p+Nr_b)),order='F')


uur_s=np.reshape(uu[Nt_b*Nr_p+Nt_b*Nr_b:Nt_b*Nr_p+Nt_b*Nr_b+Nt_b*Nr_s],(Nt_b,Nr_s),order='F')


uut_s=np.reshape(uu[Nt_b*Nr_p+Nt_b*Nr_b+Nt_b*Nr_s:Nt_b*Nr_p+Nt_b*Nr_b+2*Nr_s*Nt_b],(Nt_b,Nr_s),order='F')


uur_sca_bp=(1/ome)*(D1r_bp @ iph_sca_bp.T).T

                                
      
uut_sca_bp=(1/ome)*(np.diag(1/np.concatenate((r_p,r_b)))*(sp.csc_matrix(D1t)*sp.csc_matrix(iph_sca_bp)).T).T

#%% STRESS TENSOR

#ttxy is the stress tensor (tau) component xy (xy=rr_s,rt).
ttrr_s=(kap**2)*(D1r_s@uur_s.T).T+(kap**2)*(np.diag(1/r_s)@uur_s.T).T-2*(np.diag(1/r_s)@uur_s.T).T+(kap**2)*(np.diag(1/r_s)@(D1t@uut_s).T).T-2*(np.diag(1/r_s)@(D1t@uut_s).T).T
ttrt_s=(np.diag(1/r_s)@(D1t@uur_s).T).T+(D1r_s@uut_s.T).T-(np.diag(1/r_s)@uut_s.T).T
ttrr_sca_bp=-ome*(rh_b/rh_s)*iph_sca_bp

uur_sca_p=uur_sca_bp[:,0:Nr_p]
uut_sca_p=uut_sca_bp[:,0:Nr_p]
ttrr_sca_p=ttrr_sca_bp[:,0:Nr_p]

uur_sca_b=uur_sca_bp[:,Nr_p:(Nr_bp)]
uut_sca_b=uut_sca_bp[:,Nr_p:(Nr_bp)]
ttrr_sca_b=ttrr_sca_bp[:,Nr_p:(Nr_bp)]

uur_b=uur_inc_b+uur_sca_b
uut_b=uut_inc_b+uut_sca_b
ttrr_b=ttrr_sca_b-pp_inc



#%% FAR FIELD COMPUTATION + PLOT

p_sca_spw=-ttrr_sca_b[:,Nr_b-1]
pr_sca_spw=(ome**2)*(rh_b/rh_s)*uur_sca_b[:,Nr_b-1]
D_sca,t_fix=fn_farfield([],p_sca_spw,pr_sca_spw,t,ome*(cT_s/cF_b),plot_correction)
