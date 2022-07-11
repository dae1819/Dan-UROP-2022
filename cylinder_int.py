# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 18:42:07 2022

@author: Dan
"""

import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt

class cylinder_int:
    
    def __init__(self,k,a,theta_i,phi_i,is_hard=True):
        self.k = k
        self.a = a
        self.ka = k*a
        
        self.theta_i = theta_i
        self.phi_i = phi_i
        
        self.is_hard = is_hard
        
        
    def directivity(self,phis,N=40):
        try:
            sums = np.zeros(len(phis))
        except:
            sums = 0
            
        
        factor = np.sqrt(2/(np.pi*self.k))*np.exp(-1j*np.pi/4)
        for n in range(-N,N+1):
                
            if self.is_hard:
                Jn_prime = 0.5*(sc.jn(abs(n)-1,self.ka*np.sin(self.theta_i))-
                                sc.jn(abs(n)+1,self.ka*np.sin(self.theta_i)))
                
                Yn_prime = 0.5*(sc.yn(abs(n)-1,self.ka*np.sin(self.theta_i))-
                                sc.yn(abs(n)+1,self.ka*np.sin(self.theta_i)))
                    
                Hn_prime = Jn_prime + 1j*Yn_prime
                    
                ratio = Jn_prime/Hn_prime
                
            
            else:
                Jn = sc.jn(abs(n),self.ka*np.sin(self.theta_i))
                Yn = sc.yn(abs(n),self.ka*np.sin(self.theta_i))
                           
                Hn = Jn + 1j*Yn
                
                ratio = Jn/Hn
            
            
            exp = np.exp(1j*n*(phis-self.phi_i))
            sums = sums + ((-1)**(np.abs(n)))*ratio*exp
            
            
        return -1*factor*sums
    
#%%

hard_int = cylinder_int(6, 1, np.pi/2, np.pi)
soft_int = cylinder_int(6, 1, np.pi/2, np.pi,is_hard=False)

phis = np.linspace(0,2*np.pi,300)

hard_dirs = hard_int.directivity(phis)       
soft_dirs = soft_int.directivity(phis)


#plt.polar(phis,hard_dirs)
plt.polar(phis,soft_dirs)     

#%%
ks = np.linspace(1,100,100)
inters = [cylinder_int(k,1,np.pi/2, np.pi) for k in ks]  
dirs = [inter.directivity(np.pi) for inter in inters]

plt.plot(ks,dirs)
plt.xlabel("k")
plt.ylabel("Backscattered Directivity")
      