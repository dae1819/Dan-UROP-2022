# -*- coding: utf-8 -*-
"""
For nicer plots...

@author: Dan
"""


import matplotlib.pyplot as plt
import numpy as np


def pol2cart(phi,rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def ppcolor(rr,tt,ss):
    plot_grid_lines=1
    
    if min(np.shape(rr))==1:
      rr,tt=np.meshgrid(rr,tt)
    
    if np.shape(rr)[0] != np.shape(tt)[0] or np.shape(rr)[1] != np.shape(tt)[1]:
      print('Inputs the wrong size...breaking')
      return
    
    

    
    
    maxr=np.max(rr)
    
    xx,yy=pol2cart(tt,rr)

    plt.pcolor(xx,yy,ss)
    #shading interp could be added??   
      
    
    plt.xlim([-1.1*8/6.2*maxr,1.1*8/6.2*maxr])
    plt.ylim([-1.1*maxr,1.1*maxr])
    
    plt.axis('off')
    
    n_lines=8
    n_crosses=5
    
    if plot_grid_lines==1:
      for i in range(0,n_lines):
        if np.mod(i,2)==1:
            plt.plot([0,1.1*maxr*np.cos(2*np.pi*i/n_lines)],[0,1.1*maxr*np.sin(2*np.pi*i/n_lines)],':',color='white')
         
            plt.plot(np.arange(0,n_crosses)/(n_crosses-1)*maxr*np.cos(2*np.pi*i/n_lines), \
                    np.arange(0,n_crosses)/(n_crosses-1)*maxr*np.sin(2*np.pi*i/n_lines),'o', color='white') 
  	        
        else:
            plt.plot([0,1.1*maxr*np.cos(2*np.pi*i/n_lines)],[0,1.1*maxr*np.sin(2*np.pi*i/n_lines)],':',color='black')
            plt.plot(np.arange(0,n_crosses)/(n_crosses-1)*maxr*np.cos(2*np.pi*i/n_lines),np.arange(0,n_crosses)/ \
      	     (n_crosses-1)*maxr*np.sin(2*np.pi*i/n_lines),'o',color='black')
     
      
     
     
  
    

