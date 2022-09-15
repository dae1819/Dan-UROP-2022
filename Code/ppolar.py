# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:33:39 2022

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
    
    
    # tt=np.vstack((tt,np.zeros(np.shape(tt)[1])))
    # rr=np.vstack((rr,rr[0,:]))
    # ss=np.vstack((ss,ss[0,:]))
    
    
    maxr=np.max(rr)
    
    xx,yy=pol2cart(tt,rr)

    plt.pcolor(xx,yy,ss)
    #shading interp   
      
    #axis([-1.1*8/6.2*maxr,1.1*8/6.2*maxr,-1.1*maxr,1.1*maxr])
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
     
      
      
      # for i in range(0,n_crosses)
      #   r_plot=i/(n_crosses-1)*maxr
      #   text(-r_plot*cos(pi/2-1/20),-r_plot*sin(pi/2-1/20),num2str(r_plot),'color','white')
      #   text(r_plot*cos(pi/2-1/20),r_plot*sin(pi/2-1/20),num2str(r_plot))
      
      #if wasitheld==0
        #hold off
     
  
    

