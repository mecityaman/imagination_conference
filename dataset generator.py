# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:37:04 2019

@author: myaman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, datetime, time

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)


CHEMICAL_UNIVERSE = pd.read_pickle('chemicals\chemicals 0-117.pkl')
CHEMICAL_UNIVERSE.drop(CHEMICAL_UNIVERSE.tail(5).index,inplace=True)
N_CHANNELS = len(CHEMICAL_UNIVERSE)

def create_molecules(components=[]):
 
  if type(components) != list:
    component_set = np.random.choice(CHEMICAL_UNIVERSE.columns, components)
    return component_set
  return components

def create_archetypes(universal_set,
                           n_classes,
                           n_components,
                           fractionlimits=(0.18,0.5)):
  
  # an archetype is a unique mix of a set of molecules and their fractions
  # in some instances this can be same molecules but different concentrations
 
  
  
  if len(universal_set) < n_components:
    print('number of classes required is more than available components,', end=' ')
    print('repeated chemicals from the universal set will occur.')
  
  analytes = np.random.choice(universal_set,(n_classes,n_components))
  sigma, mu = fractionlimits
  
  
  fractions = np.sort(sigma*np.random.randn(n_classes,n_components)+mu)
  n_universe = CHEMICAL_UNIVERSE.shape[1]
  archetype_matrix = np.zeros((n_classes,n_universe))

  for i,(j,k) in enumerate(zip(analytes,fractions)):
    archetype_matrix[i][j]=k

  plt.rcParams["figure.figsize"] = 12,9
  plt.matshow(archetype_matrix, cmap='gray', aspect='auto', vmax=1, vmin=0)
  plt.xlabel('chemicals')
  plt.ylabel('classes')
  plt.show()
  plt.rcParams["figure.figsize"] = 8,6
  plt.title('fraction profile')
#  plt.plot(np.sort(fractions.flatten()),'o',alpha=0.15)
  plt.plot(sigma*np.random.randn(n_classes*n_components)+mu,'.')
  plt.show()
  time.sleep(3)
  return analytes, fractions


def measure_archetype_set(meas_set,
                          n_measurements=10,
                          variation_type='uniform',
                          variation=0.01,
                          noise_factor=0.005,
                          water=0,
                          CO2=0,
                          debug=False):
  
  def modify_fractions(fractions,
                       variation_type='uniform',
                       variation=variation):
    # play Baroque fugues here
    
    if variation_type=='uniform':
      '''
      add uniform gaussian variation to all components
      '''
      percent_fraction_variation = np.random.normal(0,variation,len(fractions))
      fractions = fractions + fractions*percent_fraction_variation
      delta = percent_fraction_variation
    
    if variation_type=='Gaussian':
      '''
      add a Gaussian variation that is most strongest at the nth
      component (the most disturbed) and decays away from the central
      disturbed component
      '''
      uniform_variation, gaussian_intensity, center, width = variation
      x = np.arange(len(fractions))
      gaussian_variation = gaussian_intensity* np.exp(-(1/width)*((x-center)**2))
      percent_fraction_variation = np.random.normal(0,uniform_variation,len(fractions))
      
      fractions = fractions + fractions*(percent_fraction_variation+gaussian_variation)
      delta = (percent_fraction_variation+gaussian_variation)
    
    return fractions, delta
  
  analytes_container, fractions_container = meas_set
  n = len(analytes_container)
  
  measurements = np.zeros((n_measurements, N_CHANNELS))
  labels = np.zeros(n_measurements, dtype=int)
  
  for i in range(n_measurements):
    
    j = np.random.randint(0,n)
    
    j=i%5
    
    analytes, fractions = analytes_container[j], fractions_container[j]
    
    temp = CHEMICAL_UNIVERSE[analytes]
    temp2 = CHEMICAL_UNIVERSE[analytes]
    
    new_fractions,delta = modify_fractions(fractions,
                                     variation_type=variation_type,
                                     variation=variation)
    
    temp  = (1 - (1-temp)  * new_fractions)
    temp2 = (1 - (1-temp2) * fractions) # no variation for debugging.
    
    spectrum = temp.product(axis=1)
    spectrum2 = temp2.product(axis=1)
    
    noise = np.random.randn(N_CHANNELS)*noise_factor
    spectrum = spectrum + noise  
    spectrum2 = spectrum2 + noise  
    
    measurements[i] = spectrum
    labels[i] = j
    
    if debug:
      print('meas %d analyte %d' %(i,j))  
      
#      print('analytes', analytes_container[j])
#      print()
#      print('fractions', [float(str(i)[:4]) for i in fractions])
#      print('new fractions', [float(str(i)[:4]) for i in new_fractions])
#      print()
#      print('variations', [float(str(i)[:4]) for i in new_fractions])
      variations = np.abs(new_fractions-fractions)
      fu = fractions+variations
      fl = fractions-variations
      xaxs = np.arange(len(fl))
      plt.ylim(-1.20,1.25)
      plt.plot(fractions,alpha=.35)
#      plt.plot(new_fractions, alpha=.1)
      plt.fill_between(xaxs, fu, fl,alpha=.2)
      plt.fill_between(xaxs, variations,alpha=.15)
      plt.plot(xaxs,delta-1,'-', alpha=.25)
      
      plt.title('%s variation '%(variation_type)+str(variation))
      plt.show()
       
      print()
      plt.ylim(0,1.1)
      xaxs = spectrum.index
      plt.plot(spectrum, alpha=0.75)
      plt.xticks([])
      plt.yticks([])
      
#      plt.plot(spectrum2, alpha=0.25)
#      plt.fill_between(xaxs,spectrum,spectrum2,alpha=.94)
      
      titlestring = '#%d, %d analytes. noise %.1f, variation '%(j, len(analytes),noise_factor)+str(variation)
      
      
      plt.title(titlestring)
      
      plt.show()
      
      time.sleep(debug)
      
    else:
      print(i, end=' ')


  print()            
  return measurements, labels
  

def make_data(n_molecules = 116,
              n_classes = 20,
              n_components = 5,
              fraction_limits=(0.0,0.5),
              n_train = 100,
              n_test = 50,
              variation_type  = 'uniform',
              variation=0.01,
              noise_factor=0.01,
              filename = '',
              debug=0):
  
  molecules = create_molecules(n_molecules)
  print('molecules', molecules)
  print('\ncombinations space',ncr(n_molecules,n_components))
  archetypes = create_archetypes(molecules,
                              n_classes=n_classes,
                              n_components=n_components,
                              fractionlimits=fraction_limits)

  analytes,fractions = archetypes
  print('\narchetypes', analytes)

  time.sleep(3)
  
#  print('\nfractions',fractions)
#  print()
#  
  train_data = measure_archetype_set(archetypes,
                      n_measurements=n_train,
                      variation_type  = variation_type,
                      variation=variation,
                      noise_factor=noise_factor,
                      debug=debug)

  train_metadata = {'n_molecules'     : n_molecules,
              'n_classes'       : n_classes,
              'n_components'    : n_components,
              'fraction_limits' : fraction_limits,
              'n_measurements'  : n_train,
              'variation'       : variation,
              'variation_type' : variation_type,
              'noise_factor'    : noise_factor,
              'analytes'        : analytes,
              'fractions'       : fractions,
              'molecules'       : molecules}
  
  test_data = measure_archetype_set(archetypes,
                      n_measurements=n_test,
                      variation_type=variation_type,
                      variation=variation,
                      noise_factor=noise_factor,
                      debug=debug)

  test_metadata = {'n_molecules'     : n_molecules,
              'n_classes'       : n_classes,
              'n_components'    : n_components,
              'fraction_limits' : fraction_limits,
              'n_measurements'  : n_test,
              'variation_type' : variation_type,
              'variation'       : variation,
              'noise_factor'    : noise_factor,
              'analytes'        : analytes,
              'fractions'       : fractions,
              'molecules'       : molecules}
  
  full_set = (train_data,train_metadata),(test_data,test_metadata)
  
  if filename == '':
    filenumber = str(len(os.listdir('data')))
    timestamp =' '+ str(datetime.datetime.now()).replace(':','-')[:-7]
    filename = '%3d %3dx%d  sig%.1f-mu%.1f  var=%s noise=%5.2f %5d-%d'%(n_molecules, n_classes, n_components,
                   fraction_limits[0],fraction_limits[1],
                   str(variation), noise_factor,
                   n_train, n_test)
    filename = filenumber+' ' +filename+timestamp
    
  np.save('data/'+filename, full_set)
  print('\ndata/ '+filename, '.npy saved.')
  print('\a')
  return full_set


data = make_data( n_molecules     = 117,
                  n_classes       = 100,
                  n_components    = 50,
                  fraction_limits = (0.2,0.5), # sigma=0.18 and mu=.5
                  # to see the distribution, plt.plot(0.18*np.random.randn(1000)+0.5, '.')
                  
#                  variation_type  = 'Gaussian',
#                  variation       = (0.5, 2.50, 75, 5),
                  variation_type  = 'uniform',
                  variation       = 0.5,
#                  
                  
                  noise_factor    = 0.5,
                  
                  filename        = '',
                  
                  n_train         = 1000,
                  n_test          = 300, 
                  
                  debug           = 2.0)