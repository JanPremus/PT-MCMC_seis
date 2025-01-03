# PT-MCMC_seis
Paralel Tempering MCMC dynamic earthquake source inversion.
Authors: Jan Premus, Jean Paul Ampuero
Additional necessary libraries and codes include:
  - PTMCMCSampler - Parallel Tempering Markov Chain Monte Carlo code (https://github.com/JanPremus/PTMCMCSampler.git@master), original at (https://github.com/nanograv/PTMCMCSampler)
  - SEM2DPACk - 2.5D dynamic modeling code (https://github.com/jpampuero/sem2dpack)
  - fd3d_seisall - calculation of seismograms from available AXITRA solution (https://github.com/JanPremus/fd3d_seisall) 
  - fd3d_gpsall - calculation of static gps displacements from available OKADA solution (https://github.com/JanPremus/fd3d_gpsall)

F. Cotton and Coutant O., 1997, Dynamic stress variations due to shear faults in a plane-layered medium, GEOPHYSICAL JOURNAL INTERNATIONAL,Vol 128, 676-688
Y. ﻿Okada. Surface deformation due to shear and tensile faults in a half-space. Bull. Seism. Soc. Am., 75, 1135-1154, 1985

Folder /base contains the setup of the forward and inverse problem, including all necessary binaries.
Code PT_GAN.py starts %Nchain parallel chains and creates %Nchain copies of the \base folder to run paralel calculations.
Code sem2d_read_fault.py reads SEM2DPACK outputs, original can be found at (https://github.com/jpampuero/sem2dpack)

