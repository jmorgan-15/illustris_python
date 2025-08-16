import numpy as np
import h5py
import six
import requests
import pickle
import astropy
import os
import copy
from .snapshot import loadHalo, loadSubhalo
from .sublink import loadTree
from illustris_python.util import partTypeNum
from os.path import isfile
from astropy import constants as c, units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value


sim_edge=[0, 75000]
basepath='./TNG100-1/output/'
h=0.6774
H0=100*h*u.km.to('Mpc')
G=c.G.to('kpc3 / (solMass s2)').value*10**10  #this converts G to correspond with Illustris units (aside from a factor of h) 
snap_info=pickle.load(open('tng_snap_info', 'rb'))
redshift_list=np.array([snap['redshift'] for snap in snap_info])  #IN REVERSE CHRONOLOGICAL ORDER
age_list=np.array([snap['universe_age'] for snap in snap_info])
kpctokm=3.086e16

#These are halo/sub fields that are commonly used to filter out/select objects. These are not all fields available!
halo_fields=['GroupFirstSub', 'GroupNsubs', 'GroupPos', 'GroupVel', 'Group_R_Crit200', 'GroupMass', 'GroupMassType', 'GroupWindMass', 'GroupSFR', 'GroupGasMetalFractions', 'GroupGasMetallicity', 'GroupStarMetalFractions', 'GroupStarMetallicity']
sub_fields=['SubhaloGrNr', 'SubhaloPos', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloVmaxRad', 'SubhaloMass', 'SubhaloMassType', 'SubhaloWindMass', 'SubhaloMassInHalfRad', 'SubhaloMassInHalfRadType', 'SubhaloMassInMaxRad', 'SubhaloMassInMaxRadType', 'SubhaloMassInRad', 'SubhaloMassInRadType', 'SubhaloSFR', 'SubhaloSFRinHalfRad', 'SubhaloStarMetallicity', 'SubhaloStarMetallicityHalfRad', 'SubhaloGasMetallicity', 'SubhaloGasMetallicityHalfRad',  'SubhaloBfldDisk', 'SubhaloBfldHalo', 'SubhaloVelDisp', 'SubhaloVmax']
tree_fields=halo_fields+sub_fields+['SubfindID']

tree_props_no_interpolation=('DescendantID', 'FirstProgenitorID', 'FirstSubhaloInFOFGroupID', 'GroupFirstSub', 'GroupLen', 'GroupNsubs', 'LastProgenitorID', 'MainLeafProgenitorID', 'NextProgenitorID', 'NextSubhaloInFOFGroupID', 'NumParticles', 'RootDescendantID', 'SubfindID', 'SubhaloGrNr', 'SubhaloID', 'SubhaloIDMostbound', 'SubhaloIDRaw', 'SubhaloLen', 'SubhaloParent', 'TreeID')


tree_props_multidim=('GroupCM', 'GroupGasMetalFractions', 'GroupLenType', 'GroupMassType', 'GroupPos', 'GroupStarMetalFractions', 'GroupVel', 'SubhaloCM', 'SubhaloGasMetalFractions', 'SubhaloGasMetalFractionsHalfRad', 'SubhaloGasMetalFractionsMaxRad', 'SubhaloGasMetalFractionsSfr', 'SubhaloGasMetalFractionsSfrWeighted', 'SubhaloHalfmassRadType', 'SubhaloLenType', 'SubhaloMassInHalfRadType', 'SubhaloMassInMaxRadType', 'SubhaloMassInRadType', 'SubhaloMassType', 'SubhaloPos', 'SubhaloSpin', 'SubhaloStarMetalFractions', 'SubhaloStarMetalFractionsHalfRad', 'SubhaloStarMetalFractionsMaxRad', 'SubhaloStellarPhotometrics', 'SubhaloVel')


#Most snapshots are "mini" and do not have all fields; we use these fields for them instead
#halo_fields_limited=['GroupFirstSub', 'GroupNsubs', 'GroupMassType', 'Group_R_Crit200', 'GroupSFR', 'GroupWindMass', 'GroupPos']
#subhalo_fields_limited=['SubhaloGrNr', 'SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloHalfmassRadType']
#tree_fields_limited=halo_fields_limited+subhalo_fields_limited+['SnapNum']

circ_fields=['CircAbove07Frac', 'CircAbove07Frac_allstars', 'CircAbove07MinusBelowNeg07Frac', 'CircAbove07MinusBelowNeg07Frac_allstars', 'CircTwiceBelow0Frac', 'CircTwiceBelow0Frac_allstars', 'MassTensorEigenVals', 'ReducedMassTensorEigenVals', 'SpecificAngMom', 'SpecificAngMom_allstars', 'SubfindID']
special_SF_fields=[]
special_SF_types, special_SF_times=['SFR_MsunPerYrs_in_r5pkpc_', 'SFR_MsunPerYrs_in_InRad_', 'SFR_MsunPerYrs_in_r30pkpc_', 'SFR_MsunPerYrs_in_all_'], [10,50,100,200,1000]
special_SF_fields.extend([sftype+f'{x}Myrs' for sftype in special_SF_types for x in  special_SF_times])

def temp_calc(u, x_e):
  X_H=0.76 #"Hydrogen mass function"
  k_B=1.380648e-16 #in CGS (ergs)
  UEUM=(3.086e21/3.1536e16)**2 #unit energy/unit mass. It is equal to (unitLength/unitTime)**2=(kpc/Gyr)**2, in CGS. 3.086 cm in kpc, 3.1536e16s in Gyr 
  mp=1.6726e-24 #proton mass in grams
  gamma=5.0/3.0
  mu=4./(1+3*X_H+4*X_H*x_e)*mp #mean molecular weight IN GRAMS
  T=(gamma-1.)*(u/k_B)*UEUM*mu
  return T, mu


   
#def orientation(origin, groupvel, radial_cut, coordinates, velocities, masses, otype):
def orientation(primary_info, particle_info, otype):
  #primary_info is a list of position, velocity, and stellar halfmass radius/other radial cut. 
  #particle info is the coords, velocities, and masses of particles
  dxdydz, dvxdvydvz, masses=particle_info[0]-primary_info[0], particle_info[1]-primary_info[1], particle_info[2]
  r2=np.sum((dxdydz)**2, axis=1) #NOTE r IS SQUARED STILL. Multiply by (scale/h)**2 for physical units. 
  mask=(r2<primary_info[2]**2)
  if otype=='L':
    Lx=np.sum(masses[mask]*(dxdydz[mask, 1]*dvxdvydvz[mask, 2]-dxdydz[mask, 2]*dvxdvydvz[mask, 1]))
    Ly=np.sum(masses[mask]*(dxdydz[mask, 2]*dvxdvydvz[mask, 0]-dxdydz[mask, 0]*dvxdvydvz[mask, 2]))
    Lz=np.sum(masses[mask]*(dxdydz[mask, 0]*dvxdvydvz[mask, 1]-dxdydz[mask, 1]*dvxdvydvz[mask, 0]))
    L=np.array((Lx, Ly, Lz))
    rot_theta, rot_phi=np.arctan2(np.sqrt(Lx**2+Ly**2), Lz), np.arctan2(Ly, Lx)
    unit_vector=np.array((np.cos(rot_phi)*np.sin(rot_theta), np.sin(rot_phi)*np.sin(rot_theta), np.cos(rot_theta)))
    vals_vec, orientation=np.array((L, unit_vector)), np.array((rot_theta, rot_phi))
  elif otype=='sm':
    I_xx, I_yy, I_zz=np.sum(masses[mask]*dxdydz[mask, 0]**2), np.sum(masses[mask]*dxdydz[mask, 1]**2), np.sum(masses[mask]*dxdydz[mask, 2]**2)
    I_xy, I_xz, I_yz=np.sum(masses[mask]*dxdydz[mask, 0]*dxdydz[mask, 1]), np.sum(masses[mask]*dxdydz[mask, 0]*dxdydz[mask, 2]), np.sum(masses[mask]*dxdydz[mask, 1]*dxdydz[mask, 2])
    inertia_tensor=np.array([np.array([I_xx, I_xy, I_xz]), np.array([I_xy, I_yy, I_yz]), np.array([I_xz, I_yz, I_zz])])
    vals_vec=np.linalg.eigh(inertia_tensor)  #returns (eigenvalues, eigenvectors), though eigenvectors are transposed from what we actualy want; "The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i]. Will return a matrix object if a is a matrix object." We want eigenvectors[i] to give us this, not eigenvectors[:, i]
    sm_vec=vals_vec[1].T
    rot_theta, rot_phi=np.arctan2(np.sqrt(sm_vec[:, 0]**2+sm_vec[:, 1]**2), sm_vec[:, 2]), np.arctan2(sm_vec[:, 1], sm_vec[:, 0])
    orientation=np.array(list(zip(rot_theta, rot_phi)))
  return (vals_vec[0], sm_vec), orientation
  
  



#types 2, 3 unused, tracers respectively
particle_types=['gas', 1, 2, 3, 'star', 'bh']
#particle_types=['gas', 'dm', 'bh']
def offset_calc(snap, halo_info, all_halos, all_subs, base_path=basepath):
#this function calculates the offsets for the subhalos/inner fuzz, returns all particle ids for each halo and the offsets
  halo_list=halo_info['SubhaloGrNr']
  #grouping={str(haloid):{} for haloid in halo_list}
  grouping=[{} for haloid in halo_list]
  for q in range(len(halo_list)):
    haloid=halo_list[q]
    print(haloid)
    grouping[q]['halo_and_sub_properties']={'SubhaloGrNr':haloid}
    #halo_gas=loadHalo(basepath, snap, haloid, 'gas')
    firstsub, nsubs=halo_info['GroupFirstSub'][q], halo_info['GroupNsubs'][q]
    lastsub=firstsub+nsubs
    offsets=[[] for j in range(nsubs+1)]  #we need an extra spot for the inner fuzz
    #Constructing dictionary
    #grouping[haloidstr]['halo_and_sub_properties']['sm_eigenvectors'], grouping[haloidstr]['halo_and_sub_properties']['sm_eigenvalues']=sm_eigenvectors, sm_eigenvalues    
    inner_fuzz_offsets=[]    
    for i in range(len(particle_types)):
      ptype=particle_types[i]
      start_index=0
      if isinstance(ptype, int):   #we don't gather tracer particles, and parttype2 is unused
        inner_fuzz_offsets.append([0, 0])
        for j in range(nsubs):
          offsets[j].append([0, 0])          
        continue
      grouping[q][f'PartType{i}']=loadHalo(base_path, snap, haloid, ptype) #loadHalo and loadSubhalo for some reason use keyword, 'gas', 'star', etc instead of PartType0, PartType4, etc. But we want grouping object to come out with PartTypeN.
      #for each particle type, we find the offsets for each subhalo
      for j in range(nsubs):
        subid=firstsub+j
        sub_particles=loadSubhalo(base_path, snap, subid, ptype, fields='ParticleIDs')
        if isinstance(sub_particles, dict):
          partlen=0
        else:
          partlen=len(sub_particles)
        end_index=start_index+partlen
        offsets[j].append([start_index, end_index])
        start_index=end_index
    #we should now have a list of offsets for each subhalo for each particle type: offsets-->subid-->particle type. We should also have particle properties for each ptype for each halo. Now we just need to get halo/subhalo info   
      inner_fuzz_offsets.append([start_index, grouping[q][f'PartType{i}']['count']])
    offsets[-1]=inner_fuzz_offsets  
    grouping[q]['halo_and_sub_properties']['SubfindID'], grouping[q]['halo_and_sub_properties']['offsets']=np.arange(firstsub, lastsub), np.int64(offsets)
    for hfield in all_halos.keys():
      if hfield=='count':
        continue
      grouping[q]['halo_and_sub_properties'][hfield]=all_halos[hfield][haloid]
    for circ_field in circ_fields:
      grouping[q]['halo_and_sub_properties'][circ_field]=halo_info[circ_field][q]
    for special_SF_field in special_SF_fields:
      grouping[q]['halo_and_sub_properties'][special_SF_field]=halo_info[special_SF_field][q]
    for sfield in all_subs.keys(): 
      if sfield=='count':
        continue 
      #This line below *ALMOST* works perfectly, but it makes an array for SubhaloGrNr that has a lnegth equal to number of subhalos in halo, ie, for a halo 1492 with 3 subhalos, we end up with 
      #grouping['1492']['SubhaloGrNr']==array([1492, 1492, 1492]). Honestly at this point it is easier to fix this when we make new coordinate systems
      #I THINK THIS IS FIXED NOW, JUST AN IF STATEMENT WAS NEEDED. IN FACT, KIND OF A GOOD IDEA TO KEEP THE FIELD, "SUBHALOGRNR" THE SAME LENGTH AS NUMBER OF SUBHALOS, SINCE IT IS FUNDAMENTALLY A SUBHALO PROPERTY, EVEN IF JUST REPEATED FOR ALL SUBHALOS. THEN MAKE HALOID WITH LEN=1. BUT I ALREADY HAVE HALOS EXTRACTED WITH LEN(SUBHALOGRNR)==1, SO OH WELL
      if sfield=='SubhaloGrNr':
        grouping[q]['halo_and_sub_properties'][sfield]=haloid
        grouping[q]['halo_and_sub_properties']['HaloID']=haloid
        continue
      grouping[q]['halo_and_sub_properties'][sfield]=np.array([all_subs[sfield][subid] for subid in np.arange(firstsub, lastsub)])
  return grouping


def general_finder(origin, coords, search_radius):
#ESSENTIAL TO PASS COORDS AS OPENED ARRAY, NOT DATASET! IE, f['PartType0']['Coordinates'][()], never leave off [()]!
#This function works by taking particles near the halo but on the other side of the boundary and "moving" them to coords on the other side, basically on-periodic-ing the boundary. ie, shave off left side, add it to right, or shave off right side, add it to left. Really, we are moving the boundary so that this halo no longer crosses it. I am a genius. 
  near_boundary=False 
  search_origin=copy.deepcopy(origin)
  for i in range(3):
    if origin[i]+search_radius>sim_edge[1]:  #if crossing far boundary, create additional, equidistant "fake" origin at near boundary; pretend halo is located at neagtive position  
      search_origin[i]=-(sim_edge[1]-origin[i])
      near_boundary=True
    elif origin[i]-search_radius<sim_edge[0]:  #if crossing near boundary, create additional, equidistant "fake" origin at far boundary; pretend halo is located at positon past 750000
      search_origin[i]=sim_edge[1]+origin[i]
      near_boundary=True
  if near_boundary:
    mask1=np.sum((coords-origin)**2, axis=1)<(search_radius)**2
    mask2=np.sum((coords-search_origin)**2, axis=1)<(search_radius)**2
    radial_mask=mask1 | mask2
  else:
    radial_mask=np.sum((coords-origin)**2, axis=1)<(search_radius)**2
  return radial_mask 
  
    

  

def ray_transform(theta, phi, origin, ray): #ray object should be passed as [ray_start, ray_end]
  ray_start, ray_end=np.array(ray[0]).reshape(3, 1), np.array(ray[1]).reshape(3, 1) #turns our row vectors into column vectors. Also, a matrix is not the same thing as a compound array; in compound arrays, all operators are done element-wise. This means x*y=[[x11*y11, x12*y12], [x21*y21, x22*y22]], which is NOT how you multiply matrices. To avoid confusion, I just turn arrays into matrices and then back into arrays when done
  R_zT=np.array([[np.cos(phi), np.sin(phi), 0], [-1*np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
  R_yT=np.array([[np.cos(theta), 0, -1*np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
  #rotation matrix
  rotated_start, rotated_end=np.array(R_zT @ R_yT @ ray_start).reshape(1, 3), np.array(R_zT @ R_yT @ ray_end).reshape(1, 3) #returns us to row vectors/arrays
  return np.array([rotated_start[0]+origin, rotated_end[0]+origin])






def lookback_to_snap_index(lookback_time):
  #This function returns the index of the TNG snapshot closest to the requested lookback time; snapshots go backwards in time, so snapshot 97 is at index 2  
  age=age_list[0]-lookback_time
  index=np.argmin(abs(age_list-age))  
  return index

def simple_numMergers(mass_array, merger_thresh):
  initial_masses, final_masses=mass_array[1:], mass_array[:-1]  #remember, indexing weird because of reverse temporal order
  ratio=final_masses/initial_masses-1.
  mergers=np.sum(((merger_thresh<ratio) & (ratio<1./merger_thresh)))
  #Check both ratio and inverse ratio, because we want neither our subhalo of interest nor the subhalo it merges with to be significantly larger or smaller than the other
  return mergers
  
def subhalo_history(subid, snapshot=99, last_indexOI=-51):
  #Last_index_OI defines the last snapshots we may want data from. Basically there as easiest way to enforce same size array for all subhalo histories. This makes the data easier to work with later
  #sub=get(url0+str(snapshot)+'/subhalos/'+str(subid))
  #mpb_file=get(sub['trees']['sublink_mpb'], my_folder='./merger_trees/')  #retrives file
  mpb=loadTree('../TNG100-1/output/', snapshot, subid, onlyMPB=True)
  
  snapnumlist=np.array(mpb['SnapNum'])[::-1]  #This makes the snapnumbers and list of ids go forward in time. This is necessary for interpolation
  
  snapnum_interp=np.linspace(snapnumlist[0], snapnumlist[-1], snapnumlist[-1]-snapnumlist[0]+1)
  missing_snaps=sorted(set(snapnum_interp)-set(snapnumlist))  #this tells us the snapshots that we have to interpolate over
  
  offset_snap=snapnumlist[0]
  #Not all subhalos formed at same snapshot; to make all trees length 100, pad with np.nans at the beginning of all properties for early snapshots
  
  #Some halos are nonexistant during certain snapshots. This shouldn't happen much as it is associated with merging, but in case it does we interpolate values/replace with NaN's
  snapzid={}  
  for key, value in mpb.items():
    if key=='count':
      continue
    reverse_val=np.array(value)[::-1]   #this puts all of our properties in the same temporal order as snapnum-- again, necessary for interpolation
    
    #It's not appropriate to interpolate things like SubfindID. For these, just want to fill empty spots with np.nan
    if key in tree_props_no_interpolation:
      data_holder=[]
      for i in range(len(snapnum_interp)):
        snap=snapnum_interp[i]
        if snap in missing_snaps:
          data_holder.append(-99999)
        else:
          index=np.squeeze(np.where(snapnumlist==snap))[()]
          data_holder.append(reverse_val[index]) 
      snapzid[key]=np.int32(data_holder)[:last_indexOI:-1]  #we REFLIP our data temporally because all halos end at snapshot 99, not all begin at same snapshot
            
    #interp is 1-D only, so if we have vectors we have to interp each component seperately      
    elif key in tree_props_multidim:
      dims=reverse_val.shape[1]
      data_holder=[]
      for dim in range(dims):
        dim_vals=np.interp(snapnum_interp, snapnumlist, reverse_val[:, dim])
        data_holder.append(dim_vals)
      snapzid[key]=np.array(data_holder).T[:last_indexOI:-1]  #we REFLIP our data temporally because all halos end at snapshot 99, not all begin at same snapshot
    
    #If not multidimensional or non-interpolatable, then just interpolate and save the value  
    else:
      snapzid[key]=np.interp(snapnum_interp, snapnumlist, reverse_val)[:last_indexOI:-1]  
      #we REFLIP our data temporally because all halos end at snapshot 99, not all begin at same snapshot
  snapzid['SnapNum']=snapnum_interp[:last_indexOI:-1]  #we replace original snapshot numbers with our interpolated ones
  return snapzid
  
def spherify(comov_pos, comov_vel, scale):
#due to ratios of distance/velocity, need to use scale factor to accurately find theta, phi. This functions takes as arguments comoving but centered cartesean coordinates/velocities in illustris units. It returns spherical coords centered on same origin as cartesean, in same units as illustris (so center coords and rotate them to fit the orientation of the disk first). We basically have to convert everything to physical units to get angular quanities right, then convert back
  rootscale=np.sqrt(scale)
  pos, vel=comov_pos*(scale/h), comov_vel*rootscale #position in physical kpc, velocity in km/s
  x, y, z=pos[:, 0], pos[:, 1], pos[:, 2]
  vx, vy, vz=vel[:, 0], vel[:, 1], vel[:, 2]
  
  r=np.sqrt(x**2+y**2+z**2)
  theta=np.arctan2(np.sqrt(x**2+y**2), z)
  phi=np.arctan2(y, x) 
  
  phidot=(1./((y/x)**2+1.))*(vy/x-vx*y/x**2)/kpctokm
  
  #this is for ease of reading. thetadot is annoying to compute
  form_of_arctan=1./((x**2+y**2)/z**2+1.) #looks like 1/(x**2+1)
  ddtsqrtx2plusy2=(x*vx+y*vy)/np.sqrt(x**2+y**2)
  first_term=ddtsqrtx2plusy2/z
  second_term=vz*np.sqrt(x**2+y**2)/z**2
  thetadot=form_of_arctan*(first_term-second_term)/kpctokm #Finally!

  rdot=(x*vx+y*vy+z*vz)/r
  spherco, sphervel=np.transpose(np.array([r/(scale/h), theta, phi])), np.transpose(np.array([rdot/rootscale, thetadot, phidot]))
  return spherco, sphervel
 





















