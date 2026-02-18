

#%% ################################################################################
# Set these parameters according to file locations on local computer

# Where to put plots
OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/OUTPUT202602/'
# Where to find synthetic images
SYNTHETIC_IMAGE_PATH = '/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_MW_EatingDispersion/Synthetic_data/'


#%% ################################################################################

from PIL import Image
import numpy as np
import pandas as pd 

import math

from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, disk

from scipy.signal import correlate
    # from scipy.signal import correlate2d # VERY SLOW
from scipy.spatial.distance import pdist

import cv2
import scipy.ndimage as ndi

import matplotlib.pyplot as plt
import seaborn as sns
import time # for debugging/optimization

import glob
import os

cm_to_inch = 1/2.54

#%% ################################################################################
# Create output dir if it doesn't exist

# Create in __main__ to avoid side effects during import

#%% ################################################################################
# Functions

def get_largest_mask(img, method='bg10', return_status=False, apply_smooth=False):
    """
    Finds Otsu threshold for image, applies threshold, and
    then selects the largest continuous region.
    Returns that region as binary mask.
    
    mask_user can be used to ignore parts of the input image (img)
    """
    # img = img_leaf; method='bg10'
    
    if method == 'otsu':
        threshold_val = threshold_otsu(img)
    elif method == 'triangle':
        threshold_val = threshold_triangle(img)
    elif method == 'bg10':
        # using percentile
        # threshold_val = 10*np.percentile(img.ravel(), 3)
        # determine mode
        threshold_val = 10 * np.bincount(img.ravel()).argmax()
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'otsu', 'triangle', or 'bg10'.")
        
    img_mask = img > threshold_val
    if not np.any(img_mask):
        if return_status:
            return np.zeros(img.shape, dtype=bool), False
        return np.zeros(img.shape, dtype=bool)

    img_lbl = label(img_mask)
    
    lbl_largest = np.argmax([region.area for region in regionprops(img_lbl)]) + 1
    
    img_mask = img_lbl == lbl_largest
    
    if apply_smooth:
        # perform morphological closing with a radius of 5 pixels to smooth the mask
        img_mask = opening(img_mask, disk(10))
        # if this erased the mask, act accordingly
        if not np.any(img_mask):
            if return_status:
                return np.zeros(img.shape, dtype=bool), False
            return np.zeros(img.shape, dtype=bool)
        
    if return_status:
        return img_mask, True
    return img_mask

def get_mask(img, mask_user=None, method='otsu', return_status=False):
    
    if mask_user is None:
        mask_user = np.ones(img.shape, dtype=bool)

    if not np.any(mask_user):
        if return_status:
            return np.zeros(img.shape, dtype=bool), False
        return np.zeros(img.shape, dtype=bool)
        
    if method == 'otsu':
        threshold_val = threshold_otsu(img[mask_user])
    elif method == 'triangle':        
        threshold_val = threshold_triangle(img[mask_user])
    elif method == 'bg2':        
        threshold_val = 2 * np.bincount(img[mask_user].ravel()).argmax()
    elif method == 'pct10':
        threshold_val = 10 * np.percentile(img[mask_user], 10)
        
    img_mask = img > threshold_val
    found = np.any(img_mask & mask_user)
    if return_status:
        return img_mask, found
    return img_mask
      
def determine_leaf_roundness(mask_leaf):
    """ Calculate the roundness of the leaf """

    # calculate properties
    the_regionprops = regionprops(label(mask_leaf))
    # check assumption there's only 1 region
    if len(the_regionprops) > 1: 
        raise ValueError("Multiple regions found in mask_leaf, cannot determine roundness.")
    
    # calculate the roundness
    roundness = 4*np.pi*the_regionprops[0].area/the_regionprops[0].perimeter**2
    
    # return the roundness
    return roundness
   
def get_zoombox(mask, margin=0):
    '''
    return coordinates "zoom" to be able
    to zoom on image like img[z1:z2, z3:z4]
    based on mask
    '''
    
    # get bbox
    regions = regionprops(mask.astype(int))
    if len(regions) == 0:
        # return whole image box if regions don't exist
        print("WARNING: TAKING WHOLE IMAGE, NO BBOX IDENTIFIED")
        return [0, mask.shape[0], 0, mask.shape[1]]

    thebbox = regions[0].bbox
    
    # add margin on all sides, taking original mask size into account
    z1 = max(0, thebbox[0] - margin)
    z2 = min(mask.shape[0], thebbox[2] + margin)
    z3 = max(0, thebbox[1] - margin)    
    z4 = min(mask.shape[1], thebbox[3] + margin)
    
    return [z1, z2, z3, z4]
    
    
def plot_images(
    img_leaf,
    img_dmg,
    mask_leaf,
    mask_damage,
    leaf_channel_spec,
    damage_channel_spec,
    centroid_leaf=None,
    img0=None,
    reference_channel_spec=None
):
    
    zm = get_zoombox(mask_leaf, margin=10)
    
    fig, axs = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    
    if not img0 is None and reference_channel_spec is not None:
        ref_idx = reference_channel_spec['channel']
        ref_name = reference_channel_spec['name']
        axs[0].imshow(img0[:, :, ref_idx][zm[0]:zm[1],zm[2]:zm[3]]); axs[0].set_title(f'{ref_name} channel (idx={ref_idx})')
    else:
        axs[0].axis('off')
    
    leaf_idx = leaf_channel_spec['channel']
    leaf_name = leaf_channel_spec['name']
    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]]); axs[1].set_title(f'{leaf_name} channel (idx={leaf_idx}, leaf)')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)    
    if not centroid_leaf is None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    damage_idx = damage_channel_spec['channel']
    damage_name = damage_channel_spec['name']
    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]]); axs[2].set_title(f'{damage_name} channel (idx={damage_idx}, damage)')
    axs[2].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    
    plt.show(); plt.close()

def get_radial_pdf(img, CoM, mask_user=None):
    '''
    Given an image (img) and center of mass (CoM), integrate 
    along phi to get the radial distribution profile
    of the image intensity.
    '''
    
    if mask_user is None:
        mask_user = np.ones(img.shape, dtype=bool)
    
    # Create arrays of y and x coordinates for each pixel
    y, x = np.indices(img.shape)
    # Compute the distance of each pixel from the center of mass (CoM)
    r = np.sqrt((x - CoM[1])**2 + (y - CoM[0])**2)
        # this result in an overlay mask with the r values
        # plt.imshow(r); plt.show(); plt.close()
    
    # Convert distances to integer values (binning by radius)
    r = r.astype(int)
    # Find the maximum radius value (not used further, but could be for plotting)
    r_max = r[mask_user].max() + 1
    
    # Sum the image values for each radius
    radial_sum = np.bincount(r[mask_user].ravel(), img[mask_user].ravel())
    # Count the number of pixels at each radius
    radial_count = np.bincount(r[mask_user].ravel())
    # Normalize by pixels in each bin (np.maximum replaces zeros by ones)
    radial_avg = radial_sum / np.maximum(radial_count, 1)
    
    # now nowmalize such that the sum of the pdf is 1
    if np.sum(radial_avg) == 0:
        radial_pdf = np.zeros_like(radial_avg, dtype=float)
    else:
        radial_pdf = radial_avg / np.sum(radial_avg)
        
    return radial_count, radial_sum, radial_avg, radial_pdf, r_max        

# now calculate the autocorrelation
def get_autocorrelation(img, mask_user=None):
    '''
    Calculate the autocorrelation of an image.
    '''
    
    if mask_user is None:
        mask_user = np.ones(img.shape, dtype=bool)
    
    # apply mask
    img_masked = img.copy()
    img_masked[~mask_user] = 0
    
    # calculate autocorrelation    
    acf = correlate(img_masked.astype(float), img_masked.astype(float), method='fft', mode='full')
    
    # normalize
    acf_max = np.max(acf)
    if acf_max == 0:
        acf_norm = np.zeros_like(acf, dtype=float)
    else:
        acf_norm = acf / acf_max
    
    # also calculate the center coordinate of this acf
    acf_center = np.round(np.array(acf.shape)/2).astype(int)
    
    return acf, acf_norm, acf_center


# now a function that goes over each separate region, ignores the region itself,
# but calculates the distances to nearest neighbors pixels from the other regions
def get_inter_island_distances(mask_leaf, mask_damage):
    '''
    Calculate inter-island distances.
    '''
    # mask_damage = mask_damages['disk']; mask_leaf=mask_leafs['disk']
    
    # get bounding box of the leaf
    zm = get_zoombox(mask_leaf, margin=0)
    
    # get the labels of the damage mask
    lbl_damage = label(mask_damage[zm[0]:zm[1], zm[2]:zm[3]])
        # plt.imshow(lbl_damage); plt.show(); plt.close()
        
    # loop over the labels
    if np.max(lbl_damage)<2:
        return [0]
    else:
        distances = [None]*(np.max(lbl_damage))
        for lbl in np.unique(lbl_damage):
            # lbl=1
            
            # generate lbl map with current island removed
            current_lbl = lbl_damage.copy()
            current_lbl[current_lbl==lbl] = 0        
                # plt.imshow(current_lbl); plt.show(); plt.close()
            
            # generate distance map to closest non-zero pixel 
            # (Speed-test with 1000x running this, showed cv2 is 5x faster than ndi.distance_transform_edt)
            img_dist = cv2.distanceTransform(src = (current_lbl==0).astype(np.uint8),
                                        distanceType=cv2.DIST_L2, 
                                        maskSize=cv2.DIST_MASK_PRECISE)
                # plt.imshow(img_dist); plt.show(); plt.close()
            
            distances[lbl-1] = np.min(img_dist[lbl_damage==lbl])
            
    return distances

def get_island_counts(mask_leaf, mask_damage):
    '''
    Calculate nr of detected islands (regions)
    '''
    # mask_damage = mask_damages['disk']; mask_leaf=mask_leafs['disk']
    
    # get bounding box of the leaf
    zm = get_zoombox(mask_leaf, margin=0)
    
    # get the labels of the damage mask
    lbl_damage = label(mask_damage[zm[0]:zm[1], zm[2]:zm[3]])
        # plt.imshow(lbl_damage); plt.show(); plt.close()
        
    return np.max(lbl_damage)
    

#%% ################################################################################
# Now let's first look at data I generated myself
# Load the synthetic data

# open tiff stack image
from skimage import io

def load_synthetic_data(synthetic_image_path, leaf_channel_spec, damage_channel_spec):
    """
    Load synthetic TIFF stacks and split them into leaf and damage channels.
    """

    leaf_idx = leaf_channel_spec['channel']
    damage_idx = damage_channel_spec['channel']

    img_leafs = {}
    img_damages = {}
    
    # Load the leaf w/ eaten disk
    img_disk_path = synthetic_image_path + 'synthetic_eatendisk.tif'
    img_disk = io.imread(img_disk_path)  # io.read required for img stack
    img_leafs['disk'] = img_disk[:, :, leaf_idx]  # configured leaf channel
    img_damages['disk'] = img_disk[:, :, damage_idx]  # configured damage channel

    # Load the leaf w/ eaten spots
    img_spots_damage_path = synthetic_image_path + 'synthetic_eatenspots.tif'
    img_spots_damage = io.imread(img_spots_damage_path)  # io.read required for img stack
    img_leafs['spots'] = img_spots_damage[:, :, leaf_idx]  # configured leaf channel
    img_damages['spots'] = img_spots_damage[:, :, damage_idx]  # configured damage channel

    # Load the image w/ eaten donut
    img_donut_path = synthetic_image_path + 'synthetic_eatendonut.tif'
    img_donut = io.imread(img_donut_path)  # io.read required for img stack
    img_leafs['donut'] = img_donut[:, :, leaf_idx]  # configured leaf channel
    img_damages['donut'] = img_donut[:, :, damage_idx]  # configured damage channel

    # Load dual-spot sample
    img_dualspot_path = synthetic_image_path + 'synthetic_dualspot.tif'
    img_dualspot = io.imread(img_dualspot_path)  # io.read required for img stack
    img_leafs['dualspot'] = img_dualspot[:, :, leaf_idx]  # configured leaf channel
    img_damages['dualspot'] = img_dualspot[:, :, damage_idx]  # configured damage channel

    return img_leafs, img_damages, img_disk


#%% ################################################################################
# Analysis for multiple synthetic samples

# plot the acf centerline
def plot_img_n_acf(img_damage, acf_norm, acf_center, acf_norms_avgr, name):
    # img_damage = img_damages['disk']; acf_norm = acf_norms['disk']; acf_center = acf_centers['disk']; acf_norms_avgr = acf_norms_avgrs['disk']
    
    fig, axs = plt.subplots(1, 2, figsize=(15*cm_to_inch, 5*cm_to_inch))
    axs[0].imshow(img_damage, cmap='gray')
    
    x_axis = np.arange(acf_norm.shape[1]) - acf_center[1]
    
    axs[1].plot(x_axis, acf_norm[acf_center[0],:], color='grey', linestyle=':', label='1d')
    axs[1].plot(acf_norms_avgr, color='black', linestyle='-', label='Radial average')
    axs[1].set_title(f'Autocorrelation for {name}')
    # axs[1].legend()
    
    plt.show(); plt.close()

# now get masks for leaf and damage, plus centroid for all 
def run_synthetic_analysis(
    img_leafs,
    img_damages,
    img_disk,
    leaf_channel_spec,
    damage_channel_spec,
    reference_channel_spec
):
    """
    Run synthetic-data diagnostics and plots to verify analysis behavior.
    """

    # Build masks and centroids
    mask_leafs = {}
    mask_damages = {}
    centroids = {}
    for key in img_leafs.keys():
        mask_leafs[key] = get_largest_mask(img_leafs[key], method='otsu')
        mask_damages[key] = get_mask(img_damages[key], mask_leafs[key], method='bg2')  # bg2, otsu, triangle, pct10
        centroids[key] = regionprops(mask_leafs[key].astype(int))[0].centroid

    # Visual QC for channels/masks
    for key in img_leafs.keys():
        plot_images(
            img_leafs[key],
            img_damages[key],
            mask_leafs[key],
            mask_damages[key],
            leaf_channel_spec,
            damage_channel_spec,
            centroids[key],
            img0=img_disk,
            reference_channel_spec=reference_channel_spec
        )

    # Autocorrelation analysis for each synthetic sample
    acfs = {}
    acf_norms = {}
    acf_centers = {}
    acf_norms_avgrs = {}
    for key in img_leafs.keys():
        acfs[key], acf_norms[key], acf_centers[key] = get_autocorrelation(img_damages[key], mask_user=mask_leafs[key])
        _, _, acf_norms_avgrs[key], _, _ = get_radial_pdf(acf_norms[key], acf_centers[key])

    # Plot ACF curves
    for key in img_leafs.keys():
        plot_img_n_acf(img_damages[key], acf_norms[key], acf_centers[key], acf_norms_avgrs[key], key)

    # Radial PDFs for synthetic masks
    radial_pdf = {}
    for key in img_leafs.keys():
        _, _, _, radial_pdf[key], _ = get_radial_pdf(mask_damages[key], centroids[key], mask_leafs[key])

    for key in img_leafs.keys():
        fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
        axs[0].imshow(mask_damages[key])
        axs[1].plot(radial_pdf[key])
        plt.show(); plt.close()

    # Summarize island spacing
    total_interisland_distances = {}
    for key in img_leafs.keys():
        interisland_distances = get_inter_island_distances(mask_leafs[key], mask_damages[key])
        total_interisland_distances[key] = np.sum(interisland_distances)

    plt.bar(list(img_leafs.keys()), list(total_interisland_distances.values()))

    # Summarize island counts
    island_counts = {}
    for key in img_leafs.keys():
        island_counts[key] = get_island_counts(mask_leafs[key], mask_damages[key])

    plt.bar(list(island_counts.keys()), list(island_counts.values()))

#%% ######################################################################
# Now let's get real data working

def get_data_file_paths(condition_path_map):
    """
    Collect all TIFF file paths for each condition.

    Parameters
    ----------
    condition_path_map : dict
        User-defined mapping where keys are condition names (e.g. 'infected')
        and values are folder paths that contain TIFF files for that condition.
    """

    data_file_paths = {}
    for condition, base_path in condition_path_map.items():
        data_file_paths[condition] = glob.glob(os.path.join(base_path, '*.tif'))

    return data_file_paths

def run_complete_analysis(data_file_paths, leaf_channel_spec, damage_channel_spec,
                          leaf_threshold_method = 'bg10', leaf_roundness_threshold=0,
                          apply_smooth_leafmask=False,
                          pixel_to_cm2_factor=None):
    """
    Run all analyses (as for synthetic data) for all files in data_file_paths.
    Stores results in dicts for easy plotting and further analysis.
    """

    # Prepare output structures
    leaf_idx = leaf_channel_spec['channel']
    damage_idx = damage_channel_spec['channel']

    img_rgbs = {}
    img_leafs = {}
    img_damages = {}
    mask_leafs = {}
    mask_damages = {}
    centroids = {}
    acfs = {}
    acf_norms = {}
    acf_centers = {}
    acf_norms_avgrs = {}
    radial_pdfs = {}
    total_interisland_distances = {}
    island_counts = {}
    total_damage_area_px = {}
    total_damage_area_cm2 = {}
    leaf_roundnesses = {}
    leaf_found = {}
    damage_found = {}
    analysis_status = {}

    for condition, file_list in data_file_paths.items():
        # condition, file_list = list(data_file_paths.items())[0]
        img_rgbs[condition] = []
        img_leafs[condition] = []
        img_damages[condition] = []
        mask_leafs[condition] = []
        mask_damages[condition] = []
        centroids[condition] = []
        acfs[condition] = []
        acf_norms[condition] = []
        acf_centers[condition] = []
        acf_norms_avgrs[condition] = []
        radial_pdfs[condition] = []
        total_interisland_distances[condition] = []
        island_counts[condition] = []
        total_damage_area_px[condition] = []
        total_damage_area_cm2[condition] = []
        leaf_roundnesses[condition] = []
        leaf_found[condition] = []
        damage_found[condition] = []
        analysis_status[condition] = []

        for file_path in file_list:
            # file_path = file_list[0]
            # file_path = file_list[7]
            
            # Update user on what's happening
            print(f'Processing {file_path} for condition: {condition}')
            
            img = np.array(Image.open(file_path))
            # in case the image doesn't have 3 dimensions, expand to three
            img = np.atleast_3d(img)
            img_leaf = img[:, :, leaf_idx]
            img_damage = img[:, :, damage_idx]

            mask_leaf, this_leaf_found = \
                get_largest_mask(img_leaf, 
                                 method=leaf_threshold_method, 
                                 apply_smooth=apply_smooth_leafmask,
                                 return_status=True)

            # Additional check for leaf validity, check roundness
            if this_leaf_found:
                leaf_roundness = determine_leaf_roundness(mask_leaf)
                if not leaf_roundness > leaf_roundness_threshold:
                    this_leaf_found = False
                    print("WARNING: Leaf roundness below threshold, marking as no leaf found.")
            else:
                leaf_roundness = np.nan
                
            # If leaf detection fails, mark downstream metrics as missing/NA.
            if not this_leaf_found:
                mask_damage = np.zeros_like(mask_leaf, dtype=bool)
                centroid = None
                acf = None
                acf_norm = None
                acf_center = None
                acf_norm_avgr = None
                radial_pdf = None
                total_interisland = np.nan
                island_count = np.nan
                damage_area_px = np.nan
                damage_area_cm2 = np.nan
                this_damage_found = False
                this_status = 'no_leaf_mask'
            else:
                mask_damage, this_damage_found = get_mask(img_damage, mask_leaf, method='bg2', return_status=True)
                centroid = regionprops(mask_leaf.astype(int))[0].centroid

                # If no damage is detected inside leaf, keep valid zeros for damage metrics.
                if not this_damage_found:
                    acf = None
                    acf_norm = None
                    acf_center = None
                    acf_norm_avgr = None
                    radial_pdf = None
                    total_interisland = 0.0
                    island_count = 0
                    damage_area_px = 0.0
                    if pixel_to_cm2_factor is None:
                        damage_area_cm2 = np.nan
                    else:
                        damage_area_cm2 = damage_area_px * pixel_to_cm2_factor
                    this_status = 'no_damage_mask'
                else:
                    acf, acf_norm, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)
                    _, _, acf_norm_avgr, _, _ = get_radial_pdf(acf_norm, acf_center)
                    _, _, _, radial_pdf, _ = get_radial_pdf(mask_damage, centroid, mask_leaf)
                    interisland_distances = get_inter_island_distances(mask_leaf, mask_damage)
                    total_interisland = np.sum(interisland_distances)
                    island_count = get_island_counts(mask_leaf, mask_damage)
                    damage_area_px = float(np.sum(mask_damage))
                    if pixel_to_cm2_factor is None:
                        damage_area_cm2 = np.nan
                    else:
                        damage_area_cm2 = damage_area_px * pixel_to_cm2_factor
                    this_status = 'ok'

            img_rgbs[condition].append(img)
            img_leafs[condition].append(img_leaf)
            img_damages[condition].append(img_damage)
            mask_leafs[condition].append(mask_leaf)
            mask_damages[condition].append(mask_damage)
            centroids[condition].append(centroid)
            acfs[condition].append(acf)
            acf_norms[condition].append(acf_norm)
            acf_centers[condition].append(acf_center)
            acf_norms_avgrs[condition].append(acf_norm_avgr)
            radial_pdfs[condition].append(radial_pdf)
            total_interisland_distances[condition].append(total_interisland)
            island_counts[condition].append(island_count)
            total_damage_area_px[condition].append(damage_area_px)
            total_damage_area_cm2[condition].append(damage_area_cm2)
            leaf_roundnesses[condition].append(leaf_roundness)
            leaf_found[condition].append(this_leaf_found)
            damage_found[condition].append(this_damage_found)
            analysis_status[condition].append(this_status)

    # Return all results as a dictionary of dictionaries/lists
    return {
        'img_rgbs': img_rgbs,
        'img_leafs': img_leafs,
        'img_damages': img_damages,
        'mask_leafs': mask_leafs,
        'mask_damages': mask_damages,
        'centroids': centroids,
        'acfs': acfs,
        'acf_norms': acf_norms,
        'acf_centers': acf_centers,
        'acf_norms_avgrs': acf_norms_avgrs,
        'radial_pdfs': radial_pdfs,
        'total_interisland_distances': total_interisland_distances,
        'island_counts': island_counts,
        'total_damage_area_px': total_damage_area_px,
        'total_damage_area_cm2': total_damage_area_cm2,
        'leaf_roundness': leaf_roundnesses,
        'leaf_found': leaf_found,
        'damage_found': damage_found,
        'analysis_status': analysis_status
    }

# %% ########################################################################

# Generate a plot of the acf_norms_avgrs, all in the same panel, and 
# annotated per condition
def plot_acf_norms_avgrs(data_all, outputdir):
    """
    Plot the average radial autocorrelation for each condition.
    """
    
    os.makedirs(outputdir+'/plots/', exist_ok=True)
    
    fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 10*cm_to_inch))
    
    mycolors = ['blue', 'red']
    
    # loop over keys to get conditions
    for idx, condition in enumerate(data_all['acf_norms_avgrs'].keys()):
        
        # loop over the different acf_norms_avgrs for each condition
        for acf_norms_avgr in data_all['acf_norms_avgrs'][condition]:
            if acf_norms_avgr is None:
                continue
            
            axs[0].plot(acf_norms_avgr, color=mycolors[idx], linewidth=.5)
    
    mylinestyles = ['-',':']
    for idx, condition in enumerate(data_all['acf_norms_avgrs'].keys()):

        valid_acf_norms = [x for x in data_all['acf_norms_avgrs'][condition] if x is not None]
        if len(valid_acf_norms) == 0:
            continue

        # determine the average line per condition, using
        # df like done below
        acf_norms_avgr_avg = pd.DataFrame(valid_acf_norms).mean()
        # plot the average line
        axs[1].plot(acf_norms_avgr_avg, linewidth=2, 
                label=f'Avg {condition}', color=mycolors[idx])#)linestyle=mylinestyles[idx])
        # ax.plot(acf_norms_avgr_avg, color=mycolors[idx], linewidth=.5, label=f'Avg {condition}')
        
    fig.suptitle('Radial Autocorrelation')    
    axs[0].set_xlabel('Radius (pixels)')
    
    axs[1].set_xlabel('Radius (pixels)')
    axs[1].set_ylabel('Normalized Autocorrelation')
    
    plt.tight_layout()
    plt.savefig(outputdir+'/plots/Radial_acf.pdf', dpi=150)
    
    axs[0].set_xlim([0,200]); axs[1].set_xlim([0,200])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(outputdir+'/plots/Radial_acf_lims.pdf', dpi=150)
        
    plt.show(); plt.close()
    
# Now the same for the inter-island distance metric
def plot_interisland_distances(data_all, outputdir, remove_zerocnt=True):
    """
    Plot the total inter-island distances for each condition.
    """    
    
    os.makedirs(outputdir+'/plots/', exist_ok=True)
    
    # create df with separate points 
    df_dist = pd.DataFrame({'cond':[],'total_dist':[],'island_count':[]})
    
    for cond in data_all['total_interisland_distances'].keys():
        # create df with points for this condition, append to total df
        df_dist = \
            pd.concat([df_dist, 
                    pd.DataFrame({'cond':cond,
                                'total_dist':data_all['total_interisland_distances'][cond],
                                'island_count':data_all['island_counts'][cond]})])
    
    # Reset index
    df_dist = df_dist.reset_index(drop=True)
    
    # Now remove datapoints with zero islands
    if remove_zerocnt:
        df_dist = df_dist[df_dist['island_count'] > 0]
        
    fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 10*cm_to_inch))
    
    # plot strippplot using seaborn
    sns.barplot(x='cond', y='total_dist', 
                data=df_dist, ax=axs[0], palette=['blue', 'red'])
    sns.violinplot(x='cond', y='total_dist', 
                   data=df_dist, ax=axs[0], color='black', alpha=0.2)
    sns.stripplot(x='cond', y='total_dist', 
                  data=df_dist, ax=axs[0], color='black')
    
    axs[0].set_title(f'Total Closest-Island\nDistances')
    axs[0].set_ylabel('Distance (pixels)')
    axs[0].set_ylim([0, np.max(df_dist['total_dist']) * 1.02])
    # rotate axis 90 deg
    axs[0].tick_params(axis='x', rotation=45)
    
    # now also plot the island counts
    sns.barplot(x='cond', y='island_count', 
                data=df_dist, ax=axs[1], palette=['blue', 'red'])
    sns.violinplot(x='cond', y='island_count',
                     data=df_dist, ax=axs[1], color='black', alpha=0.2)
    sns.stripplot(x='cond', y='island_count',
                  data=df_dist, ax=axs[1], color='black')
    axs[1].set_ylim([0, np.max(df_dist['island_count']) * 1.02])
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].set_title(f'Total Islands')
    
    plt.tight_layout()
    
    # save
    nozero_string = '_nozero' if remove_zerocnt else ''
    fig.savefig(outputdir+f'/plots/interisland_distances_{nozero_string}.pdf', dpi=150)
    plt.show(); plt.close()
    
def plot_damaged_area(data_all, outputdir):
    """
    Plot the total damaged area for each condition.
    Uses cm² when converted areas are available; otherwise uses pixels.
    
    (This function was generated by ChatGPT Codex 5.3, and it seems a bit
    overly complex; TODO: take a look at this later.)    
    """

    os.makedirs(outputdir + '/plots/', exist_ok=True)

    # Determine whether converted cm² values are available (at least one finite value)
    cm2_available = False
    if 'total_damage_area_cm2' in data_all:
        all_cm2_values = [
            val
            for cond, val_list in data_all['total_damage_area_cm2'].items()
            for val in val_list
        ]
        cm2_available = np.any(pd.notna(all_cm2_values))

    if cm2_available:
        metric_key = 'total_damage_area_cm2'
        y_label = 'Damaged area (cm²)'
        file_suffix = 'cm2'
    else:
        metric_key = 'total_damage_area_px'
        y_label = 'Damaged area (pixels)'
        file_suffix = 'px'

    # Build dataframe for plotting
    df_area = pd.DataFrame({'cond': [], 'damaged_area': []})
    for cond in data_all[metric_key].keys():
        df_area = pd.concat([
            df_area,
            pd.DataFrame({
                'cond': cond,
                'damaged_area': data_all[metric_key][cond]
            })
        ])

    # Keep only valid numeric values for plotting
    df_area = df_area.reset_index(drop=True)
    df_area = df_area[pd.notna(df_area['damaged_area'])]

    if df_area.empty:
        print('WARNING: No valid damaged-area values available for plotting.')
        return

    fig, ax = plt.subplots(1, 1, figsize=(8 * cm_to_inch, 8 * cm_to_inch))

    sns.barplot(x='cond', y='damaged_area', data=df_area, ax=ax, palette=['blue', 'red'])
    sns.violinplot(x='cond', y='damaged_area', data=df_area, ax=ax, color='black', alpha=0.2)
    sns.stripplot(x='cond', y='damaged_area', data=df_area, ax=ax, color='black')

    ax.set_title('Total Damaged Area')
    ax.set_ylabel(y_label)
    ax.tick_params(axis='x', rotation=45)

    ymax = np.max(df_area['damaged_area'])
    if ymax > 0:
        ax.set_ylim([0, ymax * 1.02])

    plt.tight_layout()
    fig.savefig(outputdir + f'/plots/damaged_area_{file_suffix}.pdf', dpi=150)
    plt.show(); plt.close()
    
# plot the radial distribution functions similar to the acf above
# for all samples, in one panel, colored by condition
def plot_radial_pdfs(data_all, outputdir):
    """
    Plot the radial PDFs for each condition.
    """
    
    os.makedirs(outputdir+'/plots/', exist_ok=True)
    
    fig, axs = plt.subplots(2,1,figsize=(10*cm_to_inch, 10*cm_to_inch))
    
    mycolors = ['blue', 'red']
    
    # loop over keys to get conditions
    for idx, condition in enumerate(data_all['radial_pdfs'].keys()):
        # loop over the different radial_pdfs for each condition
        for radial_pdf in data_all['radial_pdfs'][condition]:
            if radial_pdf is None:
                continue
            axs[0].plot(radial_pdf, color=mycolors[idx], alpha=1, linewidth=.2)
        
    # now in black, add average line per condition
    mylinestyles=['-',':']
    for idx, condition in enumerate(data_all['radial_pdfs'].keys()):
        valid_radial_pdfs = [x for x in data_all['radial_pdfs'][condition] if x is not None]
        if len(valid_radial_pdfs) == 0:
            continue
        # calculate mean, using df since that handles different lengths well
        radial_pdf_avg = pd.DataFrame(valid_radial_pdfs).mean()
        axs[1].plot(radial_pdf_avg, color=mycolors[idx], linewidth=2, label=f'Avg {condition}',
                linestyle=mylinestyles[idx])
    
    axs[0].set_xlabel('Radius (pixels)')
    axs[0].set_ylabel('Radial PDF')
        
    axs[1].set_xlabel('Radius (pixels)')
    axs[1].set_ylabel('Radial PDF')
    axs[1].legend()
    
    plt.tight_layout()
    
    # save as pdf to outputdir
    plt.savefig(outputdir+'/plots/radial_pdfs.pdf', dpi=150)
    
    plt.show(); plt.close()

# %%

# now create a copy of "plot_images()", which
# can be used in a loop over each of the datafiles, to create
# a plot of the images masks etc, and store 
# the plot in outputdir + 'plots/', saved in subdirectories
# according to the original directory structure 

def plot_and_save_images(
    img_leaf,
    img_dmg,
    mask_leaf,
    mask_damage,
    leaf_channel_spec,
    damage_channel_spec,
    leaf_roundness=None,
    total_damage_area_px=None,
    total_damage_area_cm2=None,
    centroid_leaf=None,
    img0=None,
    reference_channel_spec=None,
    filename_suffix='',
    file_path=None,
    outputdir=None
):
    """
    Plots the images and masks, and saves the figure to outputdir/plots/ preserving subdirectory structure.
    file_path: original file path of the image (used to reconstruct subdirectory structure)
    outputdir: base output directory where plots/ will be created
    """
    zm = get_zoombox(mask_leaf, margin=10)
    fig, axs = plt.subplots(1, 3, figsize=(17.2*cm_to_inch, 5*cm_to_inch))
    # set global font size to 8 pts
    plt.rcParams.update({'font.size': 6})
    
    if img0 is not None and reference_channel_spec is not None:
        ref_idx = reference_channel_spec['channel']
        ref_name = reference_channel_spec['name']
        axs[0].imshow(img0[:, :, ref_idx][zm[0]:zm[1],zm[2]:zm[3]])
        axs[0].set_title(f'{ref_name}\nch={ref_idx}')
    else:
        axs[0].axis('off')
    
    leaf_idx = leaf_channel_spec['channel']
    leaf_name = leaf_channel_spec['name']
    if leaf_roundness is None or (isinstance(leaf_roundness, float) and np.isnan(leaf_roundness)):
        leaf_roundness_text = 'roundness=NA'
    else:
        leaf_roundness_text = f'roundness={leaf_roundness:.3f}'

    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]])
    axs[1].set_title(f'{leaf_name}\nch={leaf_idx}\n{leaf_roundness_text}')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    if centroid_leaf is not None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    damage_idx = damage_channel_spec['channel']
    damage_name = damage_channel_spec['name']
    if total_damage_area_px is None or (isinstance(total_damage_area_px, float) and np.isnan(total_damage_area_px)):
        damage_area_text = 'area=NA'
    elif total_damage_area_cm2 is None or (isinstance(total_damage_area_cm2, float) and np.isnan(total_damage_area_cm2)):
        damage_area_text = f'area={total_damage_area_px:.0f} px'
    else:
        damage_area_text = f'area={total_damage_area_px:.0f} px ({total_damage_area_cm2:.4f} cm²)'

    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]])
    axs[2].set_title(f'{damage_name}\nch={damage_idx}\n{damage_area_text}')
    axs[2].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    
    plt.tight_layout()

    # Save figure if file_path and outputdir are provided
    if file_path is not None and outputdir is not None:
        # Get relative path after the data root (e.g., after 'Infected/' or 'Non infected/')
        rel_path = os.path.relpath(file_path, start=os.path.commonpath([file_path, outputdir]))
        # Remove file extension, add optional suffix, and replace with .png
        rel_base = os.path.splitext(rel_path)[0]
        rel_path_noext = rel_base + filename_suffix + '.png'
        # Compose output path
        save_path = os.path.join(outputdir, 'plots', rel_path_noext)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close(fig)

# now write a loop as described above
def run_plot_and_save(
    data_all,
    data_file_paths,
    outputdir,
    leaf_channel_spec,
    damage_channel_spec,
    reference_channel_spec
):
    """
    Run the plot_and_save_images function for each image in data_all.
    Saves the plots in outputdir/plots/ preserving subdirectory structure.
    """
    
    for condition, img_leafs in data_all['img_leafs'].items():
        for idx, img_leaf in enumerate(img_leafs):
            # Add suffix if no leaf was found, but still plot whatever data is available.
            filename_suffix = ''
            if 'leaf_found' in data_all and not data_all['leaf_found'][condition][idx]:
                filename_suffix = '_NOLEAF'
                print("PLOTTING WITH NO LEAF MASK FOR: ", data_file_paths[condition][idx])

            img_rgb = data_all['img_rgbs'][condition][idx]
            img_dmg = data_all['img_damages'][condition][idx]
            mask_leaf = data_all['mask_leafs'][condition][idx]
            mask_damage = data_all['mask_damages'][condition][idx]
            leaf_roundness = data_all['leaf_roundness'][condition][idx] if 'leaf_roundness' in data_all else None
            damage_area_px = data_all['total_damage_area_px'][condition][idx] if 'total_damage_area_px' in data_all else None
            damage_area_cm2 = data_all['total_damage_area_cm2'][condition][idx] if 'total_damage_area_cm2' in data_all else None
            centroid_leaf = data_all['centroids'][condition][idx]
            file_path = data_file_paths[condition][idx]  # original file path
            
            plot_and_save_images(
                img_leaf,
                img_dmg,
                mask_leaf,
                mask_damage,
                leaf_channel_spec,
                damage_channel_spec,
                leaf_roundness,
                damage_area_px,
                damage_area_cm2,
                centroid_leaf,
                img0=img_rgb,
                reference_channel_spec=reference_channel_spec,
                filename_suffix=filename_suffix,
                file_path=file_path,
                outputdir=outputdir
            )

# %% ################################################################################
# Export some data

def export_singledatapoints(data_all, data_file_paths, data_singledatapoint=['total_interisland_distances', 'island_counts', 'leaf_roundness', 'total_damage_area_px', 'total_damage_area_cm2']):
    '''
    For metrics quantified as a single parameter, store those
    in a single dataframe. 
    Also include file paths as column.
    '''
    # data_singledatapoint=['total_interisland_distances', 'island_counts', 'leaf_roundness', 'total_damage_area_px', 'total_damage_area_cm2']
    
    # Set up dataframe with condition and filename first
    cond_fp = [[cond, fp] for cond, fp_list in data_file_paths.items() for fp in fp_list]
    df_singledata = pd.DataFrame(cond_fp, columns=['condition','file_path'])

    # Add mask-detection status columns when available.
    if 'leaf_found' in data_all:
        df_singledata['leaf_found'] = [
            val for cond, val_list in data_all['leaf_found'].items() for val in val_list
        ]
    if 'damage_found' in data_all:
        df_singledata['damage_found'] = [
            val for cond, val_list in data_all['damage_found'].items() for val in val_list
        ]
    if 'analysis_status' in data_all:
        df_singledata['analysis_status'] = [
            val for cond, val_list in data_all['analysis_status'].items() for val in val_list
        ]
    
    # Now add metrics that were calculated before
    # (This assumes these are in the same order!!)
    for metric in data_singledatapoint:
        # Gather both value and conditions
        data_cond = np.array([[val, cond] for cond, val_list in data_all[metric].items() for val in val_list])
        # Double check that conditions match
        if df_singledata['condition'].tolist() == data_cond[:,1].tolist():
            # if so, add data
            df_singledata[metric] = data_cond[:,0].astype(float)
        else:
            # else raise error
            raise ValueError(f'Conditions do not match for metric {metric}!')
    
    return df_singledata
        
        
        
# Also create two example plots using the dataframe       
def simplebarplotseaborn(df_singledata):
    '''
    Example code how to make simple dataplot
    '''    
    
    # plot with condition vs. island_count
    sns.violinplot(x='condition', y='island_counts', data=df_singledata)
    sns.stripplot(x='condition', y='island_counts', data=df_singledata, color='black')
    sns.barplot(x='condition', y='island_counts', data=df_singledata, alpha=0.5, color='grey')
    plt.ylim([0,None])
    plt.show()
    
    # same for interisland distances
    sns.violinplot(x='condition', y='total_interisland_distances', data=df_singledata)
    sns.stripplot(x='condition', y='total_interisland_distances', data=df_singledata, color='black')
    sns.barplot(x='condition', y='total_interisland_distances', data=df_singledata, alpha=0.5, color='grey')
    plt.ylim([0,None])
    plt.show()
    
    
    
    
        


# %%

if __name__ == "__main__":

    # PART A, synthetic data
    
    # 1) Ensure base output directory exists
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # 2) Define channel configuration (index + display name)
    leaf_channel_spec = {'channel': 1, 'name': 'Leaf'}
    damage_channel_spec = {'channel': 2, 'name': 'Damage'}
    reference_channel_spec = {'channel': 0, 'name': 'Reference'} # can be set to None

    # 3) Load synthetic example images and run synthetic sanity-check analysis/plots
    img_leafs_syn, img_damages_syn, img_disk = load_synthetic_data(
        SYNTHETIC_IMAGE_PATH,
        leaf_channel_spec,
        damage_channel_spec
    )
    run_synthetic_analysis(
        img_leafs_syn,
        img_damages_syn,
        img_disk,
        leaf_channel_spec,
        damage_channel_spec,
        reference_channel_spec
    )

    # PART B, real data

    # 1) Tell script where data is and which channels should be used
    # Conditions and paths to images for that condition
    condition_path_map = {
        'infected': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Infected',
        'noninfected': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Non infected'
    }
    # Channel configuration
    leaf_channel_spec = {'channel': 1, 'name': 'Leaf'}
    damage_channel_spec = {'channel': 2, 'name': 'Damage'}
    reference_channel_spec = {'channel': 0, 'name': '(Not used)'} # can be set to None
    # Optional conversion from pixel area to cm^2 (set to e.g. 0.0004 if known)
    pixel_to_cm2_factor = None
    # obtain 
    data_file_paths = get_data_file_paths(condition_path_map)

    # 2) Run the complete analysis pipeline
    data_all = run_complete_analysis(
        data_file_paths,
        leaf_channel_spec,
        damage_channel_spec,
        pixel_to_cm2_factor=pixel_to_cm2_factor
    )

    # 3) Generate summary plots for radial ACF, inter-island distances, and radial PDFs
    plot_acf_norms_avgrs(data_all, OUTPUTDIR)
    plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=False)
    plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=True)
    plot_radial_pdfs(data_all, OUTPUTDIR)

    # 4) Export per-image mask overlays to output folders
    run_plot_and_save(
        data_all,
        data_file_paths,
        OUTPUTDIR,
        leaf_channel_spec,
        damage_channel_spec,
        reference_channel_spec
    )

    # 5) Export single-value metrics to CSV and Excel
    df_singledata = export_singledatapoints(
        data_all,
        data_file_paths,
        data_singledatapoint=['total_interisland_distances', 'island_counts', 'leaf_roundness', 'total_damage_area_px', 'total_damage_area_cm2']
    )
    df_singledata.to_csv(OUTPUTDIR + '/leaf_damage_singlemetrics.csv', index=False)
    df_singledata.to_excel(OUTPUTDIR + '/leaf_damage_singlemetrics.xlsx', index=False)

    # 8) Optional: inspect dataframe plots interactively
    # simplebarplotseaborn(df_singledata)