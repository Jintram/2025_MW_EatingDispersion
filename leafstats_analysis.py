

#%% ################################################################################
# Set these parameters according to file locations on local computer


#%% ################################################################################

from PIL import Image
import numpy as np
import pandas as pd 

import math

from skimage import io
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
import warnings

cm_to_inch = 1/2.54
# set plotting params
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8
})

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
        # determine mode (= background value) and set threshold to 10x
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
        # perform morphological closing with a radius of 10 pixels to smooth the mask
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
        # determine mode 
        the_mode = np.bincount(img[mask_user].ravel()).argmax()
        # deal with issue: oversaturation and mode = max val
        # NOTE: I'm using max val since there are images were max was not equal 
        # to the max of the range (254 instead of 255 -- potential acquisition artifact)
        if the_mode == np.max(img):
            the_mode = np.bincount(img[mask_user][img[mask_user]<np.max(img)].ravel()).argmax()
            # raise warning
            warnings.warn("Mode value equals image maximum value, likely due to saturation. Using second mode instead.")
        # set the threshold
        threshold_val = 2 * the_mode        
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
    based on outer edges of "mask"
    returns coordinates "zoom" to be able
    to zoom on image like img[z1:z2, z3:z4]
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
    config_channels,
    centroid_leaf=None,
    img0=None
):
    """
    Plots three channels side by side.
    config_channels: dict with keys 'Leaf', 'Damage', and optional 'Reference' (value may be None).
    """
    
    zm = get_zoombox(mask_leaf, margin=10)
    
    fig, axs = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    
    # Reference channel (skipped if not present)
    if img0 is not None and config_channels.get('Reference') is not None:
        axs[0].imshow(img0[:, :, config_channels.get('Reference')][zm[0]:zm[1],zm[2]:zm[3]]); axs[0].set_title(f'Reference channel (idx={config_channels.get("Reference")})')
    else:
        axs[0].axis('off')
    
    # Leaf channel
    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]]); axs[1].set_title(f'Leaf channel (idx={config_channels['Leaf']}, leaf)')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)    
    if centroid_leaf is not None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    # Damage channel    
    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]]); axs[2].set_title(f'Damage channel (idx={config_channels['Damage']}, damage)')
    axs[2].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)

    plt.tight_layout()
        
    return fig, axs
    
        

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


def load_synthetic_data(synthetic_image_path, config_channels):
    """
    Load synthetic TIFF stacks and split them into leaf and damage channels.
    config_channels: dict with at least 'Leaf' and 'Damage' keys mapping to channel indices.
    """

    img_leafs = {}
    img_damages = {}
    
    # Load the leaf w/ eaten disk
    img_disk_path = synthetic_image_path + 'synthetic_eatendisk.tif'
    img_disk = io.imread(img_disk_path)  # io.read required for img stack
    img_leafs['disk'] = img_disk[:, :, config_channels['Leaf']]  # configured leaf channel
    img_damages['disk'] = img_disk[:, :, config_channels['Damage']]  # configured damage channel

    # Load the leaf w/ eaten spots
    img_spots_damage_path = synthetic_image_path + 'synthetic_eatenspots.tif'
    img_spots_damage = io.imread(img_spots_damage_path)  # io.read required for img stack
    img_leafs['spots'] = img_spots_damage[:, :, config_channels['Leaf']]  # configured leaf channel
    img_damages['spots'] = img_spots_damage[:, :, config_channels['Damage']]  # configured damage channel

    # Load the image w/ eaten donut
    img_donut_path = synthetic_image_path + 'synthetic_eatendonut.tif'
    img_donut = io.imread(img_donut_path)  # io.read required for img stack
    img_leafs['donut'] = img_donut[:, :, config_channels['Leaf']]  # configured leaf channel
    img_damages['donut'] = img_donut[:, :, config_channels['Damage']]  # configured damage channel

    # Load dual-spot sample
    img_dualspot_path = synthetic_image_path + 'synthetic_dualspot.tif'
    img_dualspot = io.imread(img_dualspot_path)  # io.read required for img stack
    img_leafs['dualspot'] = img_dualspot[:, :, config_channels['Leaf']]  # configured leaf channel
    img_damages['dualspot'] = img_dualspot[:, :, config_channels['Damage']]  # configured damage channel

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
    
    plt.tight_layout()
    
    return fig, axs

# now get masks for leaf and damage, plus centroid for all 
def run_synthetic_analysis(
    img_leafs,
    img_damages,
    img_disk,
    config_channels,
    outputdir=None
):
    """
    Run synthetic-data diagnostics and plots to verify analysis behavior.
    config_channels: dict with keys 'Leaf', 'Damage', and optional 'Reference'.
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
        fig, axs = plot_images(
            img_leafs[key],
            img_damages[key],
            mask_leafs[key],
            mask_damages[key],
            config_channels,
            centroid_leaf=centroids[key],
            img0=img_disk
        )
        fig.savefig(os.path.join(outputdir, f'synthdata_img_{key}.pdf'), dpi=150)
        fig.savefig(os.path.join(outputdir, f'synthdata_img_{key}.png'), dpi=150)
        plt.close(fig)

    # Plot the damage quantification in a bar plot
    damage_areas_percentage = {key: np.sum(mask_damages[key]) / np.sum(mask_leafs[key]) * 100 for key in img_leafs.keys()}
    fig, axs = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    axs.bar(list(damage_areas_percentage.keys()), list(damage_areas_percentage.values()))
    axs.set_ylabel("Damage area (% of leaf)")
    plt.tight_layout()
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_damage.pdf'), dpi=150)
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_damage.png'), dpi=150)
    
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
        fig, axs = plot_img_n_acf(img_damages[key], acf_norms[key], acf_centers[key], acf_norms_avgrs[key], key)
        fig.savefig(os.path.join(outputdir, f'synthdata_acf_{key}.pdf'), dpi=150)
        fig.savefig(os.path.join(outputdir, f'synthdata_acf_{key}.png'), dpi=150)
        plt.close(fig)

    # Radial PDFs for synthetic masks
    radial_pdf = {}
    for key in img_leafs.keys():
        _, _, _, radial_pdf[key], _ = get_radial_pdf(mask_damages[key], centroids[key], mask_leafs[key])
    # plot radial PDFs
    for key in img_leafs.keys():
        fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))        
        axs[0].imshow(mask_damages[key])
        axs[1].plot(radial_pdf[key])
        axs[1].set_ylabel('Radial distribution function')
        plt.tight_layout();
        fig.savefig(os.path.join(outputdir, f'synthdata_radialpdf_{key}.pdf'), dpi=150)
        fig.savefig(os.path.join(outputdir, f'synthdata_radialpdf_{key}.png'), dpi=150)
        plt.close(fig)

    # Summarize island spacing
    total_interisland_distances = {}
    for key in img_leafs.keys():
        interisland_distances = get_inter_island_distances(mask_leafs[key], mask_damages[key])
        total_interisland_distances[key] = np.sum(interisland_distances)
    # plot island spacing
    fig, axs = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    axs.bar(list(img_leafs.keys()), list(total_interisland_distances.values()))
    axs.set_xticklabels(list(img_leafs.keys()), rotation=45, ha="right")
    axs.set_ylabel("Sum inter-island distances")
    plt.tight_layout()
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_interisland.pdf'), dpi=150)
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_interisland.png'), dpi=150)

    # Summarize island counts
    island_counts = {}
    for key in img_leafs.keys():
        island_counts[key] = get_island_counts(mask_leafs[key], mask_damages[key])
    # plot island counts
    fig, axs = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    axs.bar(list(island_counts.keys()), list(island_counts.values()))
    axs.set_xticklabels(list(island_counts.keys()), rotation=45, ha="right")
    axs.set_ylabel("Island count")
    plt.tight_layout()
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_islandcount.pdf'), dpi=150)
    fig.savefig(os.path.join(outputdir, f'synthdata_summary_islandcount.png'), dpi=150)
    

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

def run_complete_analysis(data_file_paths, config_channels,
                          leaf_threshold_method = 'bg10', leaf_roundness_threshold=0,
                          apply_smooth_leafmask=False,
                          pixel_to_cm2_factor=None):
    """
    Run all analyses (as for synthetic data) for all files in data_file_paths.
    Stores scalar outputs in a dataframe and array-like outputs in a dict.
    config_channels: dict with keys 'Leaf' and 'Damage' mapping to channel indices.
    """

    # Prepare output structures
    rows = []
    array_data = {}

    for condition, file_list in data_file_paths.items():
        for file_path in file_list:
            # file_path = file_list[0]
            # file_path = file_list[7]
            
            # Update user on what's happening
            print(f'Processing {file_path} for condition: {condition}')
            
            # Load images
            img = np.array(Image.open(file_path))
            # in case the image doesn't have 3 dimensions, expand to three
            img = np.atleast_3d(img)
            img_leaf = img[:, :, config_channels['Leaf']]
            img_damage = img[:, :, config_channels['Damage']]

            # Get leaf mask
            mask_leaf, this_leaf_found = \
                get_largest_mask(img_leaf, 
                                 method=leaf_threshold_method, 
                                 apply_smooth=apply_smooth_leafmask,
                                 return_status=True)
                # plt.imshow(img_leaf); plt.contour(mask_leaf, colors='white'); plt.show(); plt.close()

            # Additional check for leaf validity, check roundness
            if this_leaf_found:
                leaf_roundness = determine_leaf_roundness(mask_leaf)
                if not leaf_roundness > leaf_roundness_threshold:
                    this_leaf_found = False
                    print("WARNING: Leaf roundness below threshold, marking as no leaf found.")
            else:
                leaf_roundness = np.nan
                
            # Set up structures to save the data (pre-filled for case data NA)
            row = {
                'condition': condition,
                'file_path': file_path,
                'leaf_found': this_leaf_found,
                'damage_found': False,
                'analysis_status': 'no_leaf_mask',
                'leaf_roundness': leaf_roundness,
                'total_interisland_distances': np.nan,
                'island_counts': np.nan,                
                'total_leaf_size_px': np.nan, 
                'total_leaf_size_cm2': np.nan,
                'total_damage_area_px': np.nan,
                'total_damage_area_cm2': np.nan, 
                'total_damage_percentage': np.nan
            }
            # Storage for arrays
            mask_damage = np.zeros_like(mask_leaf, dtype=bool)
            centroid = None
            acf = None
            acf_norm = None
            acf_center = None
            acf_norm_avgr = None
            radial_pdf = None

            if this_leaf_found:
                
                # store the size
                row['total_leaf_size_px'] = float(np.sum(mask_leaf))
                if pixel_to_cm2_factor is not None:
                    row['total_leaf_size_cm2'] = row['total_leaf_size_px'] * pixel_to_cm2_factor
                                    
                # CASE LEAF FOUND; PERFORM ANALYSIS
                mask_damage, this_damage_found = get_mask(img=img_damage,
                                                          mask_user=mask_leaf, method='bg2', return_status=True)
                centroid = regionprops(mask_leaf.astype(int))[0].centroid
                    # plt.imshow(img_damage); plt.contour(mask_damage, colors='white'); plt.show(); plt.close()
                    # plt.hist(img_damage[mask_leaf].ravel(), bins=256); plt.show(); plt.close()

                row['damage_found'] = this_damage_found

                # If no damage is detected inside leaf, keep valid zeros for damage metrics.
                if not this_damage_found:
                    row['analysis_status'] = 'no_damage_mask'
                    row['total_interisland_distances'] = 0.0
                    row['island_counts'] = 0
                    row['total_damage_area_px'] = 0.0
                    row['total_damage_area_cm2'] = (
                        np.nan if pixel_to_cm2_factor is None else 0.0 * pixel_to_cm2_factor
                    )
                    row['total_damage_percentage'] = 0.0
                else:
                    acf, acf_norm, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)
                    _, _, acf_norm_avgr, _, _ = get_radial_pdf(acf_norm, acf_center)
                    _, _, _, radial_pdf, _ = get_radial_pdf(mask_damage, centroid, mask_leaf)
                    interisland_distances = get_inter_island_distances(mask_leaf, mask_damage)
                    row['analysis_status'] = 'ok'
                    row['total_interisland_distances'] = np.sum(interisland_distances)
                    row['island_counts'] = get_island_counts(mask_leaf, mask_damage)
                    row['total_damage_area_px'] = float(np.sum(mask_damage))
                    row['total_damage_area_cm2'] = (
                        np.nan if pixel_to_cm2_factor is None
                        else row['total_damage_area_px'] * pixel_to_cm2_factor
                    )
                    row['total_damage_percentage'] = (
                        row['total_damage_area_px'] / row['total_leaf_size_px'] * 100
                    )

            rows.append(row)

            array_data[file_path] = {
                'condition': condition,
                'img_rgb': img,
                'img_leaf': img_leaf,
                'img_damage': img_damage,
                'mask_leaf': mask_leaf,
                'mask_damage': mask_damage,
                'centroid': centroid,
                'acf': acf,
                'acf_norm': acf_norm,
                'acf_center': acf_center,
                'acf_norm_avgr': acf_norm_avgr,
                'radial_pdf': radial_pdf
            }

    df_samples = pd.DataFrame(rows)
    return df_samples, array_data

# %% ########################################################################

# Generate a plot of the acf_norms_avgrs, all in the same panel, and 
# annotated per condition
def plot_acf_norms_avgrs(df_samples, array_data, outputdir, mycolors = None):
    """
    Plot the average radial autocorrelation for each condition.
    """
    
    os.makedirs(outputdir+'/plots/', exist_ok=True)   
    
    if mycolors is None:
        sns.color_palette('colorblind')    
    
    # convert acf_norm_avgr data in array_data to one big dataframe (long format)
    acf_df_rows = []
    # loop over all samples
    for _, row in df_samples.iterrows():
        # retrieve the acf
        acf_norm_avgr = array_data[row['file_path']]['acf_norm_avgr']
        # add data to df (radius can be 1:N, as pixel-based)
        if acf_norm_avgr is not None:
            for radius, value in enumerate(acf_norm_avgr):
                acf_df_rows.append({
                    'condition': row['condition'],
                    'radius': radius,
                    'acf_norm_avgr': value,
                    'file_path': row['file_path']
                })
    df_acf = pd.DataFrame(acf_df_rows)
    
    # plotting          
    fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 10*cm_to_inch))
                    
    # now plot each sample, colored by condition
    sns.lineplot(
        x='radius', y='acf_norm_avgr', hue='condition',         
        units = 'file_path', estimator=None, # plot each sample separately
        data=df_acf, 
        ax=axs[0], palette=mycolors, linewidth=0.5, legend=False)
    
    # now plot averages per condition
    sns.lineplot(
        x='radius', y='acf_norm_avgr', hue='condition',
        errorbar=None, 
        data=df_acf,
        ax=axs[1], palette=mycolors, linewidth=2)
    
    fig.suptitle('Radial Autocorrelation')    
    axs[0].set_xlabel('Radius (pixels)')
    axs[0].set_ylabel('Normalized Autocorrelation')
    axs[0].set_title('Per sample')
    
    axs[1].set_xlabel('Radius (pixels)')
    axs[1].set_ylabel('Normalized Autocorrelation')
    axs[0].set_title('Condition averages')
    
    plt.tight_layout()
    plt.savefig(outputdir+'/plots/Radial_acf.pdf', dpi=150)
    plt.savefig(outputdir+'/plots/Radial_acf.png', dpi=150)
    
    axs[0].set_xlim([0,200]); axs[1].set_xlim([0,200])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(outputdir+'/plots/Radial_acf_lims.pdf', dpi=150)
    plt.savefig(outputdir+'/plots/Radial_acf_lims.png', dpi=150)
        
    plt.show(); plt.close()
    
# Now the same for the inter-island distance metric
def plot_interisland_distances(df_samples, outputdir, remove_zerocnt=True, mycolors=None):
    """
    Plot the total inter-island distances for each condition.
    """    
    
    if mycolors is None:
        sns.color_palette('colorblind')
    
    os.makedirs(outputdir+'/plots/', exist_ok=True)

    # Drop rows with missing metrics; optionally drop zero-island samples.
    df_plot = df_samples[['condition', 'total_interisland_distances', 'island_counts']].dropna(
        subset=['total_interisland_distances', 'island_counts']
    )
    if remove_zerocnt:
        df_plot = df_plot[df_plot['island_counts'] > 0]

    if df_plot.empty:
        print('WARNING: No valid inter-island values available for plotting.')
        return

    # Plot 
    fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 10*cm_to_inch))
    
    # plot total inter-island distances using strippplot / seaborn
    sns.barplot(x='condition', y='total_interisland_distances', 
                data=df_plot, ax=axs[0], palette=mycolors, hue='condition')
    sns.violinplot(x='condition', y='total_interisland_distances', 
                   data=df_plot, ax=axs[0], color='black', alpha=0.2)
    sns.stripplot(x='condition', y='total_interisland_distances', 
                  data=df_plot, ax=axs[0], color='black')
    
    axs[0].set_title(f'Total Closest-Island\nDistances')
    axs[0].set_ylabel('Distance (pixels)')
    ymax0 = np.nanmax(df_plot['total_interisland_distances'])
    axs[0].set_ylim([0, (ymax0 if ymax0 > 0 else 1) * 1.02])
    # rotate axis 90 deg
    axs[0].tick_params(axis='x', rotation=45)
    
    # now also plot the island counts themselves
    sns.barplot(x='condition', y='island_counts', 
                data=df_plot, ax=axs[1], palette=mycolors, hue='condition')
    sns.violinplot(x='condition', y='island_counts',
                     data=df_plot, ax=axs[1], color='black', alpha=0.2)
    sns.stripplot(x='condition', y='island_counts',
                  data=df_plot, ax=axs[1], color='black')
    ymax1 = np.nanmax(df_plot['island_counts'])
    axs[1].set_ylim([0, (ymax1 if ymax1 > 0 else 1) * 1.02])
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].set_title(f'Total Islands')
    
    plt.tight_layout()
    
    # save
    nozero_string = '_nozero' if remove_zerocnt else ''
    fig.savefig(outputdir+f'/plots/interisland_distances_{nozero_string}.pdf', dpi=150)
    fig.savefig(outputdir+f'/plots/interisland_distances_{nozero_string}.png', dpi=150)
    plt.show(); plt.close()
    
def plot_damaged_area(df_samples, outputdir):
    """
    Plot the total damaged area for each condition.
    Uses cm^2 when converted areas are available; otherwise uses pixels.
    
    (This function was generated by ChatGPT Codex 5.3, and it seems a bit
    overly complex; TODO: take a look at this later.)    
    """

    os.makedirs(outputdir + '/plots/', exist_ok=True)

    cm2_available = np.any(pd.notna(df_samples['total_damage_area_cm2']))

    if cm2_available:
        metric_key = 'total_damage_area_cm2'
        y_label = 'Damaged area (cm^2)'
        file_suffix = 'cm2'
    else:
        metric_key = 'total_damage_area_px'
        y_label = 'Damaged area (pixels)'
        file_suffix = 'px'

    df_area = df_samples[['condition', metric_key]].copy()
    df_area = df_area[pd.notna(df_area[metric_key])]

    if df_area.empty:
        print('WARNING: No valid damaged-area values available for plotting.')
        return

    fig, ax = plt.subplots(1, 1, figsize=(8 * cm_to_inch, 8 * cm_to_inch))

    sns.barplot(x='condition', y=metric_key, data=df_area, ax=ax, palette=['blue', 'red'])
    sns.violinplot(x='condition', y=metric_key, data=df_area, ax=ax, color='black', alpha=0.2)
    sns.stripplot(x='condition', y=metric_key, data=df_area, ax=ax, color='black')

    ax.set_title('Total Damaged Area')
    ax.set_ylabel(y_label)
    ax.tick_params(axis='x', rotation=45)

    ymax = np.max(df_area[metric_key])
    if ymax > 0:
        ax.set_ylim([0, ymax * 1.05])

    plt.tight_layout()
    fig.savefig(outputdir + f'/plots/damaged_area_{file_suffix}.pdf', dpi=150)
    fig.savefig(outputdir + f'/plots/damaged_area_{file_suffix}.png', dpi=150)
    plt.show(); plt.close()
    
# plot the radial distribution functions similar to the acf above
# for all samples, in one panel, colored by condition
def plot_radial_pdfs(df_samples, array_data, outputdir, mycolors=None):
    """
    Plot the radial PDFs for each condition.
    """

    os.makedirs(outputdir+'/plots/', exist_ok=True)

    if mycolors is None:
        mycolors = sns.color_palette('colorblind')

    # convert radial_pdf data in array_data to one big dataframe (long format)
    pdf_df_rows = []
    for _, row in df_samples.iterrows():
        radial_pdf = array_data[row['file_path']]['radial_pdf']
        if radial_pdf is None:
            continue
        for radius, value in enumerate(radial_pdf):
            pdf_df_rows.append({
                'condition': row['condition'],
                'radius': radius,
                'radial_pdf': value,
                'file_path': row['file_path']
            })
    df_pdf = pd.DataFrame(pdf_df_rows)

    fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 10*cm_to_inch))

    # plot each sample, colored by condition
    sns.lineplot(
        x='radius', y='radial_pdf', hue='condition',
        units='file_path', estimator=None, # plot each sample separately
        data=df_pdf,
        ax=axs[0], palette=mycolors, linewidth=0.2, legend=False)

    # plot averages per condition
    sns.lineplot(
        x='radius', y='radial_pdf', hue='condition',
        errorbar=None,
        data=df_pdf,
        ax=axs[1], palette=mycolors, linewidth=2)

    axs[0].set_xlabel('Radius (pixels)')
    axs[0].set_ylabel('Radial PDF')
    axs[0].set_title('Per sample')

    axs[1].set_xlabel('Radius (pixels)')
    axs[1].set_ylabel('Radial PDF')
    axs[1].set_title('Condition averages')
    axs[1].legend()

    plt.tight_layout()

    # save as pdf to outputdir
    plt.savefig(outputdir+'/plots/radial_pdfs.pdf', dpi=150)
    plt.savefig(outputdir+'/plots/radial_pdfs.png', dpi=150)

    plt.show(); plt.close()

# %%

def plot_and_save_images(
    img_leaf,
    img_dmg,
    mask_leaf,
    mask_damage,
    config_channels,
    leaf_roundness=None,
    total_damage_area_px=None,
    total_damage_area_cm2=None,
    centroid_leaf=None,
    img0=None,
    filename_suffix='',
    file_path=None,
    outputdir=None
):
    """
    Plots the images and masks, and saves the figure to outputdir/plots/ preserving subdirectory structure.
    config_channels: dict with keys 'Leaf', 'Damage', and optional 'Reference' (value may be None).
    file_path: original file path of the image (used to reconstruct subdirectory structure)
    outputdir: base output directory where plots/ will be created
    """
    zm = get_zoombox(mask_leaf, margin=10)
    fig, axs = plt.subplots(1, 3, figsize=(17.2*cm_to_inch, 5*cm_to_inch))
    # set global font size to 8 pts
    plt.rcParams.update({'font.size': 6})
    
    if img0 is not None and config_channels.get('Reference') is not None:
        axs[0].imshow(img0[:, :, config_channels.get('Reference')][zm[0]:zm[1],zm[2]:zm[3]])
        axs[0].set_title(f'Reference\nch={config_channels.get("Reference")}')
    else:
        axs[0].axis('off')
    
    config_channels['Leaf'] = config_channels['Leaf']
    if leaf_roundness is None or (isinstance(leaf_roundness, float) and np.isnan(leaf_roundness)):
        leaf_roundness_text = 'roundness=NA'
    else:
        leaf_roundness_text = f'roundness={leaf_roundness:.3f}'

    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]])
    axs[1].set_title(f'Leaf\nch={config_channels['Leaf']}\n{leaf_roundness_text}')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    if centroid_leaf is not None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    config_channels['Damage'] = config_channels['Damage']
    if total_damage_area_px is None or (isinstance(total_damage_area_px, float) and np.isnan(total_damage_area_px)):
        damage_area_text = 'area=NA'
    elif total_damage_area_cm2 is None or (isinstance(total_damage_area_cm2, float) and np.isnan(total_damage_area_cm2)):
        damage_area_text = f'area={total_damage_area_px:.0f} px'
    else:
        damage_area_text = f'area={total_damage_area_px:.0f} px ({total_damage_area_cm2:.4f} cm²)'

    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]])
    axs[2].set_title(f'Damage\nch={config_channels['Damage']}\n{damage_area_text}')
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

def run_plot_and_save(
    df_samples,
    array_data,
    outputdir,
    config_channels
):
    """
    Run the plot_and_save_images function for each image in data_all.
    Saves the plots in outputdir/plots/ preserving subdirectory structure.
    config_channels: dict with keys 'Leaf', 'Damage', and optional 'Reference'.
    """
    for _, row in df_samples.iterrows():
        file_path = row['file_path']
        this_arrays = array_data[file_path]

        # Add suffix if no leaf was found, but still plot whatever data is available.
        filename_suffix = ''
        if not row['leaf_found']:
            filename_suffix = '_NOLEAF'
            print("PLOTTING WITH NO LEAF MASK FOR: ", file_path)

        plot_and_save_images(
            this_arrays['img_leaf'],
            this_arrays['img_damage'],
            this_arrays['mask_leaf'],
            this_arrays['mask_damage'],
            config_channels,
            leaf_roundness=row['leaf_roundness'],
            total_damage_area_px=row['total_damage_area_px'],
            total_damage_area_cm2=row['total_damage_area_cm2'],
            centroid_leaf=this_arrays['centroid'],
            img0=this_arrays['img_rgb'],
            filename_suffix=filename_suffix,
            file_path=file_path,
            outputdir=outputdir
        )



# %%

if __name__ == "__main__":

    pass