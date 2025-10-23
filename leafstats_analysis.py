

#%% ################################################################################
# Set these parameters according to file locations on local computer

# Where to put plots
OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/OUTPUT/'
# Where to find synthetic images
SYNTHETIC_IMAGE_PATH = '/Users/m.wehrens/Documents/git_repos/_UVA/_bioDSC-smallprojects/2025_MW_EatingDispersion/Synthetic_data/'

#%% ################################################################################

from PIL import Image
import numpy as np
import pandas as pd 

import math

from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops

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
# Functions

def get_largest_mask(img, method='bg10'):
    """
    Finds Otsu threshold for image, applies threshold, and
    then selects the largest continuous region.
    Returns that region as binary mask.
    
    mask_user can be used to ignore parts of the input image (img)
    """
    
    if method == 'otsu':
        threshold_val = threshold_otsu(img)
    elif method == 'bg10':
        # using percentile
        # threshold_val = 10*np.percentile(img.ravel(), 3)
        # determine mode
        threshold_val = 10 * np.bincount(img.ravel()).argmax()
        
    img_mask = img > threshold_val
    img_lbl = label(img_mask)
    
    lbl_largest = np.argmax([region.area for region in regionprops(img_lbl)]) + 1
    
    img_mask = img_lbl == lbl_largest
    
    return img_mask

def get_mask(img, mask_user=None, method='otsu'):
    
    if mask_user is None:
        mask_user = np.ones(img.shape, dtype=bool)
        
    if method == 'otsu':
        threshold_val = threshold_otsu(img[mask_user])
    elif method == 'triangle':        
        threshold_val = threshold_triangle(img[mask_user])
    elif method == 'bg2':        
        threshold_val = 2 * np.bincount(img[mask_user].ravel()).argmax()
    elif method == 'pct10':
        threshold_val = 10 * np.percentile(img[mask_user], 10)
        
    img_mask = img > threshold_val
    
    return img_mask
      
   
def get_zoombox(mask, margin=0):
    '''
    return coordinates "zoom" to be able
    to zoom on image like img[z1:z2, z3:z4]
    based on mask
    '''
    
    # get bbox
    thebbox = regionprops(mask.astype(int))[0].bbox
    
    # add margin on all sides, taking original mask size into account
    z1 = max(0, thebbox[0] - margin)
    z2 = min(mask.shape[0], thebbox[2] + margin)
    z3 = max(0, thebbox[1] - margin)    
    z4 = min(mask.shape[1], thebbox[3] + margin)
    
    return [z1, z2, z3, z4]
    
    
def plot_images(img_leaf, img_dmg, mask_leaf, mask_damage, centroid_leaf=None, img0=None):
    
    zm = get_zoombox(mask_leaf, margin=10)
    
    fig, axs = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    
    if not img0 is None:
        axs[0].imshow(img0[:,:,0][zm[0]:zm[1],zm[2]:zm[3]]); axs[0].set_title('Red channel')
    else:
        axs[0].axis('off')
    
    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]]); axs[1].set_title('Green channel (leaf)')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)    
    if not centroid_leaf is None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]]); axs[2].set_title('Blue channel (damage)')
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
    acf_norm = acf / np.max(acf)
    
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

img_leafs = {}
img_damages = {}

# Load the leaf w/ eaten disk
img_disk_path = SYNTHETIC_IMAGE_PATH + 'synthetic_eatendisk.tif'
img_disk = io.imread(img_disk_path) # io.read required for img stack
img_leafs['disk'] = img_disk[:,:,1]  # green channel (leaf)
img_damages['disk'] = img_disk[:,:,2]  # blue channel (damage)

# Load the leaf w/ eaten spots
img_spots_damage_path = SYNTHETIC_IMAGE_PATH + 'synthetic_eatenspots.tif'
img_spots_damage = io.imread(img_spots_damage_path) # io.read required for img stack
img_leafs['spots']   = img_spots_damage[:,:,1]  # green channel (leaf)
img_damages['spots'] = img_spots_damage[:,:,2]  # blue channel (damage

# Load the image w/ eaten donut
img_donut_path = SYNTHETIC_IMAGE_PATH + 'synthetic_eatendonut.tif'
img_donut = io.imread(img_donut_path) # io.read required for img stack
img_leafs['donut']   = img_donut[:,:,1]  # green channel (leaf)
img_damages['donut'] = img_donut[:,:,2]  # blue channel (damage

img_donut_path = SYNTHETIC_IMAGE_PATH + 'synthetic_dualspot.tif'
img_donut = io.imread(img_donut_path) # io.read required for img stack
img_leafs['dualspot']   = img_donut[:,:,1]  # green channel (leaf)
img_damages['dualspot'] = img_donut[:,:,2]  # blue channel (damage


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
mask_leafs = {}; mask_damages = {}; centroids ={}
for key in img_leafs.keys():
    mask_leafs[key] = get_largest_mask(img_leafs[key], method='otsu')
    mask_damages[key] = get_mask(img_damages[key], mask_leafs[key], method='bg2') # bg2, otsu, triangle, pct10
    centroids[key] = regionprops(mask_leafs[key].astype(int))[0].centroid

for key in img_leafs.keys():
    plot_images(img_leafs[key], img_damages[key], mask_leafs[key], mask_damages[key], centroids[key], img0=img_disk)

# now get the acf for all
acfs = {}; acf_norms = {}; acf_centers={}; acf_norms_avgrs={}
for key in img_leafs.keys():
    acfs[key], acf_norms[key], acf_centers[key] = get_autocorrelation(img_damages[key], mask_user=mask_leafs[key])
    _, _, acf_norms_avgrs[key], _, _ = get_radial_pdf(acf_norms[key], acf_centers[key])

# now plot 
for key in img_leafs.keys():    
    # key = list(img_leafs.keys())[0]
    plot_img_n_acf(img_damages[key], acf_norms[key], acf_centers[key], acf_norms_avgrs[key], key)
    

# Loop over and get the radial distribution functions
radial_pdf = {}
for key in img_leafs.keys():
    # key = list(img_leafs.keys())[0]
    # radial_count_msk, radial_sum_msk, radial_avg_msk, radial_pdf_msk, r_max_msk = \
    _, _, _, radial_pdf[key], _ = \
        get_radial_pdf(mask_damages[key], centroids[key], mask_leafs[key])

for key in img_leafs.keys():
    
    fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))    
    
    axs[0].imshow(mask_damages[key])
    
    axs[1].plot(radial_pdf[key])
    
    plt.show(); plt.close()

# now also calculate the sum of the inter-island distances for each
total_interisland_distances = {}
for key in img_leafs.keys():
    
    interisland_distances = get_inter_island_distances(mask_leafs[key], mask_damages[key])    
    total_interisland_distances[key] = np.sum(interisland_distances)
    
# now plot
plt.bar(list(img_leafs.keys()), list(total_interisland_distances.values()))

# now also calculate and count the amount of islands
island_counts = {}
for key in img_leafs.keys():
    island_counts[key] = get_island_counts(mask_leafs[key], mask_damages[key])
    
plt.bar(list(island_counts.keys()), list(island_counts.values()))

#%% ######################################################################
# Now let's get real data working

# get all files from the "infected" condition
data_file_paths = {}
data_file_paths['infected'] = \
    glob.glob('/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Infected/*.tif')
data_file_paths['noninfected'] = \
    glob.glob('/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Non infected/*.tif')


def run_complete_analysis(data_file_paths):
    """
    Run all analyses (as for synthetic data) for all files in data_file_paths.
    Stores results in dicts for easy plotting and further analysis.
    """

    # Prepare output structures
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

    for condition, file_list in data_file_paths.items():
        # condition, file_list = list(data_file_paths.items())[0]
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

        for file_path in file_list:
            # file_path = file_list[0]
            
            # Update user on what's happening
            print(f'Processing {file_path} for condition: {condition}')
            
            img = np.array(Image.open(file_path))
            img_leaf = img[:, :, 1]
            img_damage = img[:, :, 2]

            mask_leaf = get_largest_mask(img_leaf, method='bg10')
            mask_damage = get_mask(img_damage, mask_leaf, method='bg2')
            centroid = regionprops(mask_leaf.astype(int))[0].centroid

            acf, acf_norm, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)
            _, _, acf_norm_avgr, _, _ = get_radial_pdf(acf_norm, acf_center)
            _, _, _, radial_pdf, _ = get_radial_pdf(mask_damage, centroid, mask_leaf)
            interisland_distances = get_inter_island_distances(mask_leaf, mask_damage)
            total_interisland = np.sum(interisland_distances)
            island_count = get_island_counts(mask_leaf, mask_damage)

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

    # Return all results as a dictionary of dictionaries/lists
    return {
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
        'island_counts': island_counts
    }

data_all = run_complete_analysis(data_file_paths)

# %% ########################################################################

# Generate a plot of the acf_norms_avgrs, all in the same panel, and 
# annotated per condition
def plot_acf_norms_avgrs(data_all, outputdir):
    """
    Plot the average radial autocorrelation for each condition.
    """
    
    fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 10*cm_to_inch))
    
    mycolors = ['blue', 'red']
    
    # loop over keys to get conditions
    for idx, condition in enumerate(data_all['acf_norms_avgrs'].keys()):
        
        # loop over the different acf_norms_avgrs for each condition
        for acf_norms_avgr in data_all['acf_norms_avgrs'][condition]:
            
            axs[0].plot(acf_norms_avgr, color=mycolors[idx], linewidth=.5)
    
    mylinestyles = ['-',':']
    for idx, condition in enumerate(data_all['acf_norms_avgrs'].keys()):

        # determine the average line per condition, using
        # df like done below
        acf_norms_avgr_avg = pd.DataFrame(data_all['acf_norms_avgrs'][condition]).mean()        
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
    
plot_acf_norms_avgrs(data_all, OUTPUTDIR)    

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


plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=False)
plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=True)

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
            axs[0].plot(radial_pdf, color=mycolors[idx], alpha=1, linewidth=.2)
        
    # now in black, add average line per condition
    mylinestyles=['-',':']
    for idx, condition in enumerate(data_all['radial_pdfs'].keys()):
        # calculate mean, using df since that handles different lengths well
        radial_pdf_avg = pd.DataFrame(data_all['radial_pdfs'][condition]).mean()
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
    
plot_radial_pdfs(data_all, OUTPUTDIR)

# %%

# now create a copy of "plot_images()", which
# can be used in a loop over each of the datafiles, to create
# a plot of the images masks etc, and store 
# the plot in outputdir + 'plots/', saved in subdirectories
# according to the original directory structure 

def plot_and_save_images(img_leaf, img_dmg, mask_leaf, mask_damage, centroid_leaf=None, img0=None, 
                         file_path=None, outputdir=None):
    """
    Plots the images and masks, and saves the figure to outputdir/plots/ preserving subdirectory structure.
    file_path: original file path of the image (used to reconstruct subdirectory structure)
    outputdir: base output directory where plots/ will be created
    """
    zm = get_zoombox(mask_leaf, margin=10)
    fig, axs = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    
    if img0 is not None:
        axs[0].imshow(img0[:,:,0][zm[0]:zm[1],zm[2]:zm[3]])
        axs[0].set_title('Red channel')
    else:
        axs[0].axis('off')
    
    axs[1].imshow(img_leaf[zm[0]:zm[1],zm[2]:zm[3]])
    axs[1].set_title('Green channel (leaf)')
    axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    if centroid_leaf is not None:
        axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
            
    axs[2].imshow(img_dmg[zm[0]:zm[1],zm[2]:zm[3]])
    axs[2].set_title('Blue channel (damage)')
    axs[2].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
    
    plt.tight_layout()

    # Save figure if file_path and outputdir are provided
    if file_path is not None and outputdir is not None:
        # Get relative path after the data root (e.g., after 'Infected/' or 'Non infected/')
        rel_path = os.path.relpath(file_path, start=os.path.commonpath([file_path, outputdir]))
        # Remove file extension and replace with .png
        rel_path_noext = os.path.splitext(rel_path)[0] + '.png'
        # Compose output path
        save_path = os.path.join(outputdir, 'plots', rel_path_noext)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.close(fig)

# now write a loop as described above
def run_plot_and_save(data_all, outputdir):
    """
    Run the plot_and_save_images function for each image in data_all.
    Saves the plots in outputdir/plots/ preserving subdirectory structure.
    """
    
    for condition, img_leafs in data_all['img_leafs'].items():
        for idx, img_leaf in enumerate(img_leafs):
            img_dmg = data_all['img_damages'][condition][idx]
            mask_leaf = data_all['mask_leafs'][condition][idx]
            mask_damage = data_all['mask_damages'][condition][idx]
            centroid_leaf = data_all['centroids'][condition][idx]
            file_path = data_file_paths[condition][idx]  # original file path
            
            plot_and_save_images(img_leaf, img_dmg, mask_leaf, mask_damage, 
                                 centroid_leaf, img0=img_disk,
                                 file_path=file_path, outputdir=outputdir)

run_plot_and_save(data_all, OUTPUTDIR)

# %% ################################################################################
# Export some data

def export_singledatapoints(data_all, data_file_paths, data_singledatapoint=['total_interisland_distances', 'island_counts']):
    '''
    For metrics quantified as a single parameter, store those
    in a single dataframe. 
    Also include file paths as column.
    '''
    # data_singledatapoint=['total_interisland_distances', 'island_counts']
    
    # Set up dataframe with condition and filename first
    cond_fp = [[cond, fp] for cond, fp_list in data_file_paths.items() for fp in fp_list]
    df_singledata = pd.DataFrame(cond_fp, columns=['condition','file_path'])
    
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
        
# now generate and export to dataframe metrics with single data points
df_singledata = export_singledatapoints(data_all, data_file_paths,
                                        data_singledatapoint=['total_interisland_distances', 'island_counts'])
df_singledata.to_csv(OUTPUTDIR+'/leaf_damage_singlemetrics.csv', index=False)

        
        
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
