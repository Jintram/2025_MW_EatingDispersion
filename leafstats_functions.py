

#%% ################################################################################

from PIL import Image
import numpy as np

import math

from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops

from scipy.signal import correlate
    # from scipy.signal import correlate2d # VERY SLOW
from scipy.spatial.distance import pdist

import cv2
import scipy.ndimage as ndi

import matplotlib.pyplot as plt

import time # for debugging/optimization

import glob

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

#%% ################################################################################
# standard analysis function

def standard_analysis(img_leaf, img_damage):
    
    mask_leaf = get_largest_mask(img_leaf, method='otsu')
    mask_damage = get_mask(img_damage, mask_leaf, method='bg2') # bg2
    centroid_leaf = regionprops(mask_leaf.astype(int))[0].centroid

    plot_images(img_leaf, img_damage, mask_leaf, mask_damage, centroid_leaf)

    # now get the acf
    acf_msk, acf_norm_msk, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)

    # plot the acf centerline
    plt.imshow(img_damage)
    plt.show()
    x_axis = np.arange(acf_norm_msk.shape[1]) - acf_center[1]
    plt.plot(x_axis, acf_norm_msk[acf_center[0],:])
    plt.show(); plt.close()

#%% ################################################################################

# Images are RGB images.
# In green, there's the leaf image, and in blue, there's the "damage intensity" (based on infrared profile).

# let's open a test image
img_test_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Non infected/Tomato GFP 6 X5Y1.tif'
img_test_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Non infected/Tomato GFP 8 X3Y3.tif'
img_test = np.array(Image.open(img_test_path))
    # img_test.shape

img_leaf   = img_test[:,:,1]  # green channel (leaf)
img_damage = img_test[:,:,2]  # blue channel (damage)

# let's threshold the leaf using Otsu
mask_leaf   = get_largest_mask(img_test[:,:,1])
mask_damage = get_mask(img_test[:,:,2], mask_leaf, method='bg2') # bg2, otsu, triangle, pct10

# get the center of mass of the leaf mask
centroid_leaf = regionprops(mask_leaf.astype(int))[0].centroid

# visual inspeection of leaf damage channel
# channels next to each other
zm = get_zoombox(mask_leaf, margin=10)
fig, axs = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
axs[0].imshow(img_test[:,:,0][zm[0]:zm[1],zm[2]:zm[3]]); axs[0].set_title('Red channel')
axs[1].imshow(img_test[:,:,1][zm[0]:zm[1],zm[2]:zm[3]]); axs[1].set_title('Green channel (leaf)')
axs[1].contour(mask_leaf[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
axs[1].plot(centroid_leaf[1]-zm[2], centroid_leaf[0]-zm[0], 'rx', markersize=15)
# axs[1].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
axs[2].imshow(img_test[:,:,2][zm[0]:zm[1],zm[2]:zm[3]]); axs[2].set_title('Blue channel (damage)')
axs[2].contour(mask_damage[zm[0]:zm[1],zm[2]:zm[3]], colors='white', linewidths=1)
plt.show(); plt.close()

# Now quantify some things
# - histogram of damage
# - plot the center of the leaf, radial distribution
# - apply metrics

radial_count_msk, radial_sum_msk, radial_avg_msk, radial_pdf_msk, r_max_msk = \
    get_radial_pdf(mask_damage, centroid_leaf, mask_leaf)

radial_count_dmg, radial_sum_dmg, radial_avg_dmg, radial_pdf_dmg, r_max_dmg = \
    get_radial_pdf(img_damage, centroid_leaf, mask_leaf)

plt.plot(radial_sum_dmg); 
plt.plot(radial_count_dmg); 
plt.xlim([0,r_max_dmg])
plt.show(); plt.close()

plt.plot(radial_pdf_msk, '-', label='binary damage')
plt.plot(radial_pdf_dmg, '--', label='damage')
plt.legend()
plt.show(); plt.close()

########################################

# get and display the mask
acf_msk, acf_norm_msk = get_autocorrelation(mask_leaf, mask_user=mask_leaf)


plt.imshow(acf_norm_msk, cmap='gray')
plt.imshow(acf_msk, cmap='gray')
plt.show(); plt.close()

# now radially integrate acf_norm_msk using around its center
acf_center = np.round(np.array(acf_norm_msk.shape)/2).astype(int)
acf_norm_msk_radius = get_radial_pdf(acf_norm_msk, acf_center, mask_leaf)[2]
plt.plot(acf_norm_msk_radius)
plt.show(); plt.close()

# get centerline acf
plt.plot(acf_norm_msk[acf_center[0],:])
plt.show(); plt.close()


plt.imshow(img_test)
plt.contour(mask_leaf, colors='white', linewidths=1)
plt.contour(mask_damage, colors='blue', linewidths=1)
plt.show(); plt.close()

# show mask projected on original image
fig, axs = plt.subplots(1, 2, figsize=(15*cm_to_inch, 5*cm_to_inch))
axs[0].imshow(img_test)
axs[1].imshow(mask_damage)
plt.show(); plt.close()


#%% ################################################################################
# Load the synthetic data

# open tiff stack image
from skimage import io

img_leafs = {}
img_damages = {}

# Load the leaf w/ eaten disk
img_disk_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_eatendisk.tif'
img_disk = io.imread(img_disk_path) # io.read required for img stack
img_leafs['disk'] = img_disk[:,:,1]  # green channel (leaf)
img_damages['disk'] = img_disk[:,:,2]  # blue channel (damage)

# Load the leaf w/ eaten spots
img_spots_damage_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_eatenspots.tif'
img_spots_damage = io.imread(img_spots_damage_path) # io.read required for img stack
img_leafs['spots']   = img_spots_damage[:,:,1]  # green channel (leaf)
img_damages['spots'] = img_spots_damage[:,:,2]  # blue channel (damage

# Load the image w/ eaten donut
img_donut_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_eatendonut.tif'
img_donut = io.imread(img_donut_path) # io.read required for img stack
img_leafs['donut']   = img_donut[:,:,1]  # green channel (leaf)
img_damages['donut'] = img_donut[:,:,2]  # blue channel (damage

img_donut_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_dualspot.tif'
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

    for condition, file_list in data_file_paths.items():
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

        for file_path in file_list:
            
            # Update user on what's happening
            print(f'Processing {file_path} for condition: {condition}')
            
            img = np.array(Image.open(file_path))
            img_leaf = img[:, :, 1]
            img_damage = img[:, :, 2]

            mask_leaf = get_largest_mask(img_leaf, method='otsu')
            mask_damage = get_mask(img_damage, mask_leaf, method='bg2')
            centroid = regionprops(mask_leaf.astype(int))[0].centroid

            acf, acf_norm, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)
            _, _, acf_norm_avgr, _, _ = get_radial_pdf(acf_norm, acf_center)
            _, _, _, radial_pdf, _ = get_radial_pdf(mask_damage, centroid, mask_leaf)
            interisland_distances = get_inter_island_distances(mask_leaf, mask_damage)
            total_interisland = np.sum(interisland_distances)

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
        'total_interisland_distances': total_interisland_distances
    }

data_all = run_complete_analysis(data_file_paths)

# %% ########################################################################

# Generate a plot of the acf_norms_avgrs, all in the same panel, and 
# annotated per condition
def plot_acf_norms_avgrs(data_all):
    """
    Plot the average radial autocorrelation for each condition.
    """
    
    fig, ax = plt.subplots(figsize=(10*cm_to_inch, 5*cm_to_inch))
    
    mycolors = ['blue', 'red']
    
    # loop over keys to get conditions
    for idx, condition in enumerate(data_all['acf_norms_avgrs'].keys()):
        # loop over the different acf_norms_avgrs for each condition
        for acf_norms_avgr in data_all['acf_norms_avgrs'][condition]:
            
            ax.plot(acf_norms_avgr, color=mycolors[idx])
    
    ax.set_title('Radial Autocorrelation')
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Normalized Autocorrelation')
    ax.legend()
    
    plt.show(); plt.close()
    
plot_acf_norms_avgrs(data_all)    

# Now the same for the inter-island distance metric
def plot_interisland_distances(data_all):
    """
    Plot the total inter-island distances for each condition.
    """
    
    fig, ax = plt.subplots(figsize=(10*cm_to_inch, 5*cm_to_inch))
    
    conditions = list(data_all['total_interisland_distances'].keys())
    distances = [np.mean(data_all['total_interisland_distances'][cond]) for cond in conditions]
    
    ax.bar(conditions, distances, color=['blue', 'red'])
    
    ax.set_title('Total Inter-Island Distances')
    ax.set_ylabel('Distance (pixels)')
    
    plt.show(); plt.close()


plot_interisland_distances(data_all)