

#%% ################################################################################

from PIL import Image
import numpy as np

from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops

from scipy.signal import correlate
    # from scipy.signal import correlate2d # VERY SLOW

import matplotlib.pyplot as plt

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
    acf = correlate(img_masked.astype(float), img_masked.astype(float), method='fft', mode='same')
    
    # normalize
    acf_norm = acf / np.max(acf)
    
    # also calculate the center coordinate of this acf
    acf_center = np.round(np.array(acf.shape)/2).astype(int)
    
    return acf, acf_norm, acf_center

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
# Try with synthetic data

# open tiff stack image
from skimage import io
img_test_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_eatendisk.tif'
img_test = io.imread(img_test_path) # io.read required for img stack
    # img_test.shape

img_leaf   = img_test[:,:,1]  # green channel (leaf)
img_damage = img_test[:,:,2]  # blue channel (damage)

mask_leaf = get_largest_mask(img_leaf, method='otsu')
mask_damage = get_mask(img_damage, mask_leaf, method='bg2') # bg2
centroid_leaf = regionprops(mask_leaf.astype(int))[0].centroid

plot_images(img_leaf, img_damage, mask_leaf, mask_damage, centroid_leaf)

# now get the acf
acf_msk, acf_norm_msk, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)

# plot the acf centerline
plt.imshow(img_damage)
plt.show()
plt.plot(acf_norm_msk[acf_center[0],:])
plt.show(); plt.close()

########################################
# now the same for the eatenspots
img_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Synthetic_data/synthetic_eatenspots.tif'

img_test = io.imread(img_path) # io.read required for img stack
    # img_test.shape

img_leaf   = img_test[:,:,1]  # green channel (leaf)
img_damage = img_test[:,:,2]  # blue channel (damage)

mask_leaf = get_largest_mask(img_leaf, method='otsu')
mask_damage = get_mask(img_damage, mask_leaf, method='bg2') # bg2
centroid_leaf = regionprops(mask_leaf.astype(int))[0].centroid

plot_images(img_leaf, img_damage, mask_leaf, mask_damage, centroid_leaf)

# now get the acf
acf_msk, acf_norm_msk, acf_center = get_autocorrelation(img_damage, mask_user=mask_leaf)

# plot the acf centerline
plt.imshow(img_damage)
plt.show()
plt.plot(acf_norm_msk[acf_center[0],:])
plt.show(); plt.close()

# %%
