
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