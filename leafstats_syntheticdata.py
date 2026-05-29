
################################################################################
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)

import os

################################################################################
# %%


# Where to put plots
OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250529_SynthData/OUTPUT'

# Where to find synthetic images
SYNTHETIC_IMAGE_PATH = '/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_MW_EatingDispersion/Synthetic_data/'

# 1) Ensure base output directory exists
os.makedirs(OUTPUTDIR, exist_ok=True)

# 2) Define channel configuration (index + display name)
leaf_channel_spec = {'channel': 1, 'name': 'Leaf'}
damage_channel_spec = {'channel': 2, 'name': 'Damage'}
reference_channel_spec = {'channel': 0, 'name': 'Reference'} # can be set to None

# 3) Load synthetic example images and run synthetic sanity-check analysis/plots
img_leafs_syn, img_damages_syn, img_disk = lsa.load_synthetic_data(
    SYNTHETIC_IMAGE_PATH,
    leaf_channel_spec,
    damage_channel_spec
)
lsa.run_synthetic_analysis(
    img_leafs = img_leafs_syn, 
    img_damages = img_damages_syn,
    img_disk = img_disk, 
    leaf_channel_spec = leaf_channel_spec, 
    damage_channel_spec = damage_channel_spec,
    reference_channel_spec = reference_channel_spec,
    outputdir = OUTPUTDIR
)
# img_leafs = img_leafs_syn; img_damages = img_damages_syn; img_disk = img_disk
# leaf_channel_spec = leaf_channel_spec; damage_channel_spec = damage_channel_spec; reference_channel_spec = reference_channel_spec
# outputdir = OUTPUTDIR