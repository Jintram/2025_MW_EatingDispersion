
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

# 2) Define channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 1,
    'Damage': 2,
    'Reference': 0
}

# 3) Load synthetic example images and run synthetic sanity-check analysis/plots
img_leafs_syn, img_damages_syn, img_disk = lsa.load_synthetic_data(
    SYNTHETIC_IMAGE_PATH,
    config_channels
)
lsa.run_synthetic_analysis(
    img_leafs = img_leafs_syn, 
    img_damages = img_damages_syn,
    img_disk = img_disk, 
    config_channels = config_channels,
    outputdir = OUTPUTDIR
)
# img_leafs = img_leafs_syn; img_damages = img_damages_syn; img_disk = img_disk
# config_channels = config_channels
# outputdir = OUTPUTDIR
# %%
