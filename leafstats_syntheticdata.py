
################################################################################
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)

import os

################################################################################
# %%

# Paths below are relative, and are interpreted relative to your working
# directory, so run this script from the root of the repository.
# (Use absolute paths instead when your own data lives elsewhere.)

# Where to put plots
OUTPUTDIR = 'Synthetic_data/OUTPUT'

# Where to find synthetic images
SYNTHETIC_IMAGE_PATH = 'Synthetic_data/'

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
