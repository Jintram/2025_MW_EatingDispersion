

# To do / done (2/9/2026)

- [ ] Working on readme, currently editing "Island count"
    - But also made some changes (see below), maybe quickly go over whole readme.md
    as well.
    - Check if new synthetic plots (usual pipeline) can be used to improve
    illustrations as well.
- [ ] Assessing correct working of pipeline
- [ ] Adding LLM acknowledgement


- [X] Added average nearest-island distance. Note that both total and
    average serve their own purpose. Total is "how much did the thrips
    walk without eating" (very colloquially put), the other is
    "how far are islands typically apart" (related but not the same).

- [X] Bug fix regarding the acf. I made a mistake and didn't realize
    the scipy correlate function calculates the raw 2nd moment, 
    instead of the pearson correlation. The function `get_autocorrelation()`
    now properly calculates the Pearson correlation. This also 
    changes the sensitivity of the acf plots; we can now actually see
    differences in the example data. 
        - Might be interesting to re-generate
        these plots for real data and inspect if we can see changes.

- [X] Fixed something else; previously, the radial distribution was based
    on the damage **mask**, now it is based on the damage **intensity**.
    I think the latter is preferred, as maks is the same information but
    more coarse grained.
    
- [X] Moved synthetic data to within the repository.
- [X] Also applied "normal" pipeline to the synthetic data.
    - Modified synthetic images to have background = 1 for the damage, 
    leaf mask needs to be identified with otsu threshold (background = 0 
    led to artifacts with the thresholds for damage, for leaf, 
    Otsu threshold could be used such that that background could remain
    0).

#### To do for later

- [ ] The code can still be improved from a software engineering perspective, 
and could also be further improved regarding readability (e.g. function
names and comments). 

# Changelog, notes on updates (18/2/2026)

- To allow for 1-channel image to be processed (ie no independent channel for 
leave segmentation), the script was modified to take images with 1 channel as 
input.
    - See the script `leafstats_project_example_1channel.py` for an example handling
    1-channel image data. 
    - The example script `leafstats_projects_example_3channels.py` shows an example
    for handling data which does contain 3 channels.
        - comment: i think data with both leaf and damage channel is preferred, 
        given that determining a good threhsold is much ahrder in 1-channel images.
- To process 1-channel images, the procedure to determine the threshold was
changed, such that other methods can now be chosen. The threshold for 1-channel
images needs to be chosen much more carefully, and I achieved this using the 'triangle' method.
- In addition, new data contained samples that were empty. Automatic handling for this
(determined by no leaf region found) was implemented.
- In addition, to handle artifact or otherwise faulty leaf regions, i implemented
a rounndess determining function, that can be used for filtering.
- The total area of the damage is now calculated, and if `pixel_to_cm2_factor` is set, 
it will calculate that area also in units of $cm^2$. Note that the absolute 
amount of damage is taken, as the nr of thrips added to each leaf is constant
within each experiment (so normalizing for leaf area not prudent).