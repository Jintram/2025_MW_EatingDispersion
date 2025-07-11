




def get_ripley_k(mask, r_max, step = 1):
    """
    Fast Ripley's K for a binary mask (no edge-correction).
    
    CURRENTLY NOT WORKING
    HAVE TO TAKE A DEEPER LOOK AT PRECISELY HOW K AND L ARE DEFINED 
    AND HOW L-R IS EXPECTED TO BEHAVE.     
    """
    # mask = mask_damage['disk']; r_max=30; step=1
    
    coords = np.column_stack(np.nonzero(mask))
    
    n = coords.shape[0]
    if n < 2:
        return np.zeros((r_max + step - 1) // step, dtype=float)

    # condensed distance vector (length n*(n-1)//2)
    d = pdist(coords, metric='euclidean')

    XXXX THINGS ARENT RIGHT BELOW HERE

    # histogram the pairwise distances once
    bins = np.arange(0, r_max + step, step, dtype=float)
    hist, _ = np.histogram(d, bins=bins)
    
    # cumulative → number of pairs with distance ≤ r
    cum_pairs = np.cumsum(hist)

    # normalise by A / N, where N = 
    # total number of unique ordered pairs (n*(n-1))/2
    k_values = cum_pairs * (2) / (n * (n - 1))
    
    effective_area_size = (np.shape(mask)[0] + r_max) * (np.shape(mask)[1] + r_max)
    effective_area_size = (np.shape(mask)[0]) * (np.shape(mask)[1])
    k_values_uniform = np.cumsum((2 * math.pi * bins) / effective_area_size)

    plt.plot(k_values, color='blue', label='observed K(r)')
    plt.plot(k_values_uniform, color='red', label='uniform K(r)')
    plt.legend()
    plt.show(); plt.close()

    # now also determine Ripley's L
    # L(r) = sqrt(K(r)/pi)
    l_values = np.sqrt(k_values / np.pi)
    
    # and now also formulate the r values
    r_values = np.arange(0, r_max, step, dtype=float)

    return r_values, k_values, l_values



# now calculate the ripley functions
r_values = {}; k_values = {}; l_values = {}
for n, key in enumerate(mask_damage.keys()):
    # key = list(mask_damage.keys())[0]
    r_values[key], k_values[key], l_values[key] = get_ripley_k(mask_damage[key], r_max=30, step=1)
    print(f'Calculation {1+n} done')

# now print all k_values
for key in k_values.keys():
    fig, axs = plt.subplots(1, 2, figsize=(15*cm_to_inch, 5*cm_to_inch))    
    axs[0].imshow(mask_damage[key])
    axs[1].plot(r_values[key], l_values[key], label=key)
    axs[1].plot(r_values[key], r_values[key], linestyle='--', color='black')
    plt.show(); plt.close()
