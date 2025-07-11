        time_start = time.time()
        for i in range(1000):
            img_dist = cv2.distanceTransform(src = (current_lbl==0).astype(np.uint8),
                                        distanceType=cv2.DIST_L2, 
                                        maskSize=cv2.DIST_MASK_PRECISE)
        time_end = time.time()
        print(f'Calculation time for label {lbl}: {time_end - time_start:.2f} seconds')
            
        time_start = time.time()
        for i in range(1000):        
            img_dist = ndi.distance_transform_edt(current_lbl==0)
                # plt.imshow(img_dist); plt.contour(lbl_damage==lbl, colors='red'); plt.show(); plt.close()
        time_end = time.time()
        print(f'Calculation time for label {lbl}: {time_end - time_start:.2f} seconds')