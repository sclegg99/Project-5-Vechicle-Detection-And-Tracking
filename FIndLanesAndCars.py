#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Fri Jul  7 09:52:15 2017@author: sclegg1. Input camera and distortion matrix.2. Read image then undistort and warp3. Split image into seperate color channels4. Apply gradient operator to color channel(s)5. Plot data"""import imageioimport numpy as npimport cv2import sys, getoptimport timeimport pickleimport matplotlib.pyplot as pltfrom camdistortion import distort, un_distortfrom animateImage import animateImagefrom laneFinding import get_transform, get_lane_lineimport findCarFunctions as fcimport createheatmap as chmfrom scipy.ndimage.measurements import label# Create a couple of video windows. One to display input the other to display outputvidOut = animateImage(title='Output Video', axis=False,                      position=(300,300))    # Define a class to receive the characteristics of each line detectionclass Lane():    def __init__(self):        # was the line detected in the last iteration?        self.detected = False          # number of frames without detection        self.notdetected = 0        # x values of the last n fits of the line        self.recent_xfitted = np.empty(0)         #average x values of the fitted line over the last n iterations        self.bestx = None         #std polynomial coefficients averaged over the last n iterations        self.std_fit = None        #polynomial coefficients averaged over the last n iterations        self.average_fit = None          #polynomial coefficients for the most recent fit        self.current_fit = [np.array([False])]          #radius of curvature of the line in some units        self.radius_of_curvature = None         #distance in meters of vehicle center from the line        self.line_base_pos = None        #direction of turn (left vs right)        self.direction = None         #difference in fit coefficients between last and new fits        self.diffs = np.array([0,0,0], dtype='float')         #x values for detected line pixels        self.allx = None          #y values for detected line pixels        self.ally = None# define x and y scaling for pixels#scaleX = 3.1/114.#scaleY = 14.1/90.#scaleR = 1.0# define x and y scaling for pixelsscaleX = 3.6/118.140scaleY = 56.4/210.400scaleR = np.sqrt(scaleX*scaleX+scaleY*scaleY)print("scaleX {:8.5f}, scaleY {:8.5f}, scaleR {:8.5f}".format(scaleX,scaleY,scaleR))# Routine to store the bounding boxes.# This routine stores the last N frames of bounding# boxes.def storeBoxes(box_list, bboxes, N):    if len(box_list) == N:        box_list.pop(0)    box_list.append(bboxes)    return box_list# Routine to flatten the stored bounding boxes.def flattenBoxList(box_list):    for bboxes in box_list:        for bbox in bboxes:            yield bbox                def find_lane_lines(inputFile, outputFile, cameraData, classifierFile):        # Load trained classifier data    with open(classifierFile, "rb" ) as f:        classifier = pickle.load(f)    f.close()    clffile = classifier['clf']    colorSpace = classifier['colorSpace']    colorChannel = classifier['colorChannel']    spatialSize = classifier['spatialSize']    spatialFeat = classifier['spatialFeat']    histBins = classifier['histBins']    histFeat = classifier['histFeat']    winSize = classifier['winSize']    blockSize = classifier['blockSize']    blockStride = classifier['blockStride']    cellSize = classifier['cellSize']    nbins = classifier['nbins']    derivAperture = classifier['derivAperture']    winSigma = classifier['winSigma']    histogramNormType = classifier['histogramNormType']    L2HysThreshold = classifier['L2HysThreshold']    gammaCorrection = classifier['gammaCorrection']    nlevels = classifier['nlevels']    hogFeat = classifier['hogFeat']    xScaler = classifier['xScaler']    signed_gradient = classifier['signed_gradient']        # Load trained SVM classifer    clf=cv2.ml.SVM_load(clffile)    # initialize HOG descriptor based on training parameters    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,                            derivAperture, winSigma, histogramNormType,                            L2HysThreshold, gammaCorrection, nlevels,                            signed_gradient)        # Open video input stream    vidin = imageio.get_reader(inputFile)        # Get video stream meta data    meta_data = vidin.get_meta_data()    size = meta_data['size']    width = size[0]    height = size[1]    # Create video writer object    vidout = imageio.get_writer(outputFile, '.mp4', 'I')           # Set some constant parameters    imageSize = (width, height)    colStart = 640    colStop = imageSize[0]    rowStart1 = 400    rowStop1 = 464    rowStart2 = 400    rowStop2 = 720        # ROI to search for cars    winStride = (16, 16)    padding = (0,0) # No image padding            # Adjust ROI to get an integer number of window strides    colStop = int((colStop-colStart)/winStride[0])*winStride[0]+colStart    rowStop1 = int((rowStop1-rowStart1)/winStride[1])*winStride[1]+rowStart1    rowStop2 = int((rowStop2-rowStart2)/winStride[1])*winStride[1]+rowStart2    ROI1 =((rowStart1, rowStop1), (colStart, colStop))    ROI2 =((rowStart2, rowStop2), (colStart, colStop))            # Frame scale factor    scale1 = 1.    scale2 = 2.    box_list = []    # Parameters for lane finding        clip_left = int(.1*width) # pixels to remove from the left of the image    clip_right = int(.1*width) # pixels to removed for the right of the image    clip_top = int(.55*height) # pixels to remove from the top of the image    clip_bottom = int(.1*height) # pixels to remove from the bottom of the image    # Define threshold values    sxy_thresh = (50, 200)        # read camera data    camera_pickle = pickle.load(open(cameraData, 'rb'))    mtx = camera_pickle["mtx"]    dist= camera_pickle["dist"]    print("Got Camera Data")        # Create transformation matrix for wrapping image to birds-eye    # view and inverse transform for unwrapping from birds-eye    # First set src and dst vectors based on image analysis of     # straight lane images    warp_image_size = (320, 640)    warp_margin = 101    warp_width, warp_height = warp_image_size    src = np.float32([[880.815,244.583],                      [569.335, 46.033],                      [499.367, 46.033],                      [208.187,244.583]])    dst = np.float32([[warp_width-warp_margin, warp_height],                      [warp_width-warp_margin, 0],                      [warp_margin, 0],                      [warp_margin, warp_height]])        # generate perspective transform and inverse transform matrices    M, Minv = get_transform(src, dst)        # define default vertices of poly mask    mask_margin = 50    mask_left = mask_margin    mask_right = warp_image_size[0]-mask_margin    mask_top = 0    mask_bottom = warp_image_size[1]    vert0 = [ mask_left, mask_bottom]    vert1 = [ mask_left, mask_top]    vert2 = [ mask_right, mask_top]    vert3 = [ mask_right, mask_bottom]    vertices = np.array([[vert0, vert1, vert2, vert3]], dtype=np.int32)        # font for writing on image    font = cv2.FONT_HERSHEY_SIMPLEX    textPosition = (10, 50)        # define left and right lane from Lane class and     leftLane = Lane()    rightLane = Lane()    # Loop thru the video images    nframes = meta_data['nframes']    frames = list(range(0,nframes))#    frames = list(range(700, 750))    for frame in frames:        image = vidin.get_data(frame)                # Undistort image        undistorted_image = un_distort(image, mtx, dist)        # Clip image        clipped_image = undistorted_image[clip_top:-clip_bottom,clip_left:-clip_right]        # process image through pipeline        lanes, leftLane, rightLane = get_lane_line(clipped_image,                                                   leftLane, rightLane,                                                   vertices, M, Minv,                                                   warp_image_size,                                                   scaleX, scaleY, scaleR,                                                   sx_thresh=sxy_thresh,                                                   sy_thresh=sxy_thresh)                # redistort lane image then superimpose on the original camera image        lanes1 = np.zeros_like(image)        lanes1[clip_top:-clip_bottom,clip_left:-clip_right] = lanes        temp = cv2.addWeighted(undistorted_image, 1, lanes1, 0.3, 0)        result = distort(temp, mtx, dist)        # add text to image        radius = .5*(leftLane.radius_of_curvature+rightLane.radius_of_curvature)        center = .5*(leftLane.line_base_pos+rightLane.line_base_pos)        if np.isnan(radius):            radius = 10000.        if(leftLane.direction < 0):            direction = 'left'        else:            direction = 'right'        if radius > 3000.:            text = "Straight, Distance to lane center {:5.2f}m".format(center)        else:            text = "Radius of curvature: {:7.2f}m to the {}, Distance to lane center {:5.2f}m".format(radius,direction,center)        cv2.putText(result, text, textPosition, font, 1, (255,255,255), 2, cv2.LINE_AA)        # Check the time to scane the image        t1=time.time()        # Call routine to detect cars and return possible hits as        # bounding boxes        bboxes = fc.findCars(image, clf, ROI1, scale1, hog, xScaler, winStride,                              hogFeat=hogFeat, padding=padding,                             colorSpace=colorSpace, colorChannel=colorChannel,                             spatialSize=spatialSize, spatialFeat=spatialFeat,                             histBins=histBins, histFeat=histFeat)        bboxes2 = fc.findCars(image, clf, ROI2, scale2, hog, xScaler, winStride,                              hogFeat=hogFeat, padding=padding,                             colorSpace=colorSpace, colorChannel=colorChannel,                             spatialSize=spatialSize, spatialFeat=spatialFeat,                             histBins=histBins, histFeat=histFeat)        bboxes.extend(bboxes2)        t2 = time.time()                # Store bounding boxes for the last ten framees        box_list = storeBoxes(box_list, bboxes, 5)                # Flatten box list        new_list = list(flattenBoxList(box_list))        # Set heatmap array to zero        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)                # Add heat to each box in box list        heatmap = chm.add_heat(heatmap, new_list)                # Apply threshold to help remove false positives        heatmap = chm.apply_threshold(heatmap, 6)        # Find final boxes from heatmap using label function        labels = label(heatmap)        print("Processed frame {:4d} in {:5.2f} seconds".format(frame+1, t2-t1))                # Create and show image from heatmap labels        draw_img = chm.draw_labeled_bboxes(np.copy(result), labels)        # Display the resulting frame                udst_2 = cv2.resize(draw_img, (0,0), fx=0.5, fy=0.5)            vidOut.show('Output Video', udst_2)        # Write video        vidout.append_data(draw_img)            # When everything done, release the video capture object    plt.close('all')    vidout.close()    vidin.close()    def main(argv):    outputFile = './lane_finding_video.mp4' # default output file name    cameraData = '../CarND-Advanced-Lane-Lines/cameradata.p'    inputFile = '../CarND-Advanced-Lane-Lines/project_video.mp4'    classifierFile = './svcPickle.p' # default pickle file name        try:        opts, args = getopt.getopt(argv,"hi:o:c:d:",                                   ["i=","o=","c=","d="])    except getopt.GetoptError:        print("FindLanesAndCars.py -i <inputFile> -o <outputFile> -c <cameraData> -d <classifierFile>")        sys.exit(2)    for opt, arg in opts:        if opt == '-h':            print("FindLanesAndCars.py -i <inputfile> -o <outputfile> -c <cameraData> √-d <classifierFile>")            sys.exit()        elif opt in ("-i", "--i"):            inputFile = arg        elif opt in ("-o", "--o"):            outputFile = arg        elif opt in ("-c", "--c"):            cameraData = arg        elif opt in ("-d", "--d"):            classifierFile = arg    print("Input file is {}".format(inputFile))    print("Output file is {}".format(outputFile))    print("Camera data file is {}".format(cameraData))    print("Classifier data file is {}".format(classifierFile))    find_lane_lines(inputFile, outputFile, cameraData, classifierFile)if __name__ == "__main__":    main(sys.argv[1:])