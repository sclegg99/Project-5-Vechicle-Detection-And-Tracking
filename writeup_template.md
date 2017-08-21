# Project 5--Vechicle Detection and Tracking

# *Rubic Points*
## The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

## Camera Calibration
Camera calibration was performed using the python script [*CameraCalibration.py*](./CameraCalibration.py). The routine performed the following:
1. Read a chessboard image.
2. Convert the image from color to gray scale.
3. Identify the chessboard corner locations using the OpenCV routine findChessboardCorners.
4. Further refine the estimate of the corner location using the OpenCV routine cornerSubPix.
5. Append the corner locations to a list
6. Repeat steps 1-5 until all the chessboard images have been processed.
7. The camera calibration matrix and distortion coefficients are then estimated using the OpenCV routine cv2.calibrateCamera.
8. The accuracy of the calculated camera calibration is then estimated by reprojecting the corner points onto the chessboard.
9. Finally the calibration matrix and distortion coefficients are saved to a pickle file.
A total of twenty chessboard images were provided to the camera calibration.  Of those twenty files, five were rejected for processing due to incompatible image sizes.  The final estimated camera calibration values were:

Camera matrix is

| | | |
| :---: | :---: | :---: |
| 1158.7734939444706 |    0.0 | 669.6411388407582 |
|    0.0 | 1154.0749356989813 | 388.0784814554919 |
|    0.0 |    0.0 |   1.0 |


Lens distortion coefficients are

| Coefficient | Value |
| :---: | :---: |
| r1 |-0.2567800687154464 |
| r2 | 0.04339333102260943 |
| r3 |-0.1150366506303131 |
| t1 |-0.0006876303026921493 |
| t1 | 0.00012574538927438055 |

Reprojection estimates are

|  |  |
| :---: | :---: |
| Mean | 0.77856300011631 |
| Standard Deviation | 0.2915895232436049 |

A montage of the original chessboard images and their images after undistortion is show in Figure 1.

![Figure 1](./Figures/checkboard_montage.png?test=raw)

Figure 1: Montage of original and undistorted chessboard images.

The physical size of the chessboard squares was not provided for this exercise.  Hence the estimated focal lengths are not the actual focal length but are only relative to the assumed chessboard size of 1 unit by 1 unit.

---

## Lane Finding Pipeline
The first step was to initialize the camera calibration data by reading for the pickle file which contained the camera calibration matrix and distortion coefficients.  The the pipeline was executed as follows:
1. A frame of the project video was read.
2. The frame image was undistorted using the camera calibration data and the OpenCV routine undistort.
3. The undistorted image was converted to a gray scale.  Several gray scale conversions were considered and the optimal choice was to separate the image into its YCrCb channels and select on the Y channel for additional processing. (See line 173.)
4. The Sobel X and Y gradients were calculated over the extent of the gray scale image with a kernel size of 9 and thresholds of (50, 200). (See lines 182 through 187.)
5. The X and Y gradients were added weighting the X gradient by 2 and the Y gradient by 1. (See line 192.)
6. Then a morphological close was performed on the weighted binary image with a default kernel size of 3.  The purpose of this step was to eliminate spurious pixels and while connecting neighboring pixels. (See lines 203 and 204.)
7. This closed image was then warped into a birds-eye view using the OpenCV warpPerspective routine. (See line 207.)
8. The warped image was then masked such that only the areas around the lane were visible.  The mask was dynamically adjusted to follow the radius of curvature of the lanes. (See lines 212 through 218.)
9. The masked image was then processed to by the routine supplied by Udacity to estimate the curve that represents the left and right lanes.  The Udacity routine was modified to fit a 2nd order polynomial with forced the slope of the curve representing the lane to 0 at the bottom edge of the image.  It was reasoned that the lane line must be parallel to the direction of car travel hence the slope (dx/dy) is zero at the bottom of the warped video image. (See find_the_lanes -- lines 237 through 398.)
10.  The [radius of curvature](https://en.wikipedia.org/wiki/Radius_of_curvature) was then calculatingated for the left and right lane line curves.
11. A polyfill was generated representing the area between the left line curve and right line curve. (See line 231.)
12. The polyfill was then unwarped and distorted to match the original frame conditions (See lines 232 and 545.)
13. This unwarped and distorted image was then superimposed on the original frame and displayed and saved (See lines 575 and 576.)
14. Steps 1 through 13 were repeated until all the video frames were read and processed.

### 1. Distortion-corrected image.

Step 2 is illustrated in Figure 2 where a test image was undistorted using the previously calculated and stored camera calibration data.

![Figure 2](./Figures/StraightLaneUndistorted.png?test=raw)

Figure 2: Undistorted image. The sample straight lines image was used to illustrate this step

### 2. Steps 4 through 6 used color transforms, gradients, morphologic close and masking to create a thresholded binary image.

The image was mapped into YCrCb color space and the Y channel was then processed with Sobel X and Y gradients.  The YCrCb space was chosen because this color mapping seemed to be less sensitive to changes in the background color of the road surface.  I also explored the use of contrast limited adaptive histogram equalization (CLAHE) to improve the gradient selection.  However, CLAHE did not improve the line selection.  Therefore, CLAHE was not used.  Examples of the X and Y gradients are shown in Figures 3 and 4.

![Figure 3](./Figures/SobelXNoCLAHE.png?test=raw)

Figure 3: Sobel X gradients applied to sample images. Kernel was 9 and thresholds were (50, 200)

![Figure 4](./Figures/SobelYNoCLAHE.png?test=raw)

Figure 4: Sobel Y gradients applied to sample images. Kernel was 9 and thresholds were (50, 200)

The result of combining the X and Y gradients and morphologic close are illustrated in Figure 5.

![Figure 5](./Figures/ExampleBinaryImage.png?test=raw)

Figure 5: Illustration of operating on the frame (a) to obtain the sum of the gradients (b) and then the result of the morphologic close (c).

### 3. Perspective transform to obtain a birds-eye view.

A perspective transform was performed to warp the image into a birds-eye view.  The warping was performed using the OpenCV function warpPerspective which is invoked on line 207.  The straight line images were used to define the source (src) and destination (dst) points.
```
src = np.float32([[972.296, 613.839],
                      [697.561, 442.204],
                      [626.414, 442.204],
                      [372.814, 613.839]])

offset_x = (image_shape[0]-137)/2
offset_y = image_shape[1]-300

dst = np.float32([[116+offset_x,286+offset_y],
                      [114+offset_x,  0+offset_y],
                      [  2+offset_x,  0+offset_y],
                      [  0+offset_x,286+offset_y]])
```

The warped image is subsequently masked to remove spurious pixels.  Figure 6 illustrates the process of going from the gradient image to the warped image to the final masked image.  The masking operation dynamically used the prior estimate of the lane position to construct the area to mask out.

![Figure 6](./Figures/WarpedMasked.png?test=raw)

Figure 6: Illustration of gradient image (a) transformed to a birds-eye view (b) which is subsequently masked (c)

### 4. Lane-line pixels and fitting their positions with a polynomial

The basic moving window code provided by Udacity was used to identify the lane pixel positions for the left and right lane lines.  The left and right set of pixels were fit to a 2nd order polynomial.  The fit was constrained such that the slope (dx/dy) at the bottom of the frame is 0.  The rational for constraining the slope to 0 is the assumption that the lanes are parallel to the forward motion of the car.  The function representing the curve can be found on lines 417-418.

The output of the fit was a set of polynomial coefficients for the left and right lane lines.  A rolling average over the last nine frames of the left and right lane lines polynomial fits was then used to calculate the radius of curvature R of the left and right lane lines.  The rolling average served to minimize the effect of noisy fit values.

In addition to using a rolling average to minimize the impact of noisy fit values, two other checks on the fit values were imposed.  One, the number of pixels to fit had to exceed 100.  If the number of pixels for either the left or right lane dropped below 100, then the values from the previous fit were used for the frame currently being processed.  Similarly, if the lane width deviated from the expected lane width (3.1m) by more than 5%, then the fit for the left and right lane lines was rejected and the fit for the previous frame was used for the current frame.

The fit results for a sample frame are illustrated in Figure 7.  This figure shows area between the left and right lane line fits in green superimposed over the unmasked gradient image.  It should be noted that only the pixels in the lower half of the gradient image are used in the fitting routine because the perspective transform tends to blur (smear) the pixels that far into the horizon (the top half of perspective image).

![Figure 7](./Figures/FittingResults.png?test=raw)

Figure 7: Illustration of the left and right lane line fits.

### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

The formula for calculating the [radius of curvature](https://en.wikipedia.org/wiki/Radius_of_curvature) (function curvature lines 421-422).  The calculated radius of curvature is a parametric curve, hence the radius varies along the arc of the lane line.  Therefore, to obtain a single value that would represent the radius the parametric values returned by the function curvature was averaged over the length of the line.  Finally, the left and right lane radii are averaged to obtain the steering radius.

The position of the car relative to the center of the lane is calculated at line 560. The value is calculated by first obtaining the distance of the camera to the left and right lines then calculating the difference of those distances.

### 6. Plotting the lane back down to the road surface.

The estimated position of the lane was projected onto the frame as follows.  First the curves representing the average fits for the left and right lines were plotted in the birds-eye view space and then a polyfill between the two lines was performed.  This step was performed in plot_lane (lines 419-435).

Next the image of the polyfilled region was projected from the birds-eye view space to the undistorted image space using the OpenCV routine warpPerspective (see line 232).  The inverse mapping function (Minv) taking the view from the dst (birds-eye view) to the src (undistorted image) was used (see get_transform lines 60-63).

One last step is required to complete the plotting of the estimated lane position on to the camera image.  The last step required is to distort the lane image to match the camera distortion.  This step was performed distort function (lines 158-167).

Finally the lane image was superimposed upon the video frame along with text describing the lane radius of curvature and car position with respect to the lane center. An example of this final displayed value is illustrated in Figure 8.

![Figure 8](./Figures/FinalImageExample.png?test=raw)

Figure 8: Illustration of the estimated lane curvature and position overlaid upon the input video frame.

---

## Pipeline (video)

### 1. Final video output.

The project video was processed and saved according to the steps outlined above.  The [final result](./lane_finding_video.mp4) can be viewed to confirm how successful the proposed pipeline determined the lane line locations.

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail? What could you do to make it more robust?

The greatest challenge was finding a combination of gradient operators and thresholds to effectively pick out the lane lines especially when the contrast between the lines and the road was small as in the challenge video.

The pipeline will probably fail if 1) the lane width is not equal to 3.1m.  It will also probably fail if the lane makes sudden changes in direction.

Introducing a lane width parameter would be one step that could be taken to make the pipeline more robust.  In addition to that, exploring methods to improve the line contrast would improve the pipeline robustness.
