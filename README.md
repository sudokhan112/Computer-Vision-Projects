# Computer-Vision-Projects

### Project 1: Camera Calibration
In this project, you will implement camera calibration as described in your textbook and lectures. You will be given
information about a calibration rig and a picture of the rig as part of the given data for this project.
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Camera-calibration/calibration-rig.jpg" width="600" height="400">

### Project 2: Low level image processing
- Implement a **histogram equalization** program that flattens the histogram of the input image as described in lecture by creating a mapping function c(I) based on the histogram of the original image.
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Low-level-image-processing/ctscan_image_hist.png" width="600" height="400">

- Implement a function that does a **log mapping** of the input image (thus enhancing the detail in dark regions).
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Low-level-image-processing/building_log.png" width="600" height="400">

- Write a function that will take an input angle and produce an output image which is the **input image rotated by around the image center**. Normally is positive if the rotation is counter-clockwise (CCW) and negative otherwise. If the pixels in the output image do not correspond to a rotated pixel of the input image, then set their value to 0 (black).
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Low-level-image-processing/car.png" width="600" height="400">

- Implement the **Gaussian averaging filter**. You will have to design the filter mask (integer approximation or floating point samples; it’s up to you). You may want to have sigma of the Gaussian filter as an input parameter that you can easily vary. This will let you experiment with different width Gaussian filters. The filter mask size may depend on the value of sigma.
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Low-level-image-processing/car_opencv_gaussnoise.png" width="600" height="400">

- Implement the **median filter**. Use 3 x 3 neighborhood.


### Project 3 Corner-Edge-Line Detection
- Corner Detection:

Implement the **Harris-Stephens corner detection** method. Test your implementation on images. You will need to
implement the thresholding and nonmaxima suppression to obtain your corners in the image. Matlab has an implementation
of the corner detector called corner(). (see http://www.mathworks.com/help/images/ref/corner.html)
You can compare the results of your implementation to that of Matlab function.
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Corner-edge-line/harris%20keys.png" width="600" height="400">

- Edge detection:

Implement the **Canny edge detector**. This will include implementing the directional Gaussian derivatives, non-maxima
suppression, and hysteresis thresholding to find the true edges. You should attempt to automate the process as much
as possible including the threshold selection. Compare the results you obtain using your implementation with those
of a standard package. Matlab has a Canny edge detector implemented which you can call with various parameter
values. The function is edge(). The parameters will include sigma for the Gaussian and the two threshold values for the
hysteresis thresholding. (see the link.) You can compare your results to that of the OpenCV implementation.

<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Corner-edge-line/canny%20building.png" width="600" height="400">


- Line detection and fitting:

Write a program that will take the output edges detected in previous section and organize them into more abstract structures.
The higher level structures you are to detect are straight lines. You will do this by using the **Hough transform**.
Implement the Hough transform to detect straight lines from the edge images. In order to work with a reasonably
clean transform space, implement the version of Hough transform that uses the gradient direction of the edges to build
the Hough transform. The Hough transform for straight lines gives you the parameters of infinite straight lines. Now
you have to decide which points in the image belong to which straight lines. This is called the back-projection. After
you have identified the edge points that belong to one straight line, do an eigenvector line fitting to obtain the optimal
straight line parameters. Use the normal equations of the line for building the Hough parameter space. Also apply **RANSAC** for line detection.
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Corner-edge-line/hough%20pillset.png" width="600" height="400">
<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Corner-edge-line/ransac%20pills.png" width="600" height="400">

Some of the issues you have to deal with are:
1. How to get the parameters from the position and gradient direction information? For example, consider the
effect of the sign of the contrast on which bins to update. It could be that there are object edges that lie on the
same line but their contrasts are reversed (for example on a checkerboard). Do you consider them the same line
or not?
2. What quantization to use for the parameter space?
3. How to deal with counts being split into adjacent bins. Your textbook hints at a solution.
4. How to perform the peak detection? Thresholding?
5. How to do the back projection? etc.

### Project 4 Segmentation/Connected Componenet/Statistical Features

1. Do the segmentation of the image based on an analysis of the intensities in the image
(i.e. create a histogram, decide on a threshold and do the segmentation). Use a simple
thresholding strategy of finding the valley between two peaks. You may want to smooth
your histogram before trying to find the valley. Remember that the histogram is a 1D
array of frequencies and not a 2D image (although this is what we use to view it). So,
smoothing the histogram is done with a 1D filter on the 1D array. Use the following 1D
averaging mask to accomplish this task: (1/9,2/9,3/9,2/9,1/9). How you will define a
valley is up to you.

<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Segment-CCA-feature/coin_segment.png" width="600" height="400">

2. Perform a connected component analysis on the resulting binary image of step 1. Use
4-connected neighbor definition and the algorithm.

<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Segment-CCA-feature/cca%20shapes4.png" width="600" height="400">

3. For each component identified compute some features that will be useful in describing
its shape. The features you are to compute are the following:
(a) Area, A.
(b) The centroid
(c) Second order moments around the centroid (i.e., central moments).
(d) Perimeter, P
(e) Compactness measure 
(f) An elongation measure computed from second order  
Example:  
Shapes found :  5  
Moments for shape 1  
Center (41, 100)  
Area 856.5  
mu20 0.11653354147302813  
mu02 0.08105050376576435  
mu11 0.025920526449833337  
Perimeter 139.43859779834747  
Compactness 1.8064618003059207  
Elongation 0.4911668348129967  

<img src="https://github.com/sudokhan112/Computer-Vision-Projects/blob/main/Segment-CCA-feature/feature%20shape1.png" width="600" height="400">
