# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

## Implementation details

### We use the following standard methods to mark lane lines on the road from the images captured by a camera feed:
* We transform the image into grayscale
* Use Gaussian smothing to suppress noise and spurious gradients by averaging
* We then identify the edges using the Canny Edge detection algorithm. This requires optimization of the low and high thresholds to identify pixels that may form an edge.
* Determination of the region of interest is in context of the lane in which the car is driving. We have used a triangle with the upper point being the middle of the horizon (or upper horizontal edge).
* We the pursue hough lines detection with static optimization of rho, theta, threshold and the min/max line gaps. We also optimize draw_lines to make sure we  curve-fit the lines that meet the same/similar slope for both the right and left lanes.
---

**Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./test_images_output/actual_image_read_result.png "Original Image"

[image2]: ./test_images_output/gray_scale_conv_result.png "Grayscale Image"

[image3]: ./test_images_output/gaussian_blurred_result.png "Gaussian Averaging"

[image4]: ./test_images_output/canny_edge_result.png "Canny edge Result"

[image5]: ./test_images_output/masked_image_result.png "Masked Image, Region of Interest"

[image6]: ./test_images_output/hough_result_solidWhiteRight.png "Hough Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:
* Original image to process ![alt text][image1]
* Conversion to grayscale ![alt text][image2]
* Gaussian smoothing ![alt text][image3]
* Canny edge detection ![alt text][image4]
* Identifying region of interest ![alt text][image5] 
* Finally, hough result with optimization for identifying left and right lanes, with similar slope gradient ![alt text][image6]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by iterating over lines with +ve and negative slopes. We then extended the right and left edges to incorporate it in the right and left lanes respectively.


### 2. Identify potential shortcomings with your current pipeline

* Curvy lanes require trimming the region of interest. Currently its a triangle that spans across the height of the image. After trimming, we need to identify dimensions of the polygon of interest and ping-ponging between polygon that spans in steps from image after image to a triangle.
* The draw_lines function needs to be more intelligent. Instead of collating +ve and -ve sloped lines into two buckets for left and right lanes, we need to identify error-margins beyond which lines may not acceptable. This avoids incorporating edges that are actually within the min-max line gap of the hough transform but may not necessarily relate to a lane.

### 3. Suggest possible improvements to your pipeline

* Context relevant region of interest determination when straight edges turn to curves in the next series of images
* We could probably support a backtracking logic to identify transition of edges near the car and far from the car and then weigh them accordingly. Essentially the further out, the edges are, the more noise creeps in. This needs to be accomodated. Currently the pipeline works on a frame or image by image basis in time, without taking into account the transition between frames for near and far edges for left and right lanes.
