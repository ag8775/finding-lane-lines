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

[image1]: ./test_images_result/actual_image_read_result.png "Original Image"

[image2]: ./test_images_result/gray_scale_conv_result.png "Grayscale Image"

[image3]: ./test_images_result/gaussian_blurred_result.png "Gaussian Averaging"

[image4]: ./test_images_result/canny_edge_result.png "Canny edge Result"

[image5]: ./test_images_result/masked_image_result.png "Masked Image, Region of Interest"

[image6]: ./test_images_result/hough_result_solidWhiteRight.png "Hough Result"

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


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
