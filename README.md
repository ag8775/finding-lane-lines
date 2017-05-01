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

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
