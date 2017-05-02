#!/usr/bin/python

#Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import cv2
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def slope(x1, y1, x2, y2):
    return ((y2 - y1) / (x2 - x1))

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    x_right = []
    y_right = []
    x_left = []
    y_left = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_slope = slope(x1, y1, x2, y2)
            if line_slope >= 0:  #Right lane
                x_right.extend((x1, x2))
                y_right.extend((y1, y2))
            else:  #Left lane
                x_left.extend((x1, x2))
                y_left.extend((y1, y2))
    
    #print('x_right')
    #print(*x_right)
    #print('\n')

    #print('y_right')
    #print(*y_right)
    #print('\n')

    #print('x_left')
    #print(*x_left)
    #print('\n')

    #print('y_left')
    #print(*y_left)
    #print('\n')

    #x_right_sorted = sorted(x_right)
    #x_right_len = len(x_right)
    #print(*x_right_sorted)

    #x_left_sorted = sorted(x_left)
    #x_left_len = len(x_left)
    #print(*x_left_sorted)

    #y_right_sorted = sorted(y_right)
    #y_right_len = len(y_right)

    #y_left_sorted = sorted(y_left)
    #y_left_len = len(y_left)

    fitR = np.polyfit(x_right, y_right, 1)
    fit_functionR = np.poly1d(fitR)
    x1R = min(x_right)
    y1R = int(fit_functionR(x1R))
    x2R = max(x_right)
    #x2R = sorted(x_right, reverse=True)[0]
    y2R = int(fit_functionR(x2R))
    cv2.line(img, (x1R, y1R), (x2R, y2R), color, thickness)

    fitL = np.polyfit(x_left, y_left, 1)
    fit_functionL = np.poly1d(fitL)
    x1L = min(x_left)
    y1L = int(fit_functionL(x1L))
    x2L = max(x_left)
    #x2L = sorted(x_right, reverse=True)[2]
    y2L = int(fit_functionL(x2L))
    cv2.line(img, (x1L, y1L), (x2L, y2L), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def pipeline(image):
    #your pipeline to find lines should be defined here
    print('This image is:', type(image), 'with dimensions:', image.shape)
    #plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    #plt.show()
    
    #grayscale conversion
    gray_image = grayscale(image)
    #plt.imshow(gray_image, cmap=cm.gray)
    #plt.show()
    
    #Gaussian smoothing/blurring for suppressing noise and spurious gradients by averaging
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    kernel_size = 5
    blur_gray_image = gaussian_blur(gray_image, kernel_size)
    #plt.imshow(blur_gray_image, cmap=cm.gray)
    #plt.show()
    # Define parameters for Canny and run it
    # XXX: Some optimization may be needed for these thresholds
    low_threshold = 100
    high_threshold = 200
    edges_image = canny(blur_gray_image, low_threshold, high_threshold)
    # Display the image
    #plt.imshow(edges_image, cmap='Greys_r')
    #plt.show()
    
    #grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    region_select = np.copy(image)
    #define a triangle region of interest
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [xsize/2, ysize/2]
    #print(*apex)
    #Vertices of a triangle
    triangle = np.array([ left_bottom, right_bottom, apex ], np.int32)
    #print(*triangle)
    #Get the masked region with the everything masked out (hopefully!), except lane lines
    masked_region_image = region_of_interest(edges_image, [triangle])
    # Display the image
    #plt.imshow(masked_region_image, cmap='Greys_r')
    #plt.show()

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    line_image = hough_lines(masked_region_image, rho, theta, threshold, min_line_length, max_line_gap)

    # Create a "color" binary image to combine with line image
    #color_edges = np.dstack((edges_image, edges_image, edges_image)) 

    # Draw the lines on the edge image
    combo = weighted_img(line_image, image)
    #combo = weighted_img(color_edges, line_image)

    #plt.imshow(combo)
    #plt.show()
    return combo

def process_image(image):
    result = pipeline(image)
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
#%time white_clip.write_videofile(white_output, audio=False)
#get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Read in the image and print out some stats
#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve2.jpg')
#image = mpimg.imread('test_images/solidYellowLeft.jpg')
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#combo = pipeline(image)
#plt.imshow(combo)
#plt.show()










