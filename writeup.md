

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_test.png "Road Transformed"
[image3]: ./examples/Combined_image.png "Binary Example"
[image4]: ./examples/Gradient_image.png "Gradient threshold"
[image5]: ./examples/S_image.png "S channel threshold"
[image6]: ./examples/R_image.png "R channel threshold"
[image7]: ./examples/Warped_image.png "Warped image"
[image8]: ./examples/polylines.png "Polylines image"
[image9]: ./examples/drawLines_image.png "Draw lines image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./work_with_images.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the 9x6 chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (third cell).  I applied this distortion correction (fourth cell) to the test image using the `cv2.undistort()` function and obtained this result: 
 
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Aplying the distortion correction to a test image by means of cv2.undistort() function we get this corrected image:
![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding funtions are at fifth cell in `work_with_images.ipynb`). 

combined_binary[((r_binary==1) | ((dir_binary == 1) & (s_binary == 1)))] = 1


Here's an example of my output for each step. 
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in cell 12 in the file `work_with_images.ipynb`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner (cell 13):

src_points =np.float32([(200, img.shape[0]), (1125, img.shape[0]), (690, 450),(590, 450)])

dst_points = np.float32( [(320, 720), (980, 720), (980, 0), (320, 0)])

Here's the result after the transformation:
![alt text][image7]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?
The lane lines finding method chosen was the methos of looking for peaks in a histogram (cell 15). I first take a histogram along all the columns in the lower half of the image like this:

histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

With this histogram I am adding up the pixel values along each column in the image. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I use that as a starting point for where to search for the lines. From that point, I use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

Once located the lane line pixels, I use their x and y pixel positions to fit a second order polynomial curve: f(y)=Ay^2+By+C

left_fit = np.polyfit(lefty, leftx, 2)

right_fit = np.polyfit(righty, rightx, 2)

The next step is the visualization which you can see below:

![alt text][image8]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In order to compute de radius of curvature and the position of the vehicle with respect to the lane center I defined a function called 'curvature()' at the cell 8 in 'pipeline.ipynb'. 

The radius of curvature at any point x of the function x=f(y) is given as follows:

\begin{equation}
R_{𝑐𝑢𝑟𝑣𝑒}=\frac{[1+ (𝑑𝑥/𝑑𝑦)^2]^\frac{3}{2}}{ |𝑑^2𝑥/𝑑𝑦^2|}
\end{equation}


In the case of the second order polynomial, the first and second derivatives are: 
\begin{equation}
𝑓̇(𝑦)=A𝑦^2+𝐵y+C
\end{equation}

\begin{equation}
\dot{𝑓̇}(𝑦)=\frac{𝑑𝑥}{𝑑𝑦}=2𝐴𝑦+𝐵
\end{equation}

\begin{equation}
\ddot{𝑓̈}(𝑦)=\frac{𝑑^2𝑥}{𝑑𝑦^2}=2𝐴 
\end{equation}

So, our equation for radius of curvature becomes:

\begin{equation}
𝑅_{𝑐𝑢𝑟𝑣𝑒}=\frac{(1+(2𝐴𝑦+𝐵)^2)^\frac{3}{2}}{|2𝐴|}
\end{equation}

With these equations we would have calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. We actually need to do this calculation after converting our x and y values to real world space. This involves measuring how long and wide the section of lane is that we're projecting in our warped image. For this project, I assume that if we are projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. 

ym_per_pix = 30/720 # meters per pixel in y dimension

xm_per_pix = 3.7/700 # meters per pixel in x dimension

So, the radius of curvature is calculated, directly in meters, applying the following equation:
\begin{equation}
𝑅_{𝑐𝑢𝑟𝑣𝑒}=\frac{(1+(2𝐴yym\_per\_pix+𝐵)^2)^\frac{3}{2}}{|2𝐴|}
\end{equation}


Finally, the distance to the lane center is computed calculating the mean between tha value of f(y) when y = 720 (that is, y_max) for the right and left lane lines and substracting it to the middle point in x axle x= 640:
\begin{equation}
center_{offset} = (\frac{f_{right}(720) + f_{left}(720)}{2} - 640) xm\_per\_pix
\end{equation}

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
 With function 'drawLine' defined at cell 17 in 'work_with_images.ipynb' and at cell 10 in 'pipeline.ipynb' we can plot back down oyr result onto the image. Here's an example.
 
![alt text][image9]

### Pipeline (video)

#### 1. Provide a link to your final video output. 

You can watch the video [here](https://github.com/adricostas/CarND-Advanced-Lane-Lines/blob/master/output.mp4).

#### 2. Sanity check.
In order to make the result better, I check if the lane lines detected are more or less parallel verifying that the derivative at two different points is similar.

If the lane fit do not pass the sanity check, I use the last good fit.

### Discussion

The technique shown works well to the situation it was design for. It will not work well in situations where the curves are outside the chosen boundary region or in different lightness conditions. As a matter of fact, the result in the project video seems to get worse at the moments when the scene is a bit cheerless, and this solution does not work well to challenge videos.

Perhaps the best conclusion to take is that it is easy to create a simple algorithm that performs relatively well, but it is very hard to create one that will have a human level performance, to handle every situation. There are a lot of improvements to be done. And one thing is clear, all is about the preprocessing of the images. 


```python

```
