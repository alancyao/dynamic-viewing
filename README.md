# CS280 Project

3d graphics on monitors are unrealistic because they assume a center of projection some distance away from the center of the screen. The position of the viewer is known, however, so the center of projection is set to match the viewer's point of view; this produces a more realistic picture on the screen, much akin to looking through a window. 

## Implementation Details

1. We use a haar-like feature based cascading multi-scale classifier to find the initial face bounding box. At this step, we also ask the user to be a fixed distance away from the camera. This allows us to find an approximation to the focal length f using `width = f * X / Z`.
2. We extract the face given by the bounding box as the template t.
3. We transform t by rotating by -45, 0, and 45 degrees and scaling by 0.5, 1, and 1.5 to get locally scale and rotation invariant templates.
4. We construct gaussian image pyramids on each of those templates of depth 3.
5. For each gaussian image pyramid, we use multiscale-ssd matching to quickly find the global minimizer of the ssd loss function. For the scaled images, we also scale the ssd values according linearly to the amount of pixels the scaled version has compared to the original.
6. Once the loss-minimzing shifts are found, we interpolate the scale and rotation amount linearly by the inverse scaled ssd values. This weighs heavily towards lower values, while also leaning towards the middle for uncertain cases where all the minimal ssds are about the same. This smoothes the transformation values in practice.
7. The projection plane translation, viewing-axis rotation, and viewing-axis translations are passed to the rendering program, and the camera vectors are adjusted straightforwardly.

## Running Code

Python 2.7 is used. Using pip, install scipy, skimage, numpy, and pyglet.

Install opencv from source. Install OpenGL from source if not already installed.

The game can be run with 

    $ python world.py

## Credits

OpenCV for the haar cascade classifier and webcam capture.

Basic Minecraft framework from: https://github.com/fogleman/Minecraft
