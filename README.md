# poly-generator
Triangulate an image while preserving pixel values.
It works by subdividing a triangle one of two ways to try to minimize the maximum pixel error (computed using the area of the triangle, the maximum pixel error, and the average pixel error) using gradient descent.  
Requires opencv, install with `python -m pip install opencv-python`.
