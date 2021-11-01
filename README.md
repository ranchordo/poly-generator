# poly-generator
Triangulate an image while preserving pixel values.
It works by subdividing a triangle one of two ways to try to minimize the maximum pixel error (computed using the area of the triangle, the maximum pixel error, and the average pixel error) using gradient descent.  
Requires opencv, install with `python -m pip install opencv-python`.  
WARNING: this was a random project. Not documented well, not optimized.  
  
Example 1 is generated from [here](https://www.e-retail.com/wp-content/uploads/2020/12/7190-0qADHL._AC_SL1500_.jpg)  
Example 2 is generated from [here](https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Dragon_Ljubljana.jpg/1200px-Dragon_Ljubljana.jpg), with the background removed.
