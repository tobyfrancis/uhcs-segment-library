# uhcs-segment-library
A library of models, loading, and training procedures for uhcs-segment

# Models
Currently, only the implementation of dense-hypercolumn-net is tested. This implementation treats each image as a batch - it's a trick to allow one image to be trained onto multiple outputs. 
The ImageDataGenerator is a tiny bit broken right now - I'm attempting to crop to the rotation and then upsample the image and label-image from inside the rotated part, but my cropping code is a bit messed up. 
