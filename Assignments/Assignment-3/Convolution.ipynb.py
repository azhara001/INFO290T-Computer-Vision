# perform convolution with 1-D filter
def convolution(im, h):
    
    # we can have the boundaries as zero in the output image
    im_h = np.zeros_like(im)
    
    # start and end positions
    if len(h.shape) < 2:
        sx = int(np.floor(h.shape[0]/2))
        sy = 0
    else:
        sy = int(np.floor(h.shape[0]/2))
        sx = int(np.floor(h.shape[1]/2))
    
    # use two for loops for convolution
    for i in range(im_h.shape[0]):
        for j in range(im_h.shape[1]):
            
            # wrap the pixel location around
            ys = np.arange(i-sy, i+sy+1)
            ys[ys < 0] = ys[ys < 0] + im_h.shape[0]
            ys = ys % im_h.shape[0]
            
            xs = np.arange(j-sx, j+sx+1)
            xs[xs < 0] = xs[xs < 0] + im_h.shape[1]
            xs = xs % im_h.shape[1]
                        
            # the convolved image
            im_h[i,j] = np.sum(h * im[ys, :][:, xs])
    
    return np.squeeze(im_h)
