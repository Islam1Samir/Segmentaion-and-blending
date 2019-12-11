import numpy as np
from PIL import Image
import cv2
import scipy.ndimage
import matplotlib.pylab as plt
import pickle
import scipy.misc




def oval(im,c,rx,ry):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if ((i-c[0])**2/rx**2)+((j-c[1])**2/ry**2)<1:
                im[i,j]=-1

    return im
def blending(im1=0,im2=0,mask=0):

    im1_copy =im1.copy()
    gp_im1 = [im1_copy]
    for i in range(6):
        im1_copy = cv2.pyrDown(im1_copy)
        gp_im1.append(im1_copy)
    im2_copy = im2.copy()
    gp_im2 = [im2_copy]
    for i in range(6):
        im2_copy = cv2.pyrDown(im2_copy)
        gp_im2.append(im2_copy)

    im1_copy = gp_im1[5]
    lp_im1 = [im1_copy]
    for i in range(5,0,-1):
        gaussian_expand = cv2.pyrUp(gp_im1[i],(46,69))
        laplacian = cv2.subtract(gp_im1[i-1],gaussian_expand[:gp_im1[i-1].shape[0],:gp_im1[i-1].shape[1]])
        lp_im1.append(laplacian)

    im2_copy = gp_im2[5]
    lp_im2 = [im2_copy]

    for i in range(5, 0, -1):
        gaussian_expand = cv2.pyrUp(gp_im2[i])
        laplacian = cv2.subtract(gp_im2[i - 1], gaussian_expand[:gp_im1[i-1].shape[0],:gp_im1[i-1].shape[1]])
        lp_im2.append(laplacian)

    mask_py = [mask]

    for i in range(6):
        mask = cv2.pyrDown(mask)
        mask_py.append(mask)

    im1_im2_pyr = []
    n=0

    for im1_lap,im2_lap in zip(lp_im1,lp_im2):
        _,threh =cv2.threshold(mask_py[5-n],0.2,1,cv2.THRESH_BINARY)
        lab = threh * im1_lap + (1-threh) * im2_lap
        lab = lab.astype(np.uint8)
        im1_im2_pyr.append(lab)
        n+=1


    reconstruction = im1_im2_pyr[0]
   ## reconstruction = reconstruction.astype(np.uint8)

    for i in range(1,6):
        reconstruction =cv2.pyrUp(reconstruction)
       ## reconstruction = reconstruction.astype(np.uint8)

        reconstruction = cv2.add(reconstruction[:im1_im2_pyr[i].shape[0],:im1_im2_pyr[i].shape[1]],im1_im2_pyr[i])


    reconstruction=reconstruction.astype(np.uint8)
    scipy.misc.imsave('reconstruction.jpg', reconstruction)

    return reconstruction



def default_phi(x):
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi

def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)

def curvature(f):
    fy, fx = grad(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)

def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy

def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)

def active_contour(im,p_min,p_max,num_iter=50,minValue=0.09):
    im = im - np.mean(im)

    img_smooth = scipy.ndimage.filters.gaussian_filter(im, 3)
    g = stopping_fun(img_smooth)
    F = stopping_fun(im)

   ## F = g.copy()
   ## F[F<0.05] =0.01
    g[g<minValue]=0


    phi = np.ones(im.shape[:2])
    phi[int(p_min[1]):int(p_max[1]), int(p_min[0]):int(p_max[0])] = -1
   ## Hight = int(p_max[1])-int(p_min[1])
   ## width = int(p_max[0])-int(p_min[0])
   ## center = [p_min[1]+0.5*Hight,p_min[0]+0.5*width]
    ##phi[283:633,653:813]=-1
  ##  ov=oval(phi,center,Hight/2,width/2)
    dg = grad(g)


    for i in range(num_iter):

        dphi = grad(phi)
        dphi_norm = norm(dphi)
        kappa = curvature(phi)

        smoothing = g * kappa * dphi_norm
        balloon = g * dphi_norm*5
        attachment = dot(dphi, dg)



        dphi_t = smoothing + balloon + attachment

        phi = phi +  dphi_t

    for i in range(6):

        F[F < .01] = 0
        dphi = grad(phi)
        dphi_norm = norm(dphi)
        kappa = curvature(phi)

        smoothing = F * kappa * dphi_norm
        balloon = F * dphi_norm * 5
        attachment = dot(dphi, dg)

        dphi_t = smoothing + balloon + attachment

        phi = phi + dphi_t

    dphi = grad(phi)
    dphi_norm = norm(dphi)
    kappa = curvature(phi)

    smoothing =  kappa * dphi_norm
    balloon =  dphi_norm
    attachment = dot(dphi, dg)

    dphi_t = smoothing + balloon + attachment

    phi = phi + dphi_t
    return phi

np.set_printoptions(threshold=np.inf)
im = Image.open('Moon.jpg')
im2= np.array(Image.open('background2.jpg'))
im = im.resize((im2.shape[1],im2.shape[0]))

im_copy = im.copy()
im1 = im_copy.resize((im2.shape[1]//4,im2.shape[0]//4))


plt.imshow(im1)
X=plt.ginput(4)
X=np.array(X)
plt.close()
a = np.max(X,axis=0)
b =np.min(X,axis=0)
contour = active_contour(np.array(im1.convert('L')),b,a,100,0.05)## 100 and 0.05 is parameter change depend on the input image

mask = np.clip(contour,0,1)
mask1 = 1-mask
mask1 = cv2.resize(mask1,(im2.shape[1],im2.shape[0]))
mask1 =np.dstack((mask1,mask1,mask1))

p1= mask1*im
p1=p1.astype(np.uint8)
plt.figure()
plt.imshow(p1)
plt.show()
scipy.misc.imsave('segmentedImage.jpg', p1)

im = np.array(im)
blending_img=blending(im,im2,mask1)

plt.figure()
plt.imshow(blending_img, vmin=0, vmax=255)
plt.show()
