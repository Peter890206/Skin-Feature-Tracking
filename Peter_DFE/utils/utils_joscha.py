import os
import re
import cv2
import numpy as np
import scipy.linalg
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as kb


def sorted_alphanumeric(data):
    # Sorts filenames of in alphanumeric order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def create_relabeling_img(idx,path,filenames,new_size,org_label,window_size):
    # Concatenates reference image with another image from the dataset (idx=idx)
    # The display image is has dimensions 2w x l

    #ref_img=cv2.imread(path+filenames[0],0)
    #ref_img=cv2.resize(ref_img,new_size)
    ref_img=np.zeros((new_size[1],new_size[0])).astype('uint8')
    #draw point on reference image
    ref_clone = cv2.cvtColor(ref_img.copy(),cv2.COLOR_GRAY2RGB)
    #cv2.rectangle(ref_clone, (org_label[0]-window_size[0]//2,org_label[1]-window_size[1]//2),
    #              (org_label[0]+window_size[0]//2,org_label[1]+window_size[1]//2), (255, 0, 0), 2)

    target_img=cv2.cvtColor(cv2.imread(path+filenames[idx],0),cv2.COLOR_GRAY2RGB)
    target_img=cv2.resize(target_img,new_size)
    display_img=np.concatenate((ref_clone,target_img),axis=1)
    return display_img

def create_ref_crop_mosaic(img,window_size,bbox_center,gap):
    #creates a mosaic of original image and zoomed in crop
    #concatenates original image with resized crop horizontally
    #img should be BRG mosaic is RGB
    clone=img.copy()
    x1=bbox_center[0]-window_size[0]//2
    x2=bbox_center[0]+window_size[0]//2
    y1=bbox_center[1]-window_size[0]//2
    y2=bbox_center[1]+window_size[0]//2
    cv2.rectangle(img, (x1,y1),(x2,y2), (255, 0, 0), 2)

    ref_crop=clone[y1:y2,x1:x2,:]
    h=img.shape[0]
    w=img.shape[1]
    ref_crop=cv2.resize(ref_crop,(h,h),fx=0, fy=0,interpolation = cv2.INTER_NEAREST)
    mosaic=(np.ones((h,w+h+gap,3))*255).astype('uint8')
    #print(mosaic.shape)
    mosaic[:,0:w,:]=img
    mosaic[:,w+gap:,:]=ref_crop
    return cv2.cvtColor(mosaic,cv2.COLOR_BGR2RGB)


def get_sift_descriptor(img,px,py):
    sift = cv2.xfeatures2d.SIFT_create()
    ref_point=np.expand_dims(np.expand_dims(np.array([px,py]),axis=0),axis=0)
    ref_point=cv2.KeyPoint_convert(ref_point)
    ref_descriptor=sift.compute(img,ref_point)
    return np.array(ref_descriptor[1])

def get_surf_descriptor(img,px,py):
    surf = cv2.xfeatures2d.SURF_create()
    surf.setExtended(True) # To get more rich (128-dimensional) feauture descriptors
    ref_point=np.expand_dims(np.expand_dims(np.array([px,py]),axis=0),axis=0)
    ref_point=cv2.KeyPoint_convert(ref_point)
    ref_descriptor=surf.compute(img,ref_point)
    return np.array(ref_descriptor[1])


def get_image_crops_locally(img,window_size,stride, pred):
    SW = 25 # Max of 50 x 50 Search Window around point
    max_i = int((img.shape[0]-window_size[0])/stride)
    max_j = int((img.shape[1]-window_size[1])/stride)
    local_i = (np.max([pred[1]-SW,0]), np.min([pred[1]+SW, max_i]))
    local_j =(np.max([pred[0]-SW,0]), np.min([pred[0]+SW, max_j]))
    print('search for x: '+str(local_j))
    print('search for y: '+str(local_i))
    n_crops=int(np.abs(local_i[1]-local_i[0])*np.abs(local_j[1]-local_j[0]))
    crops=np.zeros((n_crops,window_size[0],window_size[1],3))
    pos=np.zeros((n_crops,2))

    idx=0
    for i in range(local_i[0], local_i[1]):
        for j in range(local_j[0], local_j[1]):
            x=int(j+np.ceil(window_size[0]/2))
            y=int(i+np.ceil(window_size[1]/2))
            #print(idx)
            crops[idx,:,:,:]=img[i:i+window_size[1],j:j+window_size[0],:]
            pos[idx,0]=x; pos[idx,1]=y
            idx+=1
    return crops, pos.astype('int')

def get_image_crops(img,window_size,stride):
    # Gets all image crops
    max_i = int((img.shape[0]-window_size[0])/stride)
    max_j = int((img.shape[1]-window_size[1])/stride)
    n_crops=int(max_i*max_j)
    crops=np.zeros((n_crops,window_size[0],window_size[1],3))
    pos=np.zeros((n_crops,2))

    idx=0
    for i in range(max_i):
        for j in range(max_j):
            x=int(j+np.ceil(window_size[0]/2))
            y=int(i+np.ceil(window_size[1]/2))
            #print(idx)
            crops[idx,:,:,:]=img[i:i+window_size[1],j:j+window_size[0],:]
            pos[idx,0]=x; pos[idx,1]=y
            idx+=1
    return crops, pos.astype('int')

def preprocess_deep_encoding_img(path, filename,new_size):
    img=cv2.imread(path+filename)
    img=cv2.resize(img,(new_size[1],new_size[0]))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    Lab_image = convertRGB2CIELab(img)
    return Lab_image


def convertRGB2CIELab(image):
    # Converts uint8 RGB image to float32 CIELAB format
    image = np.array(image).astype('float32')
    if np.max(image) > 1:
        image *= 1/255
    Lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    Lab_image[:,:,0]=Lab_image[:,:,0]/100 # Normalize L to be betweeen 0 and 1
    Lab_image[:,:,1]=(Lab_image[:,:,1]+127)/(2*127)
    Lab_image[:,:,2]=(Lab_image[:,:,2]+127)/(2*127)
    return Lab_image

def get_deep_encodings_descriptor(img,px,py,encoder,window_size):
    ref_x1=int(px-window_size[0]/2)
    ref_x2=int(px+window_size[0]/2)
    ref_y1=int(py-window_size[1]/2)
    ref_y2=int(py+window_size[1]/2)
    ref_crop=np.array(img[ref_y1:ref_y2,ref_x1:ref_x2,:])
    ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
    return ref_encoding

def std_norm(x,mu,sigma):
    return (x-mu)/sigma

def min_max_norm(x, xmax, xmin):
    return (x-xmin)/(xmax-xmin)

def get_pts_local_window(center,neighborhood):
    # 0, 1, 2,
    # 3, 4, 5,
    # 6, 7, 8
    rel_center=np.floor(neighborhood/2)
    window=np.zeros((neighborhood,neighborhood,2))
    for i in range(neighborhood):
        for j in range(neighborhood):
            window[i,j]=[(i-rel_center)+center[0],(j-rel_center)+center[1]]
    return window # pts are in (x,y)

def quadratic_interpolation_deep_encodings(center, img, neighborhood,model, ref_encoding):
    img_dim=(img.shape[0],img.shape[1])
    rel_center=np.floor(neighborhood/2)
    left_bound=center[0]-rel_center
    right_bound=center[0]+rel_center
    upper_bound=center[1]+rel_center
    lower_bound=center[1]-rel_center

    window_size=(31,31)# Size of the crops for encoder
    if neighborhood%2==0:
        print('neighborhood must be odd number')
    elif left_bound<0 or right_bound>img_dim[1] or upper_bound>img_dim[0] or lower_bound<0:
        print('Neighborhood is too large and window exceeds image dimensions')

    else:
        window_pts=get_pts_local_window(center,neighborhood)
        window_pts=np.reshape(window_pts,(neighborhood**2,2))
        #print(window_pts.shape)
        local_crops=[]

        for i in range(window_pts.shape[0]):
            clone=np.array(img)
            xa=(window_pts[i,0]-window_size[0]/2).astype('int')
            xb=(window_pts[i,0]+window_size[0]/2).astype('int')
            ya=(window_pts[i,1]-window_size[0]/2).astype('int')
            yb=(window_pts[i,1]+window_size[0]/2).astype('int')
            if clone[ya:yb,xa:xb,:].shape != (31, 31, 3):
                print(f"crop shape is {clone[ya:yb,xa:xb,:].shape}")
                return None, None, None, None
#             print(xa,xb,ya,yb)
            local_crops.append(clone[ya:yb,xa:xb,:])
        local_crops=np.array(local_crops)
        encodings=np.squeeze(model.predict(local_crops))
        ssr=np.sum((encodings-ref_encoding)**2,axis=1)
        # print("SSR in quadratic_interpolation_deep_encodings shape: ", ssr.shape)
        # print("window_pts shape: ", window_pts.shape)
        #Prepare data for quadratic surface fitting
        window_pts[:,0]-=center[0]
        window_pts[:,1]-=center[1]
        A=np.c_[(np.ones(ssr.shape),window_pts[:,0],window_pts[:,1],np.prod(window_pts,axis=1),
                  window_pts[:,0]**2,window_pts[:,1]**2)]                        
        pred, C, A, B=find_min_surface(A,ssr)
        if not type(C) is np.ndarray: ## for the local maximum, check find_min_surface
            return pred, np.NaN, A, B
        pred=np.clip(pred,-1,1)+center
    return pred, C, A, B

def quadratic_interpolation_sift_surf(center, img, neighborhood, method, ref_descriptor):
    img_dim=(img.shape[0],img.shape[1])
    rel_center=np.floor(neighborhood/2)
    left_bound=center[0]-rel_center
    right_bound=center[0]+rel_center
    upper_bound=center[1]+rel_center
    lower_bound=center[1]-rel_center

    window_size=(31,31)# Size of the crops for encoder
    if neighborhood%2==0:
        print('neighborhood must be odd number')
    elif left_bound<0 or right_bound>img_dim[1] or upper_bound>img_dim[0] or lower_bound<0:
        print('Neighborhood is too large and window exceeds image dimensions')

    else:
        window_pts=get_pts_local_window(center,neighborhood)
        window_pts=np.reshape(window_pts,(neighborhood**2,2))
        img_points=np.expand_dims(window_pts,axis=1)
        img_points=cv2.KeyPoint_convert(img_points)
        descriptors=method.compute(img,img_points)

        ssr=np.sum((descriptors[1]-ref_descriptor[1])**2,axis=1)

        #Prepare data for quadratic surface fitting
        window_pts[:,0]-=center[0]
        window_pts[:,1]-=center[1]
        A=np.c_[(np.ones(ssr.shape),window_pts[:,0],window_pts[:,1],np.prod(window_pts,axis=1),
                  window_pts[:,0]**2,window_pts[:,1]**2)]
        pred, C, A, B=find_min_surface(A,ssr)
        pred=np.clip(pred,-1,1)+center

    return pred, C, A, B

def quadratic_interpolation_tm(center, img, neighborhood, ref_crop):
    img_dim=(img.shape[0],img.shape[1])
    rel_center=np.floor(neighborhood/2)
    left_bound=center[0]-rel_center
    right_bound=center[0]+rel_center
    upper_bound=center[1]+rel_center
    lower_bound=center[1]-rel_center

    window_size=(31,31)# Size of the crops for encoder
    if neighborhood%2==0:
        print('neighborhood must be odd number')
    elif left_bound<0 or right_bound>img_dim[1] or upper_bound>img_dim[0] or lower_bound<0:
        print('Neighborhood is too large and window exceeds image dimensions')

    else:
        window_pts=get_pts_local_window(center,neighborhood)
        window_pts=np.reshape(window_pts,(neighborhood**2,2))
        img_points=window_pts

        crops=np.zeros((neighborhood**2,window_size[0],window_size[1],3))
        for i in range(img_points.shape[0]):
            x1=int(img_points[i,0]-np.ceil(window_size[0]/2))
            x2=int(img_points[i,0]+np.floor(window_size[0]/2))
            y1=int(img_points[i,1]-np.ceil(window_size[1]/2))
            y2=int(img_points[i,1]+np.floor(window_size[1]/2))
            crops[i,:,:,:]=img[y1:y2,x1:x2,:]

        ssr=np.sum(np.sum(np.sum((crops-ref_crop)**2,axis=3),axis=1),axis=1)

        #Prepare data for quadratic surface fitting
        A=np.c_[(np.ones(ssr.shape),window_pts[:,0],window_pts[:,1],np.prod(window_pts,axis=1),
                  window_pts[:,0]**2,window_pts[:,1]**2)]
        pred=find_min_surface(A,ssr)

    return pred

def find_min_surface(A,B):
    # Compute least-squares solution to equation AxC=B
    # C=[1 x y xy x^2 y^2]'
    if A.shape[1]==6:
        C,_,_,_ = scipy.linalg.lstsq(A, B)
        fxx=2*C[4]
        fyy=2*C[5]
        fxy=C[3]
        # Apply the second derivative test
        D=fxx*fyy-[fxy**2]
        if D>0:
            if fxx>0:
                # Minimum is where fx=0 and fy=0
                A_star=np.zeros((2,2))
                A_star[0,0]=2*C[4]
                A_star[0,1]=C[3]
                A_star[1,0]=C[3]
                A_star[1,1]=2*C[5]

                B_star=np.zeros((2,1))
                B_star[0,0]=-C[1]
                B_star[1,0]=-C[2]
                # C_star=[x,y]'

                C_star,_,_,_ = scipy.linalg.lstsq(A_star, B_star)
            else:
                print('You have local maximum')
                print(B)
#                 return [[A[4,1],A[4,2]]],C,A,B #original center at pixel level
                return [[A[4,1],A[4,2]]],np.NaN,A,B
        elif D < 0:
            #print('Concavity changes at some point (saddle point)')
            #print(B)
            return [[A[4,1],A[4,2]]],C,A,B #original center at pixel level
        elif D == 0:
            #print('Second derivative test is inconclusive')
            #print(B)
            return [[A[4,1],A[4,2]]],C,A,B #original center at pixel level

    else:
        print('Can not do surface fitting of order 2')
    return np.swapaxes(C_star,0,1),np.array(C),A,B

def get_gradients_ssr(A,c,ssr,mode):
    #Gets the gradients wrt for a model that represents the x and y center estimation as a function of the ssr values
    #Create the weights for the model
    det=4*c[4]*c[5]-c[3]**2
    temp=np.zeros((6,2))
    temp[1,0]=-2*c[5]/det
    temp[2,0]=c[3]/det
    temp[1,1]=c[3]/det
    temp[2,1]=-2*c[4]/det

    my_weights=[]
    my_weights.append(np.transpose(np.linalg.pinv(A))) #Moore Penrose inverse
    if mode == 'x':
        my_weights.append(np.expand_dims(temp[:,0],axis=-1))
    elif mode == 'y':
        my_weights.append(np.expand_dims(temp[:,1],axis=-1))
    #print(my_weights[1].shape)

    #Create model
    model=Sequential([
    Dense(6,input_shape=(9,),activation='linear',use_bias=False),
    Dense(1,activation='linear',use_bias=False)
    ])
    #Calculate the gradients wrt input
    model.set_weights(my_weights)
    model.compile(optimizer='adam',loss='mse')

    x=tf.cast(np.expand_dims(ssr,axis=0), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
    grads = kb.get_value(tape.gradient(pred, x))
    return grads

def draw_salient_features_dfe(img, window_size,pos):
    for i in range(len(pos)):
        y1 = int(pos[i,1]-window_size[0]/2)
        y2 = int(pos[i,1]+window_size[0]/2)
        x1 = int(pos[i,0]-window_size[0]/2)
        x2 = int(pos[i,0]+window_size[0]/2)
        #cv2.rectangle(img, (x1, y1), (x2,y2), (0,0,255), 1)
        cv2.circle(img,(int(pos[i,0]),int(pos[i,1])),1,(255,0,0),2)
    return img
