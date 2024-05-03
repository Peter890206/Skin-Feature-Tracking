import re
import numpy as np
import torch
import cv2
import scipy.linalg
import random


def sorted_alphanumeric(data):
    """
    Sorts filenames in alphanumeric order.

    Args:
        data (list): List of filenames.

    Returns:
        list: Sorted list of filenames in alphanumeric order.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def convertRGB2CIELab(image):
    """
    Converts uint8 RGB image to float32 CIELAB format.

    Args:
        image (numpy.ndarray): RGB image in uint8 format.

    Returns:
        numpy.ndarray: CIELAB image in float32 format.
    """
    image = np.array(image).astype('float32')
    if np.max(image) > 1:
        image *= 1/255
    Lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    Lab_image[:,:,0]=Lab_image[:,:,0]/100 # Normalize L to be betweeen 0 and 1
    Lab_image[:,:,1]=(Lab_image[:,:,1]+127)/(2*127)
    Lab_image[:,:,2]=(Lab_image[:,:,2]+127)/(2*127)
    return Lab_image

def gauss_noise_tensor(img):
    """
    Adds Gaussian noise to a torch tensor image.

    Args:
        img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Image tensor with Gaussian noise added.
    """
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = random.uniform(15, 35)
    # sigma = 25
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

def get_pts_local_window(center,neighborhood=3):
    """
    Generates a local window of points around a center point.

    Args:
        center (list): Center point coordinates [x, y].
        neighborhood (int): Size of the neighborhood window, default is 3.

    Returns:
        numpy.ndarray: Local window of points in the shape (neighborhood, neighborhood, 2).
    """
    # 0, 1, 2,
    # 3, 4, 5,
    # 6, 7, 8
    rel_center=np.floor(neighborhood/2)
    window=np.zeros((neighborhood,neighborhood,2))
    for i in range(neighborhood):
        for j in range(neighborhood):
            window[i,j]=[(i-rel_center)+center[0],(j-rel_center)+center[1]]
    return window # pts are in (x,y)

def find_min_surface(A,B):
    """
    Finds the minimum surface by solving the least-squares problem AxC=B.

    Args:
        A (numpy.ndarray): Matrix A in the equation AxC=B.
        B (numpy.ndarray): Vector B in the equation AxC=B.

    Returns:
        tuple: A tuple containing:
            - C_star (numpy.ndarray): Minimum surface coordinates.
            - C (numpy.ndarray): Coefficients of the surface equation.
            - A (numpy.ndarray): Matrix A.
            - B (numpy.ndarray): Vector B.
    """
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
   
            return [[A[4,1],A[4,2]]],C,A,B #original center at pixel level
        elif D == 0:

            return [[A[4,1],A[4,2]]],C,A,B #original center at pixel level

    else:
        print('Can not do surface fitting of order 2')
    return np.swapaxes(C_star,0,1),np.array(C),A,B

def quadratic_interpolation_deep_encodings(center, window_pts, ssr):
    """
    Performs quadratic interpolation on deep encodings.

    Args:
        center (list): Center point coordinates [x, y].
        window_pts (numpy.ndarray): Window points around the center.
        ssr (numpy.ndarray): Sum of squared residuals for each window point.

    Returns:
        tuple: A tuple containing:
            - pred (numpy.ndarray): Predicted coordinates.
            - C (numpy.ndarray): Coefficients of the surface equation.
            - A (numpy.ndarray): Matrix A.
            - B (numpy.ndarray): Vector B.
    """
    window_pts[:,0]-=center[0]
    window_pts[:,1]-=center[1]
    A=np.c_[(np.ones(ssr.shape),window_pts[:,0],window_pts[:,1],np.prod(window_pts,axis=1),
                window_pts[:,0]**2,window_pts[:,1]**2)]                        
    pred, C, A, B=find_min_surface(A,ssr)
    if not type(C) is np.ndarray: ## for the local maximum, check find_min_surface
        return pred, np.NaN, A, B
    pred=np.clip(pred,-1,1)+center
    return pred, C, A, B

def predict(model, data_loader, device = "cuda"):
    """
    Performs predictions using a trained model on a data loader.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): Data loader containing input data.
        device (str, optional): Device to use for predictions (default: "cuda").

    Returns:
        torch.Tensor: Predicted outputs.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            batch_predictions = model(batch)
            predictions.append(batch_predictions)
            
    predictions = torch.cat(predictions, dim=0)

    return predictions

# """
# Convert RGB image to LAB image using kornia (not working)
# """
# class RGBtoLAB(object):
#     def __call__(self, img):
#         img = img.float() / 255
#         lab_img = kornia.color.rgb_to_lab(img)
#         # print(lab_img.shape)
#         # l = lab_img[0]/50.0 - 1.0
#         # a = lab_img[1]/110.0
#         # b = lab_img[2]/110.0
#         lab_img[0]=lab_img[0]/100 # Normalize L to be betweeen 0 and 1
#         lab_img[1]=(lab_img[1]+127)/(2*127)
#         lab_img[2]=(lab_img[2]+127)/(2*127)
        
#         # l = torch.unsqueeze(l, dim=-1)
#         # a = torch.unsqueeze(a, dim=-1)
#         # b = torch.unsqueeze(b, dim=-1)
#         # print("l = ", l.shape)
#         # print("a = ", a.shape)
#         # print("b = ", b.shape)
#         # lab_img = lab_img.permute(2, 1, 0)
#         # lab_img = torch.cat([l, a, b], dim=-1).permute(2, 0, 1)
#         # print(lab_img.shape)
#         return lab_img
