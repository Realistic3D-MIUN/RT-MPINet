'''
Author: Manu Gond (manu.gond@miun.se)
Date: Nov-15-2022
Objective:  Accumulation of some general functions which I
            use daily in my code realted to image relasted task.
            The function names and parameters are self explanetory.
Requirements: Installed python libraries which have been imported.
'''

import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
import torchmetrics
import cv2
import numpy as np
from PIL import Image
import utils


#======================= Read and Write =====================#
def readImage(location):
    image = Image.open(location).convert("RGB")
    return image


def writeImage(image, location):
    image.save(location)


def writeTensorImage(image, filename):
    save_image(image, filename)


def removeChannel(sourceLocation, targetLocation):
    img = readImage(sourceLocation)
    writeImage(img, targetLocation)


def getImageTransform(width, height):
    transform = transforms.Compose([transforms.Resize((height,width)),
                                    transforms.ToTensor()])
    return transform


def convertTensor(image):
    transform = getImageTransform(image.size[0], image.size[1])
    image = transform(image)
    return image


#=================== 360 Images =======================#

def rotateERP180(image):
    '''
    :param image: PIL Image
    :return: BxHxW Torch Tensor Image
    '''
    W = image.size[0]
    H = image.size[1]
    transform = getImageTransform(W, H)
    image = transform(image)
    image1 = image[:, :, 0:(W//2)]
    image2 = image[:, :, (W//2):W]
    image3 = torch.zeros(image.size())
    image3[:, :, 0:(W//2)] = image2
    image3[:, :, (W//2):W] = image1
    return image3


def convertERP2Cube(e_img, face_w=256, mode='bilinear', cube_format='dice'):
    '''
        e_img:  ndarray in shape of [H, W, *]
        face_w: int, the length of each face of the cubemap
        '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = utils.xyzcube(face_w)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_h2dice(cubemap)
    else:
        raise NotImplementedError()
    return cubemap


def convertCube2ERP(cubemap, h, w, mode='bilinear', cube_format='dice'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = utils.equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        utils.sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)
    return equirec



def convertCube2Slices(image):
    '''
    :param image: Image numpy array
    :return: List of Torch Tensors, CxHxW
    '''
    image = convertTensor(image)
    C, H, W = image.size()
    #print(C,H,W)
    top = torch.zeros((C,W//4,W//4))
    left = torch.zeros(top.size())
    front = torch.zeros(top.size())
    right = torch.zeros(top.size())
    back = torch.zeros(top.size())
    bottom = torch.zeros(top.size())

    top = image[:, 0:H//3, (W//4):(W//4)*2]
    left = image[:, (H//3):(H//3)*2, 0:W//4]
    front = image[:, (H//3):(H//3)*2, (W//4):(W//4)*2]
    right = image[:, (H//3):(H//3)*2, (W//4)*2:(W//4)*3]
    back = image[:, (H // 3):(H // 3) * 2, (W // 4) * 3:]
    bottom = image[:, (H//3)*2:, (W//4):(W//4)*2]

    '''
        save_image(top, 'top.png')
        save_image(left, 'left.png')
        save_image(front, 'front.png')
        save_image(right, 'right.png')
        save_image(back, 'back.png')
        save_image(bottom, 'bottom.png')
    '''
    return [top, left, front, right, back, bottom]

def convertSlicesToCube(imageList):
    '''
    top = convertTensor(readImage(imageList[0]))
    left = convertTensor(readImage(imageList[1]))
    front = convertTensor(readImage(imageList[2]))
    right = convertTensor(readImage(imageList[3]))
    back = convertTensor(readImage(imageList[4]))
    bottom = convertTensor(readImage(imageList[5]))
    '''
    top = imageList[0]
    left = imageList[1]
    front = imageList[2]
    right = imageList[3]
    back = imageList[4]
    bottom = imageList[5]

    C, H, W = 3,  top.size()[1]*3, top.size()[2]*4
    cube = torch.zeros((C, H, W))

    cube[:, 0:H//3, (W//4):(W//4)*2] = top
    cube[:, (H // 3):(H // 3) * 2, 0:W // 4] = left
    cube[:, (H // 3):(H // 3) * 2, (W // 4):(W // 4) * 2] = front
    cube[:, (H // 3):(H // 3) * 2, (W // 4) * 2:(W // 4) * 3] = right
    cube[:, (H // 3):(H // 3) * 2, (W // 4) * 3:] = back
    cube[:, (H // 3) * 2:, (W // 4):(W // 4) * 2] = bottom

    return cube



#=================== Quality Measures =======================#
'''
Predicted Shape : BxCxHxW
Original Shape  : BxCxHxW
Data Type: Torch Tensor
'''
def getSSIM(predicted, original):
    SSIM = torchmetrics.StructuralSimilarityIndexMeasure()
    return SSIM(predicted, original).item()


def getPSNR(predicted, original):
    PSNR = torchmetrics.PeakSignalNoiseRatio()
    return PSNR(predicted, original).item()


def getMSE(predicted, original):
    MSE = torchmetrics.MeanSquaredError()
    return MSE(predicted, original).item()


def getMAE(predicted, original):
    MAE = torchmetrics.MeanAbsoluteError()
    return MAE(predicted, original).item()



if __name__ == "__main__":

    '''
    img = readImage("31_image_0_0.png")
    img = convertERP2Cube(e_img=np.asarray(img), face_w=256)
    img = Image.fromarray(img.astype('uint8'),'RGB')
    convertCube2Slices(img)
    '''
    #image = convertSlicesToCube(["top.png", "left.png", "front.png", "right.png", "back.png", "bottom.png"])
    #writeTensorImage(image,'this.png')

    '''
    writeImage(img, 'cube.png')

    img = readImage('cube.png')
    img = convertCube2ERP(np.asarray(img),512,1024)
    img = Image.fromarray(img.astype('uint8'),'RGB')
    writeImage(img, 'cubeERP.png')


    img1 = readImage("31_image_0_0.png")
    img2 = readImage("cubeERP.png")
    img1 = convertTensor(img1)
    img2 = convertTensor(img2)
    print(getSSIM(img1.unsqueeze(0), img2.unsqueeze(0)))
    '''

    #img = rotateERP180(img)
    #writeTensorImage(img, 'rotated_image.png')
    #img = convertTensor(img)
    #print(getMAE(img.unsqueeze(0),img.unsqueeze(0)))




