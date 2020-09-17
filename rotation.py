import numpy as np
from os import listdir
import random
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt 
from tqdm import  tqdm

def get_true_bboxes(labelFile,W,H):

    file=open(labelFile)
    lines=file.read()
    file.close()

    cars=lines.split('\n')
    bboxes=[]
    for i in range(len(cars)-1):
        point=cars[i].split(' ')[1:5]
        xcen,ycen,w,h=float(point[0]),float(point[1]),float(point[2]),float(point[3])
        x1,y1=int((xcen-w/2)*W),int((ycen-h/2)*H)
        x2,y2=int((xcen+w/2)*W),int((ycen+h/2)*H)
        bboxes.append([x1,y1,x2,y2])

    return bboxes

def save_rotated_txt_file(bboxes,location,angle,imgName):

    filename=location+'labels/'+imgName[:-4]+'_'+str(angle)+'.txt'
    with open(filename,'w') as f:
        for l in range(len(bboxes)):
            s=' '
            s=s.join([str(int(elem)) for elem in bboxes[l]])
            s=s+'\n'
            f.write(s)
        f.close()
    


imgfileLoc='./images/'
lblfileLoc='./labels/'
location='./outputs/'
imgFiles=listdir(imgfileLoc)
lblFiles=listdir(lblfileLoc)

for k in tqdm(range(len(imgFiles))):

    imgFile=imgfileLoc+imgFiles[k]
    labelFile=lblfileLoc+lblFiles[k]

    img=Image.open(imgFile)
    W,H=img.size
    img=np.array(img)
    true_bboxes=get_true_bboxes(labelFile,W,H)
    angles=range(0,360,20)
    angle=random.choice(angles)
    seq=iaa.Sequential([iaa.geometric.Affine(rotate=angle)])
    image_aug=seq(image=img)
    rotatedImg=Image.fromarray(image_aug)
    rotatedImgName=location+'images/'+imgFiles[k][:-4]+'_'+str(angle)+'.jpg'
    rotatedImg.save(rotatedImgName)
    rotated_bboxes=[]
    for i in range(len(true_bboxes)):

        xx1,yy1,xx2,yy2=true_bboxes[i]
        bbs = BoundingBoxesOnImage([BoundingBox(x1=xx1, y1=yy1, x2=xx2, y2=yy2)], shape=(W,H))
        bbs_aug = seq(bounding_boxes=bbs)
        after = bbs_aug.bounding_boxes[0]
        rotated_bboxes.append([after.x1, after.y1,after.x2, after.y2])

    save_rotated_txt_file(rotated_bboxes,location,angle,imgFiles[k])


