import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import caffe
import os.path
import struct
import random
from pycocotools.coco import COCO

# if True you can inspect each individual image and it's keypoint annotations (COCO)
DEBUG = False

# This class contains data that is saved to separate image channel as meta-data
'''
Format of the data saved in 4-th image channel
----------------------------------------------
dataset (string)
height (float) width (float)
isValidation (uint8) numOtherPeople (uint8) peopleIndex (uint8) annolistIndex (float) writeNumber (float)
bbox_x (float) bbox_y (float) bbox_width (float) bbox_height(float)
joint_x(1) joint_x(2) ... joint_x(17) |-> all floats
joint_y(1) joint_y(2) ... joint_y(17) |-> all floats
joint_visible(1) joint_visible(2) ... joint_visible(17) |-> all floats
bbox_x_other(1) bbox_y_other(1) bbox_width_other(1) bbox_height_other(1) |-> all floats
joint_x_other(1,1) joint_x_other(1,2) ... joint_x_other(1,17) |-> all floats
joint_y_other(1,1) joint_y_other(1,2) ... joint_y_other(1,17) |-> all floats
joint_visible_other(1,1) joint_visible_other(1,2) ... joint_visible_other(1,17) |-> all floats
...
...
bbox_x_other(n) bbox_y_other(n) bbox_width_other(n) bbox_height_other(n) |-> all floats
joint_x_other(n,1) joint_x_other(n,2) ... joint_x_other(n,17) |-> all floats
joint_y_other(n,1) joint_y_other(n,2) ... joint_y_other(n,17) |-> all floats
joint_visible_other(n,1) joint_visible_other(n,2) ... joint_visible_other(n,17) |-> all floats
'''
class MetaData:
    dataset = 'NameOfTheDataset'
    height = 1080
    width = 1920
    isValidation = False
    peopleIndex = 1
    numOtherPeople = 1
    annolistIndex = 1   # image_id -> to get the corresponding image or annotations from COCO
    writeNumber = 1     # seq. number of the processed image with actual annotations
    bbox = []
    joints = []
    bboxOther = [[]]
    jointsOther = [[]]

    def saveMetaData(self):
        metaData = np.zeros(shape=(self.height, self.width, 1), dtype=np.uint8)
        clidx = 0

        # dataset(string)
        for i in range(len(self.dataset)):
            metaData[clidx][i] = ord(self.dataset[i])
        clidx += 1

        # height(float) and width(float)
        heightBinary = self.float2bytes(float(self.height))
        widthBinary = self.float2bytes(float(self.width))
        for i in range(len(heightBinary)):
            metaData[clidx][i] = ord(heightBinary[i])

        for i in range(len(widthBinary)):
            metaData[clidx][4+i] = ord(widthBinary[i])
        clidx += 1

        # isValidation(uint8), numOtherPeople(uint8), peopleIndex(uint8), annoListIndex(float), writeNumber(float)
        metaData[clidx][0] = self.isValidation
        metaData[clidx][1] = self.numOtherPeople
        metaData[clidx][2] = self.peopleIndex
        annoListIndexBinary = self.float2bytes(float(self.annolistIndex))
        writeNumberBinary = self.float2bytes(float(self.writeNumber))

        for i in range(len(annoListIndexBinary)):
            metaData[clidx][3+i] = ord(annoListIndexBinary[i])

        for i in range(len(writeNumberBinary)):
            metaData[clidx][7+i] = ord(writeNumberBinary[i])
        clidx += 1

        # bbox: x(float), y(float), width(float), height(float)
        bboxBinary = self.float2bytes(self.bbox)
        for i in range(len(bboxBinary)):
            metaData[clidx][i] = ord(bboxBinary[i])
        clidx += 1

        # joints (float) 3x17 (row1: x, row2: y, row3: visibility)
        for i in range(3):
            # every third (coco 51d vector) - separate x,y,visibility
            lineBinary = self.float2bytes(self.joints[i::3])
            for j in range(len(lineBinary)):
                metaData[clidx] = ord(lineBinary[j])
            clidx += 1

        # process other people
        for p_other in range(self.numOtherPeople):
            # bbox: x(float), y(float), width(float), height(float)
            bboxBinary = self.float2bytes(self.bboxOther[p_other])
            for i in range(len(bboxBinary)):
                metaData[clidx][i] = ord(bboxBinary[i])
            clidx += 1

            # joints (float) 3x17 (row1: x, row2: y, row3: visibility)
            for i in range(3):
                # every third (coco 51d vector) - separate x,y,visibility
                lineBinary = self.float2bytes(self.jointsOther[p_other][i::3])
                for j in range(len(lineBinary)):
                    metaData[clidx] = ord(lineBinary[j])
                clidx += 1

        return metaData

    def float2bytes(self, floats):
        if type(floats) is float:
            floats = [floats]
        return struct.pack('%sf' % len(floats), *floats)


    def getImageChannel(self):
        return 'hello world'

def writeLMDB_COCO(data_dir, lmdb_path, validation):
    # initialize lmdb
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)

    # prepare paths to annotation files
    dataTypeTrain = 'train2014'
    annFileTrain = '%s/annotations/person_keypoints_%s.json' % (data_dir, dataTypeTrain)
    dataTypeVal = 'val2014'
    annFileVal = '%s/annotations/person_keypoints_%s.json' % (data_dir, dataTypeVal)

    # initialize COCO api for keypoints annotations
    cocoTrain = COCO(annFileTrain)
    if validation:
        cocoVal = COCO(annFileVal)

    # get all image annotations that contains persons
    personId = cocoTrain.getCatIds(catNms=['person'])
    imgIdsTrain = cocoTrain.getImgIds(catIds=personId)
    imgsTrain = cocoTrain.loadImgs(imgIdsTrain)
    imgsVal = []

    if validation:
        imgIdsVal = cocoVal.getImgIds(catIds=personId)
        imgsVal = cocoVal.loadImgs(imgIdsVal)

    # merge datasets and randomize order
    imgs_merged = imgsTrain + imgsVal
    random.shuffle(imgs_merged)

    # process each image
    n_processed = 1
    n_people = 0
    for img_ann in imgs_merged:
        isVal = False
        if dataTypeTrain in img_ann['file_name']:
            img = cv2.imread('%s/images/%s/%s' % (data_dir, dataTypeTrain, img_ann['file_name']))
            coco = cocoTrain
        elif dataTypeVal in img_ann['file_name']:
            img = cv2.imread('%s/images/%s/%s' % (data_dir, dataTypeVal, img_ann['file_name']))
            isVal = True
            coco = cocoVal
        else:
            print('[WARNING] ' + dataTypeTrain + ' or ' + dataTypeVal + ' not found in annotations filename')

        height = img.shape[0]
        width = img.shape[1]
        if width < 64:
            img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            print('[WARNING] Image was padded! Width < 64!')
            width = 64

        # get annotations for keypoints
        annIds = coco.getAnnIds(imgIds=img_ann['id'], catIds=personId)
        ann_persons = coco.loadAnns(annIds)

        # Metadata object - ugly hack used to get data into Caffe Data layer as image 4-th channel
        metaData = MetaData()
        metaData.dataset = 'COCO'
        metaData.height = height
        metaData.width = width
        metaData.isValidation = isVal
        metaData.annolistIndex = img_ann['id']
        metaData.writeNumber = n_processed

        person_id = 0
        for person in ann_persons:
            # check if any keypoints at all
            if person['num_keypoints'] > 0:
                person_id += 1
            else:
                continue

            if person_id == 1:
                metaData.peopleIndex = person_id
                metaData.bbox = person['bbox']
                metaData.joints = person['keypoints']
            else:
                metaData.bboxOther.append(person['bbox'])
                metaData.jointsOther.append(person['keypoints'])

        # check if any annotated persons at all
        if person_id > 0:
            metaData.numOtherPeople = person_id - 1
            metaDataChannel = metaData.saveMetaData()
            img4ch = np.concatenate((img, metaDataChannel), axis=2)
            img4ch = np.transpose(img4ch, (2, 0, 1))
            datum = caffe.io.array_to_datum(img4ch, label=0)
            txn.put(str(metaData.annolistIndex), datum.SerializeToString())

            n_processed += 1
            n_people += person_id

            if n_processed % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print('Saved next batch of 1000 images!')

        if DEBUG:
            plt.figure()
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            coco.showAnns(ann_persons)
            plt.waitforbuttonpress()
            plt.close()

    txn.commit()
    env.close()

    print('Found total of: %d persons' % n_people)

if __name__ == "__main__":
    cocoDir = '../dataset/COCO/'
    lmdbDir = '/media/ssd250/COCO_LMDB'

    if not os.path.exists(lmdbDir):
        os.makedirs(lmdbDir)

    writeLMDB_COCO(cocoDir, lmdbDir, True)
