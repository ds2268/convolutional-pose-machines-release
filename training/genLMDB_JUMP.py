import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import lmdb
import caffe
import os
import os.path
import sys
import struct
import random
from optparse import OptionParser
import json

# if True you can inspect each individual image and it's keypoint annotations
DEBUG = False

# LIMBS used for JUMP dataset
limbs = ['nose', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'right_hip', 'left_hip',
         'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_ski_f', 'right_ski_b', 'left_ski_f', 'left_ski_b']

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

    def __init__(self):
        self.dataset = 'NameOfTheDataset'
        self.height = 1080
        self.width = 1920
        self.isValidation = False
        self.peopleIndex = 1
        self.numOtherPeople = 0
        self.annolistIndex = 1  # image_id -> to get the corresponding image or annotations from COCO
        self.writeNumber = 1  # seq. number of the processed image with actual annotations
        self.bbox = []
        self.joints = []
        self.bboxOther = []
        self.jointsOther = []

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
                metaData[clidx][j] = ord(lineBinary[j])
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
                    metaData[clidx][j] = ord(lineBinary[j])
                clidx += 1

        return metaData

    def float2bytes(self, floats):
        if type(floats) is float:
            floats = [floats]
        return struct.pack('%sf' % len(floats), *floats)


    def getImageChannel(self):
        return 'hello world'

class JumpData(object):

    def __init__(self, data_dir, validation):
        self.data_dir = data_dir
        self.validation = validation

    def readData(self):
        data = []
        for root, dirs, files in os.walk(self.data_dir):
            if 'data.json' in files:
                data_json_path = os.path.join(root, 'data.json')
                with open(data_json_path) as data_json:
                    data_json = json.load(data_json)
                    for entry in data_json:
                        curr = data_json[entry]
                        image_name = '{:04d}.bmp'.format(int(entry))

                        validation = False
                        if 'validation' in curr and curr['validation']:
                            validation = True

                        data.append(self.Frame(entry, os.path.join(root, image_name), curr['rect'], curr['x'],
                                          curr['y'], curr['visible'], validation))

        return data


    class Frame(object):

        def __init__(self, id, file_path, rect, x, y, visible, validation):
            self.id = id
            self.file_path = file_path
            self.rect = rect
            self.keypoints = [item for sublist in zip(x ,y, visible) for item in sublist]
            self.validation = validation

def writeLMDB_COCO(data_dir, lmdb_path, validation):
    jump_data = JumpData(data_dir, validation)
    frame_data = jump_data.readData()

    if not validation:
        frame_data = list(filter(lambda x: not x.validation, frame_data))

    # initialize lmdb
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)

    # randomize order
    random.shuffle(frame_data)

    # process each image
    n_processed = 1
    person_id = 0
    for img_ann in frame_data:
        isVal = img_ann.validation
        img = cv2.imread(img_ann.file_path)

        height = img.shape[0]
        width = img.shape[1]
        if width < 68:
            img = cv2.copyMakeBorder(img, 0, 0, 0, 68 - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            print('[WARNING] Image was padded! Width < 68!')
            width = 68

        if not img_ann.keypoints:
            continue

        # Metadata object - ugly hack used to get data into Caffe Data layer as image 4-th channel
        metaData = MetaData()
        metaData.dataset = 'JUMP'
        metaData.height = height
        metaData.width = width
        metaData.isValidation = isVal
        metaData.annolistIndex = img_ann.id
        metaData.writeNumber = n_processed
        metaData.peopleIndex = person_id
        metaData.bbox = img_ann.rect
        metaData.joints = img_ann.keypoints
        metaDataChannel = metaData.saveMetaData()

        img4ch = np.concatenate((img, metaDataChannel), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        datum = caffe.io.array_to_datum(img4ch, label=0)
        txn.put('%012d_jump' % n_processed, datum.SerializeToString())

        if n_processed % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print('Saved next batch of 100 persons! Total: %d' % n_processed)

        if DEBUG:
            plt.figure()
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plotPose(img_ann)
            plt.waitforbuttonpress()
            plt.close()

        person_id += 1
        n_processed += 1

    txn.commit()
    env.close()

    print('Found total of: %d persons' % n_processed)

def plotPose(img_ann):
    ax = plt.gca()
    ax.add_patch(Rectangle((img_ann.rect[0], img_ann.rect[1]), img_ann.rect[2], img_ann.rect[3], linewidth=1, edgecolor='r', facecolor='none'))
    visible = img_ann.keypoints[2::3]
    points_x = list(filter(lambda x: x[1], zip(img_ann.keypoints[::3], visible, range(len(visible)))))
    points_y = list(filter(lambda x: x[1], zip(img_ann.keypoints[1::3], visible, range(len(visible)))))
    plt.plot(map(lambda x: x[0], points_x), map(lambda x: x[0], points_y), 'ro')

    for ((point_x, _, id), (point_y, _, _)) in zip(points_x, points_y):
        plt.text(point_x, point_y, limbs[id], fontsize=10, color='yellow', fontweight='bold')

if __name__ == "__main__":
    parser = OptionParser(usage="usage: python genLMDB_COCO.py --lmdb lmdbPath")
    parser.add_option("-l", "--lmdb", dest="lmdb", help="Path to store lmdb dataset", default="LMDB_COCO/")
    parser.add_option("-i", "--input", dest="input", help="Path to the dataset directory", default="dataset")
    (options, args) = parser.parse_args(sys.argv)

    inputDir = options.input
    lmdbDir = options.lmdb

    if not os.path.exists(lmdbDir):
        os.makedirs(lmdbDir)

    writeLMDB_COCO(inputDir, lmdbDir, False)
