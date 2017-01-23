clear all;
close all;

% This is basic code to test learned model(COCO) on example image
% using getKeypointsCOCO method, it also displays found keypoints.

% COCO limb names
% 0) nose
% 1) left_eye 3) left_ear  5) left_shoulder 7) left_elbow 9) left_wrist 11) left_hip 13) left_knee 15) left_ankle
% 2) right_eye 4) right_ear 6) right_shoulder 8) right_elbow 10) right_wrist 12) right_hip 14) right_knee 16) right_ankle
labels = {'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',...
          'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'};

% settings
param.img_path = 'sample_image/nadal.png';
param.deploy_path = '../training/prototxt/COCO/pose_deploy.prototxt';
param.model_path = '../training/prototxt/COCO/caffemodel/pose_train_iter_100000.caffemodel';
param.target_scale = 0.8;
param.sigma_center = 21;
param.boxSize = 368;
param.gpuId = 0;
param.padValue = 128;
param.scaleSearch = 0.7:0.1:1.3; % fit training
param.DEBUG = false;

% setup Caffe network
caffe.set_mode_gpu();
caffe.set_device(param.gpuId);
net = caffe.Net(param.deploy_path, param.model_path, 'test');

I = imread(param.img_path);

figure(1);
imshow(I);
hold on;

% select ROI
rect_roi = getrect(1);

% get keypoints
% same format as COCO 1x51 [x1, y1, v1,...]
% visibility always 1
keypoints = getKeypointsCOCO(I, rect_roi, net, param);
caffe.reset_all()

x_all = keypoints(1:3:end);
y_all = keypoints(2:3:end);
plot(x_all, y_all, 'g*');
