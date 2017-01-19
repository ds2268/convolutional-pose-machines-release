clear all;
close all;

% This is basic code to test learned model(COCO) on example image

% COCO limb names
% 0) nose
% 1) left_eye 3) left_ear  5) left_shoulder 7) left_elbow 9) left_wrist 11) left_hip 13) left_knee 15) left_ankle
% 2) right_eye 4) right_ear 6) right_shoulder 8) right_elbow 10) right_wrist 12) right_hip 14) right_knee 16) right_ankle
labels = {'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',...
          'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'};

% settings
param.img_path = 'sample_image/roger.png';
param.deploy_path = '../training/prototxt/COCO/pose_deploy.prototxt';
param.model_path = '../training/prototxt/COCO/caffemodel_trained_100k_00008/pose_iter_100000.caffemodel';
param.target_scale = 0.8;
param.sigma_center = 21;
param.boxSize = 368;
param.gpuId = 0;
param.padValue = 128;
param.DEBUG = false;

I = imread(param.img_path);

figure(1);
imshow(I);
hold on;

% select ROI
rect_roi = getrect(1);

% get keypoints
keypoints = getKeypointsCOCO(I, rect_roi, param);

% plot keypoints
for i=1:numel(keypoints)
    plot(keypoints{i}(2), keypoints{i}(1), 'g*')
    hold on;
end
