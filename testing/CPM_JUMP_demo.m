clear all;
close all;

% This is basic code to test learned modell(JUMP) on example image
% using getKeypointsCOCO method, it also displays found keypoints.

% COCO limb names
% 0) nose
% 1) right_shoulder 3) right_elbow  5) right_wrist 7) right_hip 9) right_knee 11) right_ankle 13) right_ski_f 15) left_ski_f 
% 2) left_shoulder  4) left_elbow   6) left_wrist  8) left_hip 10) left_knee  12) left_ankle  14) right_ski_b 16) left_ski_b
labels = {'nose', 'rightShoulder', 'leftShoulder', 'rightElbow', 'leftElbow', 'rightWrist', 'leftWrist', 'rightHip', 'leftHip',...
          'rightKnee', 'leftKnee', 'rightAnkle', 'leftAnkle', 'rightSkiF', 'rightSkiB', 'leftSkiF', 'leftSkiB'};

% settings
param.img_path = '/media/hdd3000/jump_dataset/1/2/1/0590.bmp';
param.deploy_path = '../training/prototxt/COCO/pose_deploy.prototxt';
param.model_path = '../training/prototxt/COCO/caffemodel/pose_jump_only_iter_42000.caffemodel';
param.numParts = 17;
param.target_scale = 0.5;
param.sigma_center = 21;
param.boxSize = 368;
param.gpuId = 0;
param.padValue = 128;
param.threshVisible = 0.4;
param.scaleSearch = 0.7:0.2:1.3 ; % fit training
param.DEBUG = false;

% setup Caffe network
caffe.set_mode_gpu();
caffe.set_device(param.gpuId);
net = caffe.Net(param.deploy_path, param.model_path, 'test');

I = imread(param.img_path);

figure(1);
imshow(I);
hold on;

% select ROI and center of a person
rect_roi = getrect(1);
h = impoint();

% get keypoints
% same format as COCO 1x51 [x1, y1, v1,...]
% visibility always 1
keypoints = getKeypointsJUMP(I, rect_roi, getPosition(h), net, param);
caffe.reset_all()

x_all = keypoints(1:4:end);
y_all = keypoints(2:4:end);
visible = keypoints(3:4:end);
scores = keypoints(4:4:end)

for i=1:length(labels)
    if visible(i)
        plot(x_all(i), y_all(i), 'g*');
        text(double(x_all(i)), double(y_all(i)), labels(i), 'Color','red','FontSize', 14);
    end
end
