close all;
clear;
clc;
run matlab\vl_setupnn.m;

load('CWAN_L.mat')
load('CWAN_AB.mat')

im1 = imread('im_lowLight.png');
im2 = imread('im_GT.png');

res = CWAN(im1,netCWANL,netCWANAB);
p = psnr(res,im2);
[s,~]  = ssim(res,im2);
