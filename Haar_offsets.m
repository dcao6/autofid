%% Haar Offset Script

% This script is what was used to generate the offset blocks, used commonly
% in tasks requiring Haar-like features. Blocks of size 1x1x1 voxels to
% 3x3x3 voxels are created at random offset positions within 17 voxels of
% the target position. A total of 4000 blocks are created, and feature
% vectors of size 1x2000 are generated to fit each voxel being described.
% Offset .mat files have already been created, so running this script is
% not necessary (script here is just for reference).

% Coarse offsets
xmin = randsample(-17:14,4000,true)';
ymin = randsample(-17:14,4000,true)';
zmin = randsample(-17:14,4000,true)';
smin = [xmin,ymin,zmin];

larr = [];
warr = [];
darr = [];

for i = 1:4000
l = randsample(1:3,1,true);
w = randsample(1:3,1,true);
d = randsample(1:3,1,true);
larr = [larr;l];
warr = [warr;w];
darr = [darr;d];
end

smax = smin + [larr,warr,darr];
save('coarse_haaroffset.mat','smin','smax')

% Med offsets
xmin = randsample(-17:14,4000,true)';
ymin = randsample(-17:14,4000,true)';
zmin = randsample(-17:14,4000,true)';
smin = [xmin,ymin,zmin];

larr = [];
warr = [];
darr = [];

for i = 1:4000
l = randsample(1:3,1,true);
w = randsample(1:3,1,true);
d = randsample(1:3,1,true);
larr = [larr;l];
warr = [warr;w];
darr = [darr;d];
end

smax = smin + [larr,warr,darr];
save('med_haaroffset.mat','smin','smax')

% Fine offsets
xmin = randsample(-17:14,4000,true)';
ymin = randsample(-17:14,4000,true)';
zmin = randsample(-17:14,4000,true)';
smin = [xmin,ymin,zmin];

larr = [];
warr = [];
darr = [];

for i = 1:4000
l = randsample(1:3,1,true);
w = randsample(1:3,1,true);
d = randsample(1:3,1,true);
larr = [larr;l];
warr = [warr;w];
darr = [darr;d];
end

smax = smin + [larr,warr,darr];
save('fine_haaroffset.mat','smin','smax')