% Pre-allocation for random forest models.
AO = cell(1,32);

% Loop that creates a model for each fiducial.
for g = 1:32

% Pre-allocating variables.
finalpredarr = [];
mu = [13,13,13];

cd C:\Users\Spiny\Autofid\OAS1_bids_MR1\fcsvs\train
files = dir;

% Looping through every training image.
for t = 3:size(files,1)

cd C:\Users\Spiny\Autofid\OAS1_bids_MR1\imgs\train
scan = dir;
cd C:\Users\Spiny\Autofid
[~, units] = imRead3D(scan(t).name);
units = units';
nii = load_nii(scan(t).name);
niimeta = nii;
nii = nii.img;

% Point transformed from slicer3D RAS format to matlab matrix format. First
% by retrieving fiducial metadata.
cd C:\Users\Spiny\Autofid\OAS1_bids_MR1\fcsvs\train
files = dir;
allC = [];
for i = t
fileID = fopen(files(i).name);
C = textscan(fileID,'%s %f %f %f %f %f %f %f %f %f %f %s %s %s', 'Delimiter', ',','HeaderLines', 3);

% retrieving RAS coordinates for fiducials, and adding homogenous
% coordinates to prepare for matrix transform.
C_array = cell2mat(C(2:4));
[m, n] = size(C_array);
if m ~= 32
    continue
end
C_array = [C_array ones(m, 1)];
allC = [allC; C_array];
fclose(fileID);
end
cd C:\Users\Spiny\Autofid

% retrieving volume origin and voxel dimension info from template brain
% scan for matrix calculations.
ori_arr = [niimeta.hdr.hist.qoffset_x; niimeta.hdr.hist.qoffset_y; niimeta.hdr.hist.qoffset_z];
spa_arr = [niimeta.hdr.dime.pixdim(2), niimeta.hdr.dime.pixdim(3), niimeta.hdr.dime.pixdim(4)];

% calculating transformation matrix.
fourbyfour = inv([diag(spa_arr) ori_arr; 0 0 0 1]);

% outputing fiducial coordinates in matlab matrix format.
ijk_arr = [];
for i=1:size(allC,1)
    ijk = fourbyfour*allC(i,:)';
    ijk_arr = [ijk_arr ijk];
end
ijk_arr = [ijk_arr(1,:)+size(nii,1); ijk_arr(2,:); ijk_arr(3,:); ijk_arr(4,:)];

% Preprocessing. Resizing and grayscale conversion of new scan.
Im = reshape(nii,size(nii,1),[]);
out = mat2gray(Im);
out = imadjust(out,[]);
out = reshape(out,size(nii,1),size(nii,2),[]);
ijk_arr = round(ijk_arr);
out = imresize3(out,0.25);
ijk_arr = round(ijk_arr/4);
out = padarray(out,[30,30,30],0,'both');
ijk_arr = [ijk_arr(1:3,:) + 30;ones(1,32)];

% Extracting space around current fiducial.
out = out(ijk_arr(1,g)-15:ijk_arr(1,g)+15,ijk_arr(2,g)-15:ijk_arr(2,g)+15,ijk_arr(3,g)-15:ijk_arr(3,g)+15); 

% Retreiving coordinates of interest.
coordsinit = combvec(13:19,13:19,13:19);
coords = coordsinit';

% Extracts SIFT features from central coordinates.
keys = keypoint3D(coords(:,1:3),repelem(0.5,size(coords,1))'); 
[desc, location] = extractSift3D(keys,out,units);

% Sets response variable for each of the central coordinates.
X = combvec(ijk_arr(1,g)-12:ijk_arr(1,g)+12,ijk_arr(2,g)-12:ijk_arr(2,g)+12,ijk_arr(3,g)-12:ijk_arr(3,g)+12)';
y = normpdf(X,mu,[2.5,2.5,2.5]);
ysum = sum(y,2);
normysum = (ysum-min(ysum))/(max(ysum)-min(ysum));
[tf, index]=ismember(location,X,'rows');
normysum2 = normysum(index);
normysum2 = (normysum2-min(normysum2))/(max(normysum2)-min(normysum2));

% Concatenates features to array.
finalpred = [desc, normysum(index)];
finalpredarr = [finalpredarr; finalpred];
end

% Training to build model.
Mdl = TreeBagger(100,finalpredarr(:,1:end-1),finalpredarr(:,end),'Method','regression');
AO{g} = Mdl;
end