% Function to initialize haar-like features.
fine_train_init;

% Pre-allocation for random forest models.
AO = cell(1,32);

% Loop that creates a model for each fiducial.
for g = 1:32

% Pre-allocating variables.
finalpredarr = [];
mu = [61,61,61];

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
out = imresize3(out,1);
ijk_arr = round(ijk_arr);
out = padarray(out,[30,30,30],0,'both');
ijk_arr = [ijk_arr(1:3,:) + 30;ones(1,32)];

% Extracting space around current fiducial.
out = out(ijk_arr(1,g)-60:ijk_arr(1,g)+60,ijk_arr(2,g)-60:ijk_arr(2,g)+60,ijk_arr(3,g)-60:ijk_arr(3,g)+60); 

% Retreiving central coordinates.
coordsinit = combvec(58:64,58:64,58:64);
coords = coordsinit';

answerarr = [];

% Entire loop extract haar-like features for central coordinates. These
% features are randomly chosen in the extracted space around the current
% fiducial.
for i = 1:size(coords,1)
    
    conarr = [];
    farr = [];
    saved = out(coords(i,1)-40:coords(i,1)+40,coords(i,2)-40:coords(i,2)+40,coords(i,3)-40:coords(i,3)+40);
    SS = integralImage3(saved);
    
    for k = 1:1000
        
        savedsample = savedsamplearr(w(k)).s;
        idxa = savedsamplearr(w(k)).idxa;
        idxb = savedsamplearr(w(k)).idxb;
        coordsa = savedsample(idxa(k),:);
        coordsb = savedsample(idxb(k),:);
        
        [sR, sC, sP, eR, eC, eP] = deal(coordsa(1)-w(k)+1, coordsa(2)-w(k)+1, coordsa(3)-w(k)+1, coordsa(1), coordsa(2), coordsa(3));
        [sR2, sC2, sP2, eR2, eC2, eP2] = deal(coordsb(1)-w(k)+1, coordsb(2)-w(k)+1, coordsb(3)-w(k)+1, coordsb(1), coordsb(2), coordsb(3));
        
        regionSum = SS(eR+1,eC+1,eP+1) - SS(eR+1,eC+1,sP) - SS(eR+1,sC,eP+1) ...
        - SS(sR,eC+1,eP+1) + SS(sR,sC,eP+1) + SS(sR,eC+1,sP) ... 
        + SS(eR+1,sC,sP) -SS(sR,sC,sP);
        regionSum2 = SS(eR2+1,eC2+1,eP2+1) - SS(eR2+1,eC2+1,sP2) - SS(eR2+1,sC2,eP2+1) ...
        - SS(sR2,eC2+1,eP2+1) + SS(sR2,sC2,eP2+1) + SS(sR2,eC2+1,sP2) ... 
        + SS(eR2+1,sC2,sP2) -SS(sR2,sC2,sP2);        
        
        f = regionSum - regionSum2;
        farr = [farr,f];

    end
    
    con = farr;
    conarr = [conarr,con];
    answerarr = [answerarr;conarr];
    
end

% Extracts SIFT features from central coordinates.
keys = keypoint3D([coords(:,1:3)],repelem(0.5,size(coords,1)));
[desc, location] = extractSift3D(keys,out,units);

% Sets response variable for each of the central coordinates.
X = combvec(1:121,1:121,1:121)';
y = normpdf(X,mu,[8,8,8]);
ysum = sum(y,2);
normysum = (ysum-min(ysum))/(max(ysum)-min(ysum));
[tf, index]=ismember(location,X,'rows');
normysum2 = normysum(index);
normysum2 = (normysum2-min(normysum2))/(max(normysum2)-min(normysum2));

% Concatenates features to array.
finalpred = [answerarr,answerarr,repelem(t,size(answerarr,1))'];
finalpredarr = [finalpredarr; finalpred];
end

% Training to build model.
Mdl = TreeBagger(100,finalpredarr(:,1:end-2),finalpredarr(:,end-1),'Method','regression');
AO{g} = Mdl;
end

