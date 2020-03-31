%% Med Regression Forest Training Script

% Script used to train a regression forest for each fiducial marker.

% Need a NIFTI image file reader (not available in older versions of
% MATLAB). Download from https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% and add to path. ie. addpath('Your path to NIfTI_tools').
addpath('C:\Users\danie\Autofid\NIfTI_tools')

% Loop that creates a model for each fiducial.
for g = 1:32

    finalpredarr = [];
    
    % Moves to directory containing training image fiducials. Manual
    % fiducials were created for 25 brains in the OASIS-1 dataset. See
    % .fcsv files for a more detailed look at the training labels.
    %%% NOTE: For the training dataset, all brains were initially rigidly
    %%% registered to MNI space. These .fcsv files correspond to images
    %%% that have already been registered.
    cd C:\Users\danie\Autofid\OAS1_dataset\train\fcsvs\preprocessed\aligned_only
    files = dir;

    % Looping through every training image.
    for t = 3:size(files,1)
        
        % Moves to directory containing training images. In this case, the
        % OASIS-1 dataset was used (dataset: 25 brains used for training and
        % validation). Adjust the directory after 'cd' to redirect to OASIS-1
        % dataset in your computer.
        %%% NOTE: For the training dataset, all brains were initially rigidly
        %%% registered to MNI space.
        cd C:\Users\danie\Autofid\OAS1_dataset\train\imgs\preprocessed\aligned_only
        
        scan = dir;
        niimeta = load_nii(scan(t).name);
        nii = niimeta.img;

        % Point transformed from slicer3D RAS format to matlab matrix format. First
        % by retrieving fiducial metadata.
        cd C:\Users\danie\Autofid\OAS1_dataset\train\fcsvs\preprocessed\aligned_only
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
        
        % Return to parent directory (change directory after 'cd').
        cd C:\Users\danie\Autofid

        % Setting ground-truth fiducial locations in an array.
        allC = allC(:,1:3) - [niimeta.hdr.hist.qoffset_x,niimeta.hdr.hist.qoffset_y,niimeta.hdr.hist.qoffset_z];
        ijk_arr = [allC(:,1) + size(nii,1), allC(:,2), allC(:,3)];
        ijk_arr = ijk_arr';
        ijk_arr = [ijk_arr; ones(1, size(ijk_arr,2))];

        % Preprocessing. Resizing and grayscale conversion of new scan.
        nii = single(nii);
        nii = (nii-min(min(min(nii))))*(1)/(max(max(max(nii)))-min(min(min(nii))));
        out = nii;

        % Training at med resolution, so image is just the image as is.
        ijk_arr = round(ijk_arr);
        out = padarray(out,[50,50,50],0,'both');
        ijk_arr = [ijk_arr(1:3,:) + 50;ones(1,32)];

        if any(ijk_arr(1:3,g) < 60) == 1 
            continue
        end
        
        % Creates integral image from patch.
        patch = out(ijk_arr(1,g)-60:ijk_arr(1,g)+60,ijk_arr(2,g)-60:ijk_arr(2,g)+60,ijk_arr(3,g)-60:ijk_arr(3,g)+60);
        patch = (patch-min(min(min(patch))))*(1)/(max(max(max(patch)))-min(min(min(patch))));
        J = integralImage3(patch);

        % Points used for training include those voxels in a 11x11x11 sized
        % area around the ground truth, as well as every other voxel
        % outside of the 11x11x11 block but inside a 21x21x21 block around
        % the ground truth.
        coords1 = combvec(60-5:60+5,60-5:60+5,60-5:60+5);
        coords2 = combvec(60-10:2:60+10,60-10:2:60+10,60-10:2:60+10);
        coords = [coords1,coords2];
        coords = coords';
        coords = unique(coords,'rows');
        
        % Load predefined Haar-block offsets.
        load('med_haaroffset.mat')
        
        % Initialize some variables.
        mincornerlist = zeros(2537*4000,3);
        maxcornerlist = zeros(2537*4000,3);
        testerarr = zeros(2537*4000,1);
        for i = 1:size(coords,1)
            mincorner = coords(i,:) + smin;
            maxcorner = coords(i,:) + smax;
            mincornerlist((i-1)*4000+1:i*4000,:) = mincorner;
            maxcornerlist((i-1)*4000+1:i*4000,:) = maxcorner;
        end
        cornerlist = [mincornerlist,maxcornerlist];
        
        % Using the integral image to compute Haar-like features.
        for i = 1:2537*4000
            testerarr(i) = (J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,6)+1)-J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,3)) - J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,6)+1) ...
            - J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,3)) ...
            + J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,3)) - J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,3)))/((cornerlist(i,4)-cornerlist(i,1)+1)*(cornerlist(i,5)-cornerlist(i,2)+1)*(cornerlist(i,6)-cornerlist(i,3)+1));
        end
        vectorarr1 = [];
        vectorarr2 = [];
        for test = 1:2537
            vector = (test-1)*4000+1:(test-1)*4000+2000;
            vectorarr1(vector) = vector;
        end
        vectorarr1 = vectorarr1(vectorarr1~=0);
        for test = 1:2537
            vector = (test-1)*4000+2001:(test-1)*4000+4000;
            vectorarr2(vector) = vector;
        end
        vectorarr2 = vectorarr2(vectorarr2~=0);

        % Final creation of Haar-like features for target voxel. This will
        % form the feature vector.
        diff = testerarr(vectorarr1) - testerarr(vectorarr2);
        diff = reshape(diff,[2000,2537]);
        diff = diff';

        % The distance between the ground-truth voxel and each training
        % voxel acts as the label (regression value) corresponding to each
        % feature vector.
        dist = coords - 61;
        eucdist = sqrt(dist(:,1).^2+dist(:,2).^2+dist(:,3).^2);
        p = eucdist;

        % Concatenates features to array.
        finalpred = [diff,p];
        finalpredarr = [finalpredarr; finalpred];
    end

% Training to build model.
Mdl = TreeBagger(20,finalpredarr(:,1:end-1),finalpredarr(:,end),'Method','regression');
save(sprintf('mdl%d_med.mat',g),'Mdl')
clear all
end