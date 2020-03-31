function autofid_main(img, outputfcsv)

    %% Function information
    % Uses MATLAB version 2019b. Main program that performs automatic fiducial
    % placement and outputs a fiducial file (.fcsv) with all 32  fiducials from
    % the AFIDS protocol. See paper: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.24693

    % Input params
    % img: input nifti file (.nii extension) of T1w MR image for automatic
    % fiducial placement.
    % outputfcsv: name that is chosen by the user to generate a file that
    % stores the final predicted locations for the 32 fiducials.
    
    % Example
    % After adding all file contents to working directory or cd to working
    % directory, input:
    % autofid('C:\Users\name_of_user\autofid\test_set\imgs\'sub-0109_T1w.nii','name_of_fcsv_file.fcsv')
    % After running (takes about 10-20 seconds per fiducial...can modify
    % code to only output specified fiducials, see Line 149), should get an
    % output .fcsv file that roughly matches OAS1_0109_MR1_MEAN_new.fcsv in
    % the '...\autofid\test_set\autofid_fcsvs' folder.
    
    % Note: while running for the first time on an image, some time (~2
    % min) will be needed to generate a baseline feature map before
    % fiducial prediction begins.
    
    %% Initialization for Coarse Resolution
    % Need a NIFTI image file reader (not available as a default function in older versions of
    % MATLAB). I used a reader from this source: https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
    
    % Make sure folder and files are on path. Correct following line of
    % code to ensure everything is on path by adding the directory to the
    % working directory.
    disp('Starting autofid...')
    addpath('C:\Users\...\autofid-master')

    answerarr = [];
    coarseoutput_arr = cell(1,32);
    medoutput_arr = cell(1,32);
    fineoutput_arr = cell(1,32);

    % Load NIfTI image file.
    niimeta = load_nii(img);
    nii = niimeta.img;
    
    % Transformation matrix for RAS to Matlab coordinates.
    ori_arr = [niimeta.hdr.hist.srow_x(4); niimeta.hdr.hist.srow_y(4); niimeta.hdr.hist.srow_z(4)];
    spa_arr = [1,1,1];
    fourbyfour = inv([diag(spa_arr) ori_arr; 0 0 0 1]);

    % Preprocessing. Resizing and grayscale conversion of new scan.
    nii = single(nii);
    nii = (nii-min(min(min(nii))))*(1)/(max(max(max(nii)))-min(min(min(nii))));
    out = nii;
    out = imresize3(out,[niimeta.hdr.dime.pixdim(2)*size(out,1),niimeta.hdr.dime.pixdim(3)*size(out,2),niimeta.hdr.dime.pixdim(4)*size(out,3)]);
        
    %% Coarse Resolution
    % Resizes image to 1/4 original size.
    out = imresize3(out,0.25);
        
    % Pads image boundaries with 0's.
    out = padarray(out,[50,50,50],0,'both');
        
    % Proceeds through image voxels in a stepwise manner. Image voxels
    % that are selected will be fed into trained model to see its
    % likelihood of being the target location. A stepwise selection is
    % implemented because an exhaustive search would be too
    % computationally expensive (step size is optimized such that no
    % more than 25000 voxel candidates are selected.
    step = 1;
    coords = combvec(51:step:size(out,1)-50,51:step:size(out,2)-50,51:step:size(out,3)-50);
    coords = coords';
    while size(coords,1) > 25000
        coords = combvec(51:step:size(out,1)-50,51:step:size(out,2)-50,51:step:size(out,3)-50);
        coords = coords';
        step = step+1;
    end
    
    % Generates feature map file for selected voxels. Feature vectors will
    % be stored in the .mat file specified by the input file name followed
    % by '_coarsefmap'.
    strings = regexp(img,'\','split');
    fnameog = strings{end};
    fnameog = split(fnameog,'.');
    fmapname = strcat(fnameog{1},'_coarsefmap.mat');

    if isfile(fmapname)
        % Loads coarse feature map if the file already exists.
        load(fmapname,'diff')
        disp('Feature map loaded.')
    else
        disp('No feature map for this image exists on current directory. Generating feature map...')
        % The section generates feature map and stores it in a new .mat file.
        
        % Creates an integral image from the existing 1/4 resolution
        % volume.
        J = integralImage3(out);
        
        % Feature vectors come from Haar-like features (intensity
        % differences from random blocks within the image). The predefined
        % offsets of the blocks are defined in the file 'fineoffset1.mat',
        % which are now being loaded to create Haar-like features.
        load('coarse_haaroffset.mat')
        mincornerlist = zeros(size(coords,1)*4000,3);
        maxcornerlist = zeros(size(coords,1)*4000,3);
        testerarr = zeros(size(coords,1)*4000,1);
        for i = 1:size(coords,1)
            mincorner = coords(i,:) + smin;
            maxcorner = coords(i,:) + smax;
            mincornerlist((i-1)*4000+1:i*4000,:) = mincorner;
            maxcornerlist((i-1)*4000+1:i*4000,:) = maxcorner;
        end
        cornerlist = [mincornerlist,maxcornerlist];
        % Extracts Haar-like features from coarse resolution blocks.
        for i = 1:size(coords,1)*4000
            testerarr(i) = (J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,6)+1)-J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,3)) - J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,6)+1) ...
            - J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,3)) ...
            + J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,3)) - J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,3)))/((cornerlist(i,4)-cornerlist(i,1)+1)*(cornerlist(i,5)-cornerlist(i,2)+1)*(cornerlist(i,6)-cornerlist(i,3)+1));
        end
       
        for test = 1:size(coords,1)
            vector = (test-1)*4000+1:(test-1)*4000+2000;
            vectorarr1(vector) = vector;
        end
        vectorarr1 = vectorarr1(vectorarr1~=0);
        for test = 1:size(coords,1)
            vector = (test-1)*4000+2001:(test-1)*4000+4000;
            vectorarr2(vector) = vector;
        end
        vectorarr2 = vectorarr2(vectorarr2~=0);
        
        % Final feature vectors created for target voxels.
        diff = testerarr(vectorarr1) - testerarr(vectorarr2);
        diff = reshape(diff,[2000,size(coords,1)]);
        diff = diff';

        % Saves the new feature map.
        save(fmapname,'diff')
        
        % Reloads the feature map immediately for further processing.
        load(fmapname,'diff')
        disp('Feature map loaded.')
    end

    % Goes through each of the 32 fiducial locations similarly.
    %%% NOTE: Can modify numbers here to customize range of desired output
    %%% fiducials.
    disp('Starting automatic fiducial localization...')
    for g = 1:32
        
        % Load the regression forest model corresponding to the current
        % active fiducial.
        load(sprintf('cmdl%d_coarse.mat',g))
        model = Mdl;
        diff = double(diff);
        
        % Make predictions at coarse resolution level.
        unsorted = predict(model,diff); 

        % Sorts predictions in ascending order (voxel with the smallest
        % predicted distance from ground truth comes first).
        sorted = sort(unsorted,1);
        sorted_unique = unique(sorted);
        coarseoutput = [];
        for i = 0:size(sorted_unique,1)-1
            topchoice = find(unsorted==sorted_unique(end-i));
            coarseoutput = [coarseoutput; coords(topchoice,:)];
        end
        coarseoutput = [coarseoutput,flip(sorted(end-(size(sorted,1)-1):end))];
        coarseoutput = flipud(coarseoutput);
        coarseoutput_arr{g} = coarseoutput;

        % Takes coordinate location of the lowest distance score as the
        % final output at the coarse resolution level.
        testing = coarseoutput(1,1:3);

        % Account for padding and rescales to original resolution.
        testing = [(testing-50)*4,1];

        %% Initialization for Medium Resolution
        % Re-initialize volume (similar to initialization for coarse
        % resolution). Please refer to Initialization section for
        % descriptions of subsequent lines of code.
        niimeta = load_nii(img);
        nii = niimeta.img;
        
        nii = single(nii);
        nii = (nii-min(min(min(nii))))*(1)/(max(max(max(nii)))-min(min(min(nii))));
        out = nii;
        out = imresize3(out,[niimeta.hdr.dime.pixdim(2)*size(out,1),niimeta.hdr.dime.pixdim(3)*size(out,2),niimeta.hdr.dime.pixdim(4)*size(out,3)]);
        
        %% Medium Resolution
        % Works with the image at the current resolution level.        
        % Pads image boundaries with 0's.
        out = padarray(out,[50,50,50],0,'both');
        testing = [round(testing(1:3))+50,1];
        
        % Works with image patch around predicted point at coarse
        % resolution level. Patch intensities are normalized.
        patch = out(testing(1)-60:testing(1)+60,testing(2)-60:testing(2)+60,testing(3)-60:testing(3)+60);
        patch = (patch-min(min(min(patch))))*(1)/(max(max(max(patch)))-min(min(min(patch))));
        
        % Creates an integral image using the current patch volume.
        J = integralImage3(patch);

        % Extracts voxels around predicted coarse resolution point with a
        % skip size of 2.
        coordsmed = combvec(60-16:2:60+16,60-16:2:60+16,60-16:2:60+16);
        coordsmed = coordsmed';

        % Loads Haar-like features block offsets at the medium resolution
        % level.
        load('med_haaroffset.mat')
        mincornerlist = zeros(4913*4000,3);
        maxcornerlist = zeros(4913*4000,3);
        testerarr = zeros(4913*4000,1);
        for i = 1:size(coordsmed,1)
            mincorner = coordsmed(i,:) + smin;
            maxcorner = coordsmed(i,:) + smax;
            mincornerlist((i-1)*4000+1:i*4000,:) = mincorner;
            maxcornerlist((i-1)*4000+1:i*4000,:) = maxcorner;
        end
        cornerlist = [mincornerlist,maxcornerlist];
        
        % Extracts Haar-like features from medium resolution blocks.
        for i = 1:4913*4000
            testerarr(i) = (J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,6)+1)-J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,3)) - J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,6)+1) ...
            - J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,3)) ...
            + J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,3)) - J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,3)))/((cornerlist(i,4)-cornerlist(i,1)+1)*(cornerlist(i,5)-cornerlist(i,2)+1)*(cornerlist(i,6)-cornerlist(i,3)+1));
        end
        
        vectorarr1 = [];
        vectorarr2 = [];
        
        for test = 1:4913
            vector = (test-1)*4000+1:(test-1)*4000+2000;
            vectorarr1(vector) = vector;
        end
        vectorarr1 = vectorarr1(vectorarr1~=0);
        for test = 1:4913
            vector = (test-1)*4000+2001:(test-1)*4000+4000;
            vectorarr2(vector) = vector;
        end
        vectorarr2 = vectorarr2(vectorarr2~=0);
        
        % Final feature vectors created for target voxels.
        diffmed = testerarr(vectorarr1) - testerarr(vectorarr2);
        diffmed = reshape(diffmed,[2000,4913]);
        diffmed = diffmed';
    
        % Load the regression forest model corresponding to the current
        % active fiducial.
        load(sprintf('cmdl%d_med.mat',g))
        model = Mdl;
        diffmed = double(diffmed);
        
        % Make predictions at medium resolution level.
        unsorted = predict(model,diffmed);
        
        % Sorts predictions in ascending order (voxel with the smallest
        % predicted distance from ground truth comes first).
        sorted = sort(unsorted,1);
        sorted_unique = unique(sorted);
        medoutput = [];
        for i = 0:size(sorted_unique,1)-1
            topchoice = find(unsorted==sorted_unique(end-i));
            medoutput = [medoutput; coordsmed(topchoice,:)];
        end
        medoutput = [medoutput,flip(sorted(end-(size(sorted,1)-1):end))];
        medoutput = flipud(medoutput);
        medoutput_arr{g} = medoutput;

        % Takes coordinate location of the lowest distance score as the
        % final output at the medium resolution level.
        testing = [medoutput(1,1:3)-61+testing(1:3)-50,1];

        %% Initialization for Fine Resolution
        % Re-initialize volume (similar to initialization for coarse
        % resolution). Please refer to Initialization section for
        % descriptions of subsequent lines of code.
        niimeta = load_nii(img);
        nii = niimeta.img;      

        nii = single(nii);
        nii = (nii-min(min(min(nii))))*(1)/(max(max(max(nii)))-min(min(min(nii))));
        out = nii;
        out = imresize3(out,[niimeta.hdr.dime.pixdim(2)*size(out,1),niimeta.hdr.dime.pixdim(3)*size(out,2),niimeta.hdr.dime.pixdim(4)*size(out,3)]);
        
        %% Fine Resolution
        % Works with the image at the current resolution level.        
        % Pads image boundaries with 0's.
        out = imresize3(out,2);
        out = padarray(out,[50,50,50],0,'both');
        testing = [round(testing(1:3)*2)+50,1];
        
        % Works with image patch around predicted point at medium
        % resolution level. Patch intensities are normalized.
        patch = out(testing(1)-120:testing(1)+120,testing(2)-120:testing(2)+120,testing(3)-120:testing(3)+120);
        patch = (patch-min(min(min(patch))))*(1)/(max(max(max(patch)))-min(min(min(patch))));
        
        % Creates an integral image using the current patch volume.        
        J = integralImage3(patch);
        
        % Extracts voxels around predicted medium resolution point, this
        % time with no skip size.
        coordsfine = combvec(121-4:121+4,121-4:121+4,121-4:121+4);
        coordsfine = coordsfine';

        % Loads Haar-like features block offsets at the medium resolution
        % level.
        load('fine_haaroffset.mat')
        mincornerlist = zeros(729*4000,3);
        maxcornerlist = zeros(729*4000,3);
        testerarr = zeros(729*4000,1);
        for i = 1:size(coordsfine,1)
            mincorner = coordsfine(i,:) + smin;
            maxcorner = coordsfine(i,:) + smax;
            mincornerlist((i-1)*4000+1:i*4000,:) = mincorner;
            maxcornerlist((i-1)*4000+1:i*4000,:) = maxcorner;
        end
        cornerlist = [mincornerlist,maxcornerlist];
        
        % Extracts Haar-like features from fine resolution blocks.
        for i = 1:729*4000
            testerarr(i) = (J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,6)+1)-J(cornerlist(i,4)+1,cornerlist(i,5)+1,cornerlist(i,3)) - J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,6)+1) ...
            - J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,6)+1) + J(cornerlist(i,1),cornerlist(i,5)+1,cornerlist(i,3)) ...
            + J(cornerlist(i,4)+1,cornerlist(i,2),cornerlist(i,3)) - J(cornerlist(i,1),cornerlist(i,2),cornerlist(i,3)))/((cornerlist(i,4)-cornerlist(i,1)+1)*(cornerlist(i,5)-cornerlist(i,2)+1)*(cornerlist(i,6)-cornerlist(i,3)+1));
        end

        vectorarr1 = [];
        vectorarr2 = [];
        
        for test = 1:729
            vector = (test-1)*4000+1:(test-1)*4000+2000;
            vectorarr1(vector) = vector;
        end
        vectorarr1 = vectorarr1(vectorarr1~=0);
        for test = 1:729
            vector = (test-1)*4000+2001:(test-1)*4000+4000;
            vectorarr2(vector) = vector;
        end
        vectorarr2 = vectorarr2(vectorarr2~=0);

        % Final feature vectors created for target voxels.
        difffine = testerarr(vectorarr1) - testerarr(vectorarr2);
        difffine = reshape(difffine,[2000,729]);
        difffine = difffine';

        % Load the regression forest model corresponding to the current
        % active fiducial.
        load(sprintf('cmdl%d_fine.mat',g))
        model = Mdl;
        difffine = double(difffine);

        % Make predictions at medium resolution level.
        unsorted = predict(model,difffine);

        % Load the regression forest model corresponding to the current
        % active fiducial.        
        sorted = sort(unsorted,1);
        sorted_unique = unique(sorted);
        fineoutput = [];
        for i = 0:size(sorted_unique,1)-1
            topchoice = find(unsorted==sorted_unique(end-i));
            fineoutput = [fineoutput; coordsfine(topchoice,:)];
        end
        fineoutput = [fineoutput,flip(sorted(end-(size(sorted,1)-1):end))];
        fineoutput = flipud(fineoutput);
        fineoutput_arr{g} = fineoutput;

        % Takes coordinate location of the lowest distance score as the
        % final output at the coarse resolution level. Also accounts for
        % padding and rescales to original resolution.
        answer = [fineoutput(1,1:3)-121+testing(1:3)-50,1];
        answer = answer/2;
        
        % Conversion of Matlab coordinates back to RAS coordinate space,
        % and stores answers in an array.
        answer = [answer(1:3),1];
        answer = fourbyfour\answer';
        answer = [answer(1:3)',g];
        answerarr = [answerarr;answer];
        disp(sprintf('Fiducial %d localization complete',g))
    end

    % Quick check for orientation tag in NIfTI header file. Depending on
    % initial orientation, answers may have to be inverted to generate
    % correct RAS coordinate values.
    if ~isempty(niimeta.hdr.hist.flip_orient)
        for q = 1:3
            if niimeta.hdr.hist.flip_orient(q) ~= 0
                answerarr(:,q) = answerarr(:,q)-size(nii,q);
            end
        end
    end
    
    % String manipulation for creation of output fcsv file. Need to first
    % scan format of 'dummy.fcsv', which is a placeholder file that spells
    % out the default formatting of .fcsv files.
    f = fopen('dummy.fcsv');
    C = textscan(f,'%s %f %f %f %f %f %f %f %f %f %f %s %s %s', 'Delimiter', ',','HeaderLines', 3);
    fclose(f);
    gs = answerarr(:,end);
    for i = 1:14
        C{i} = C{i}(gs);
    end
    C{2} = answerarr(:,1);
    C{3} = answerarr(:,2);
    C{4} = answerarr(:,3);
    f = fopen('dummy.fcsv');
    Ctest = textscan(f,'%s','delimiter','\n');
    cellpart = [C{2},C{3},C{4},C{5},C{6},C{7},C{8},C{9},C{10},C{11}];
    fclose(f);

    cells = cell(size(cellpart,1),1);
    for i = 1:size(cellpart,1)
        floating = cellpart(i,1:3);
        floating = sprintf('%.4f,' , floating);
        whole = cellpart(i,4:end);
        whole = sprintf('%.0f,' , whole);
        full = strcat(floating,whole);
        final = strcat(C{1}(i),',',full,C{12}(i),',',C{13}(i),',',C{14}(i));
        cells{i} = final;
    end
    newfile = [Ctest{1}(1:3);vertcat(cells{:})];

    % Writes results to final .fcsv file, which is saved in the same
    % directory.
    lines = '%s\n';
    filefinal = fopen(outputfcsv,'w');
    for i = 1:size(newfile,1)
        fprintf(filefinal, lines, newfile{i});
    end
    fclose(filefinal);
    disp('.fcsv file created. Operation complete.')
end
