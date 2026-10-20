
clc;
clear;
close all;

%% 1. SELECT VIDEO
[file,path] = uigetfile( ...
    {'*.mp4;*.avi;*.mov;*.mkv','Video Files'}, ...
    'Select FRUC input video');

if isequal(file,0)
    error('No video selected.');
end

video = VideoReader(fullfile(path,file));

%% 2. READ FIVE CONSECUTIVE FRAMES
frames = cell(1,5);

for k = 1:5
    if ~hasFrame(video)
        error('Video must contain at least five frames.');
    end

    frame = readFrame(video);

    if size(frame,3) == 3
        frame = rgb2gray(frame);
    end

    frames{k} = frame;
end

F1 = frames{1};
G1 = frames{2};
G2 = frames{3};
G3 = frames{4};
F2 = frames{5};

%% 3. CROP FRAMES AS IN ORIGINAL CODE
blockSize = 16;
searchRange = 32;

[originalH,originalW] = size(F1);

H = floor(originalH/blockSize)*blockSize;
W = floor(originalW/blockSize)*blockSize;

if H < blockSize || W < blockSize
    error('Frames are too small for 16x16 blocks.');
end

for k = 1:5
    frames{k} = frames{k}(1:H,1:W);
end

F1 = frames{1};
G1 = frames{2};
G2 = frames{3};
G3 = frames{4};
F2 = frames{5};

numRows = H/blockSize;
numCols = W/blockSize;

%% 4. FORWARD MOTION ESTIMATION
MVx12 = zeros(numRows,numCols);
MVy12 = zeros(numRows,numCols);

fprintf('Estimating forward motion...\n'); %[output:33b5cf25]

for br = 1:numRows
    for bc = 1:numCols

        row = (br-1)*blockSize+1;
        col = (bc-1)*blockSize+1;

        block = double(F1( ...
            row:row+blockSize-1, ...
            col:col+blockSize-1));

        if bc > 1
            predX = MVx12(br,bc-1);
            predY = MVy12(br,bc-1);
        elseif br > 1
            predX = MVx12(br-1,bc);
            predY = MVy12(br-1,bc);
        else
            predX = 0;
            predY = 0;
        end

        [dx,dy,~] = fast_block_search( ...
            F2,block,row,col,predX,predY, ...
            searchRange,blockSize);

        MVx12(br,bc) = dx;
        MVy12(br,bc) = dy;
    end
end

%% 5. BACKWARD MOTION ESTIMATION
MVx21 = zeros(numRows,numCols);
MVy21 = zeros(numRows,numCols);

fprintf('Estimating backward motion...\n'); %[output:3b805c1a]

for br = 1:numRows
    for bc = 1:numCols

        row = (br-1)*blockSize+1;
        col = (bc-1)*blockSize+1;

        block = double(F2( ...
            row:row+blockSize-1, ...
            col:col+blockSize-1));

        if bc > 1
            predX = MVx21(br,bc-1);
            predY = MVy21(br,bc-1);
        elseif br > 1
            predX = MVx21(br-1,bc);
            predY = MVy21(br-1,bc);
        else
            predX = 0;
            predY = 0;
        end

        [dx,dy,~] = fast_block_search( ...
            F1,block,row,col,predX,predY, ...
            searchRange,blockSize);

        MVx21(br,bc) = dx;
        MVy21(br,bc) = dy;
    end
end

%% 6. FILTER MOTION VECTORS
MVx12 = medfilt2(MVx12,[3 3]);
MVy12 = medfilt2(MVy12,[3 3]);
MVx21 = medfilt2(MVx21,[3 3]);
MVy21 = medfilt2(MVy21,[3 3]);

%% 7. BLOCK-TO-PIXEL CONVERSION
MVX12 = zeros(H,W);
MVY12 = zeros(H,W);
MVX21 = zeros(H,W);
MVY21 = zeros(H,W);

for br = 1:numRows
    for bc = 1:numCols

        r1 = (br-1)*blockSize+1;
        r2 = br*blockSize;
        c1 = (bc-1)*blockSize+1;
        c2 = bc*blockSize;

        MVX12(r1:r2,c1:c2) = MVx12(br,bc);
        MVY12(r1:r2,c1:c2) = MVy12(br,bc);

        MVX21(r1:r2,c1:c2) = MVx21(br,bc);
        MVY21(r1:r2,c1:c2) = MVy21(br,bc);
    end
end

%% 8. PIXEL COORDINATES
[X,Y] = meshgrid(1:W,1:H);

%% 9. FIR CONFIGURATIONS
tapLengths = [4 8 12 16];

windowNames = { ...
    'Rectangular','Hann','Hamming','Blackman'};

numTaps = numel(tapLengths);
numWindows = numel(windowNames);

AveragePSNR = zeros(numTaps,numel(windowNames));
AverageSSIM = zeros(numTaps,numel(windowNames));

targetFrames = {G1,G2,G3};
targetTimes = [0.25 0.50 0.75];
phases = [0 0.25 0.50 0.75];

%% 10. RUN ALL CONFIGURATIONS
for ti = 1:numTaps %[output:group:03cb532f]

    N = tapLengths(ti);

    for wi = 1:numel(windowNames)

        windowName = windowNames{wi};

        fprintf('\nTesting %d taps - %s\n',N,windowName); %[output:504de319] %[output:28ab8352] %[output:496bad5d] %[output:1d2f85c0] %[output:6f547d46] %[output:70e4b5d1] %[output:744ac55e] %[output:76a93a1b] %[output:658b750d] %[output:5f2d6a51] %[output:7bd2c451] %[output:8e722b8e] %[output:8e4e3347] %[output:7b02de8c] %[output:39e3d777] %[output:3cafb002]

        Hfir = zeros(4,N);

        for p = 1:4
            Hfir(p,:) = fractional_delay_coeff( ...
                phases(p),N,windowName);
        end

        psnrValues = zeros(1,3);
        ssimValues = zeros(1,3);

        for k = 1:3

            t = targetTimes(k);

            % Original motion-compensation equations
            X1 = X-t*MVX12;
            Y1 = Y-t*MVY12;

            X2 = X+(1-t)*MVX21;
            Y2 = Y+(1-t)*MVY21;

            % Use the EXACT original function for 8-tap Hamming
            if N == 8 && strcmpi(windowName,'Hamming')

                I1temp = fir_interpolate_8tap( ...
                    F1,X1,Y1,Hfir);

                I2temp = fir_interpolate_8tap( ...
                    F2,X2,Y2,Hfir);

            else

                I1temp = fir_interpolate_variable( ...
                    F1,X1,Y1,Hfir,N);

                I2temp = fir_interpolate_variable( ...
                    F2,X2,Y2,Hfir,N);
            end

            % Original equal-weight blending
            result = uint8(max(0,min(255, ...
                (I1temp+I2temp)/2)));

            reference = targetFrames{k};

            psnrValues(k) = psnr(result,reference);
            ssimValues(k) = ssim(result,reference);
        end

        AveragePSNR(ti,wi) = mean(psnrValues);
        AverageSSIM(ti,wi) = mean(ssimValues);

        fprintf('PSNR1 = %.4f dB\n',psnrValues(1)); %[output:53ad2e35] %[output:92462a18] %[output:1b573362] %[output:17f7a426] %[output:9cbbd923] %[output:3b319d37] %[output:424b7e09] %[output:06b419bf] %[output:4759af7d] %[output:5d6bcf19] %[output:8470ec84] %[output:657c18c2] %[output:2c55653b] %[output:572a855e] %[output:6a1017d2] %[output:26eff442]
        fprintf('PSNR2 = %.4f dB\n',psnrValues(2)); %[output:71388598] %[output:412beec8] %[output:19ce1347] %[output:01dab60b] %[output:7e1d0856] %[output:704db8b6] %[output:2d6c3072] %[output:8eaa0aad] %[output:27034e9c] %[output:546995d1] %[output:2ea27465] %[output:9ef15240] %[output:18430226] %[output:10e2b815] %[output:0208280f] %[output:1f4c0936]
        fprintf('PSNR3 = %.4f dB\n',psnrValues(3)); %[output:1e15c918] %[output:518f41a6] %[output:65043433] %[output:0396b0c3] %[output:1968b996] %[output:2d9027b6] %[output:23baeb4e] %[output:1d8257c9] %[output:2c50958d] %[output:549fa7ef] %[output:2eaefea2] %[output:7bff3441] %[output:2a3e5a3a] %[output:809f507a] %[output:61fcca83] %[output:342100b3]

        fprintf('Average PSNR = %.4f dB\n', ... %[output:349fa76b] %[output:2c87fe5d] %[output:5afd2d25] %[output:17d6d578] %[output:4506e1ad] %[output:740d83af] %[output:14e8b3cf] %[output:3a8ac5d7] %[output:5e2a5f22] %[output:0b3bc812] %[output:967a8cf6] %[output:029eacfe] %[output:3ac264e8] %[output:3d3a4c73] %[output:17b94478] %[output:96132402]
            AveragePSNR(ti,wi)); %[output:349fa76b] %[output:2c87fe5d] %[output:5afd2d25] %[output:17d6d578] %[output:4506e1ad] %[output:740d83af] %[output:14e8b3cf] %[output:3a8ac5d7] %[output:5e2a5f22] %[output:0b3bc812] %[output:967a8cf6] %[output:029eacfe] %[output:3ac264e8] %[output:3d3a4c73] %[output:17b94478] %[output:96132402]

        fprintf('Average SSIM = %.4f\n', ... %[output:0aa30103] %[output:8d50a326] %[output:3b6e7893] %[output:6579f99e] %[output:31ba5a88] %[output:67780741] %[output:1bbcede5] %[output:3c5c8105] %[output:39a5bcb5] %[output:31afbdad] %[output:83ecdebe] %[output:0159f55c] %[output:7f1d43fc] %[output:2de57be9] %[output:5858d92c] %[output:8a52fd53]
            AverageSSIM(ti,wi)); %[output:0aa30103] %[output:8d50a326] %[output:3b6e7893] %[output:6579f99e] %[output:31ba5a88] %[output:67780741] %[output:1bbcede5] %[output:3c5c8105] %[output:39a5bcb5] %[output:31afbdad] %[output:83ecdebe] %[output:0159f55c] %[output:7f1d43fc] %[output:2de57be9] %[output:5858d92c] %[output:8a52fd53]

        % Verify the original configuration
        if N == 8 && strcmpi(windowName,'Hamming')
            fprintf('\n*** ORIGINAL 8-TAP HAMMING CHECK ***\n'); %[output:151d45d6]
            fprintf('Expected previous average: 44.58 dB\n'); %[output:5a0baef4]
            fprintf('Current average: %.4f dB\n', ... %[output:05e225e9]
                AveragePSNR(ti,wi)); %[output:05e225e9]
            fprintf('***********************************\n'); %[output:082f61d3]
        end
    end
end %[output:group:03cb532f]

%% 11. RESULTS TABLE
Results = table( ...
    repelem(tapLengths(:),numel(windowNames)), ...
    repmat(windowNames(:),numTaps,1), ...
    reshape(AveragePSNR.',[],1), ...
    reshape(AverageSSIM.',[],1), ...
    'VariableNames', ...
    {'Taps','Window','AveragePSNR_dB','AverageSSIM'});

fprintf('\n========= FIR COMPARISON RESULTS =========\n'); %[output:91963e25]
disp(Results); %[output:082289a3]

writetable(Results,'FRUC_FIR_Results.csv');

%% 12. REPORT BEST PSNR AND SSIM
[bestPSNR,idxPSNR] = max(AveragePSNR(:));
[bestSSIM,idxSSIM] = max(AverageSSIM(:));

[tiP,wiP] = ind2sub(size(AveragePSNR),idxPSNR);
[tiS,wiS] = ind2sub(size(AverageSSIM),idxSSIM);

fprintf('\nBEST PSNR CONFIGURATION\n'); %[output:820357f3]
fprintf('%d taps, %s window: %.4f dB\n', ... %[output:group:4f5a48e7] %[output:2d097538]
    tapLengths(tiP),windowNames{wiP},bestPSNR); %[output:group:4f5a48e7] %[output:2d097538]

fprintf('\nBEST SSIM CONFIGURATION\n'); %[output:6e68e2b8]
fprintf('%d taps, %s window: %.4f\n', ... %[output:group:3be67a72] %[output:0ad744ff]
    tapLengths(tiS),windowNames{wiS},bestSSIM); %[output:group:3be67a72] %[output:0ad744ff]

%% 13. GRAPH: PSNR VS TAP LENGTH
figure('Name','PSNR Comparison','Color','w'); %[output:25ff1708]

bar(tapLengths,AveragePSNR,'grouped'); %[output:25ff1708]

title('PSNR Comparison'); %[output:25ff1708]
xlabel('Number of taps'); %[output:25ff1708]
ylabel('Average PSNR (dB)'); %[output:25ff1708]

legend(windowNames,'Location','best'); %[output:25ff1708]
xticks(tapLengths); %[output:25ff1708]
grid on; %[output:25ff1708]
box on; %[output:25ff1708]

%% 14. GRAPH: SSIM VS TAP LENGTH
figure('Name','SSIM Comparison','Color','w'); %[output:44f4e8c7]

bar(tapLengths,AverageSSIM,'grouped'); %[output:44f4e8c7]

title('SSIM Comparison'); %[output:44f4e8c7]
xlabel('Number of taps'); %[output:44f4e8c7]
ylabel('Average SSIM'); %[output:44f4e8c7]

legend(windowNames,'Location','best'); %[output:44f4e8c7]
xticks(tapLengths); %[output:44f4e8c7]
grid on; %[output:44f4e8c7]
box on; %[output:44f4e8c7]

%% 15. GRAPH: PSNR BY WINDOW
figure('Name','PSNR by Window','Color','w'); %[output:1cdea7d8]

bar(categorical(windowNames),AveragePSNR.','grouped'); %[output:1cdea7d8]

title('PSNR by Window and Tap Length'); %[output:1cdea7d8]
xlabel('Window type'); %[output:1cdea7d8]
ylabel('Average PSNR (dB)'); %[output:1cdea7d8]

legend(string(tapLengths)+' taps','Location','best'); %[output:1cdea7d8]
grid on; %[output:1cdea7d8]
box on; %[output:1cdea7d8]

%% 16. GRAPH: SSIM BY WINDOW
figure('Name','SSIM by Window','Color','w'); %[output:4a09c78c]

bar(categorical(windowNames),AverageSSIM.','grouped'); %[output:4a09c78c]

title('SSIM by Window and Tap Length'); %[output:4a09c78c]
xlabel('Window type'); %[output:4a09c78c]
ylabel('Average SSIM'); %[output:4a09c78c]

legend(string(tapLengths)+' taps','Location','best'); %[output:4a09c78c]
grid on; %[output:4a09c78c]
box on; %[output:4a09c78c]

%% 17. ORIGINAL FAST SAD BLOCK SEARCH
function [bestDx,bestDy,minSAD] = fast_block_search( ...
    F,block,row,col,predX,predY,searchRange,blockSize)

    H = size(F,1);
    W = size(F,2);

    predX = round(predX);
    predY = round(predY);

    predX = max(-searchRange,min(searchRange,predX));
    predY = max(-searchRange,min(searchRange,predY));

    bestDx = predX;
    bestDy = predY;

    r = row+bestDy;
    c = col+bestDx;

    if r >= 1 && c >= 1 && ...
       r+blockSize-1 <= H && c+blockSize-1 <= W

        candidate = double(F( ...
            r:r+blockSize-1,c:c+blockSize-1));

        minSAD = sum(abs(block(:)-candidate(:)));
    else
        bestDx = 0;
        bestDy = 0;

        candidate = double(F( ...
            row:row+blockSize-1, ...
            col:col+blockSize-1));

        minSAD = sum(abs(block(:)-candidate(:)));
    end

    step = 16;

    while step >= 1

        candidates = [
             0  0;
            -1  0;
             1  0;
             0 -1;
             0  1;
            -1 -1;
            -1  1;
             1 -1;
             1  1];

        oldDx = bestDx;
        oldDy = bestDy;

        for k = 1:size(candidates,1)

            dx = oldDx+candidates(k,1)*step;
            dy = oldDy+candidates(k,2)*step;

            if dx < -searchRange || dx > searchRange || ...
               dy < -searchRange || dy > searchRange
                continue;
            end

            r = row+dy;
            c = col+dx;

            if r < 1 || c < 1 || ...
               r+blockSize-1 > H || ...
               c+blockSize-1 > W
                continue;
            end

            candidate = double(F( ...
                r:r+blockSize-1,c:c+blockSize-1));

            SAD = sum(abs(block(:)-candidate(:)));

            if SAD < minSAD
                minSAD = SAD;
                bestDx = dx;
                bestDy = dy;
            end
        end

        step = step/2;
    end
end

%% 18. WINDOWED-SINC FIR COEFFICIENTS
function h = fractional_delay_coeff(p,N,windowName)

    n = -(N/2)+1:N/2;
    h = sinc(n-p);

    q = 0:N-1;

    switch lower(windowName)

        case 'rectangular'
            w = ones(1,N);

        case 'hann'
            w = 0.5-0.5*cos(2*pi*q/(N-1));

        case 'hamming'
            w = 0.54-0.46*cos(2*pi*q/(N-1));

        case 'blackman'
            w = 0.42 ...
                - 0.5*cos(2*pi*q/(N-1)) ...
                + 0.08*cos(4*pi*q/(N-1));

        otherwise
            error('Unknown window.');
    end

    h = h.*w;
    h = h/sum(h);
end

%% 19. EXACT ORIGINAL 8-TAP INTERPOLATION
function output = fir_interpolate_8tap(I,X,Y,Hfir)

    I = double(I);
    [H,W] = size(I);
    output = zeros(H,W);

    for y = 1:H
        for x = 1:W

            sx = min(max(X(y,x),1),W);
            sy = min(max(Y(y,x),1),H);

            ix = floor(sx);
            iy = floor(sy);

            fx = sx-ix;
            fy = sy-iy;

            px = round(fx*4)/4;
            py = round(fy*4)/4;

            if px >= 1
                ix = min(ix+1,W);
                px = 0;
            end

            if py >= 1
                iy = min(iy+1,H);
                py = 0;
            end

            phaseX = round(px*4);
            phaseY = round(py*4);

            hx = Hfir(phaseX+1,:);
            hy = Hfir(phaseY+1,:);

            xs = ix+(-3:4);
            ys = iy+(-3:4);

            xs = max(1,min(W,xs));
            ys = max(1,min(H,ys));

            patch = I(ys,xs);
            weights = hy'*hx;

            output(y,x) = sum(sum(patch.*weights));
        end
    end
end

%% 20. GENERAL VARIABLE-TAP INTERPOLATION
function output = fir_interpolate_variable(I,X,Y,Hfir,N)

    I = double(I);
    [H,W] = size(I);
    output = zeros(H,W);

    offsets = -(N/2)+1:N/2;

    for y = 1:H
        for x = 1:W

            sx = min(max(X(y,x),1),W);
            sy = min(max(Y(y,x),1),H);

            ix = floor(sx);
            iy = floor(sy);

            fx = sx-ix;
            fy = sy-iy;

            px = round(fx*4)/4;
            py = round(fy*4)/4;

            if px >= 1
                ix = min(ix+1,W);
                px = 0;
            end

            if py >= 1
                iy = min(iy+1,H);
                py = 0;
            end

            phaseX = round(px*4);
            phaseY = round(py*4);

            hx = Hfir(phaseX+1,:);
            hy = Hfir(phaseY+1,:);

            xs = ix+offsets;
            ys = iy+offsets;

            xs = max(1,min(W,xs));
            ys = max(1,min(H,ys));

            patch = I(ys,xs);
            weights = hy'*hx;

            output(y,x) = sum(sum(patch.*weights));
        end
    end
end


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:33b5cf25]
%   data: {"dataType":"text","outputData":{"text":"Estimating forward motion...\n","truncated":false}}
%---
%[output:3b805c1a]
%   data: {"dataType":"text","outputData":{"text":"Estimating backward motion...\n","truncated":false}}
%---
%[output:504de319]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 4 taps - Rectangular\n","truncated":false}}
%---
%[output:53ad2e35]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.3078 dB\n","truncated":false}}
%---
%[output:71388598]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.0483 dB\n","truncated":false}}
%---
%[output:1e15c918]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 31.6901 dB\n","truncated":false}}
%---
%[output:349fa76b]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.3487 dB\n","truncated":false}}
%---
%[output:0aa30103]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.8998\n","truncated":false}}
%---
%[output:28ab8352]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 4 taps - Hann\n","truncated":false}}
%---
%[output:496bad5d]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 4 taps - Hamming\n","truncated":false}}
%---
%[output:1d2f85c0]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 4 taps - Blackman\n","truncated":false}}
%---
%[output:6f547d46]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 8 taps - Rectangular\n","truncated":false}}
%---
%[output:70e4b5d1]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 8 taps - Hann\n","truncated":false}}
%---
%[output:8d50a326]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9009\n","truncated":false}}
%---
%[output:2c87fe5d]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.8020 dB\n","truncated":false}}
%---
%[output:518f41a6]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.5507 dB\n","truncated":false}}
%---
%[output:412beec8]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.4066 dB\n","truncated":false}}
%---
%[output:92462a18]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4489 dB\n","truncated":false}}
%---
%[output:744ac55e]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 8 taps - Hamming\n","truncated":false}}
%---
%[output:3b6e7893]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9015\n","truncated":false}}
%---
%[output:76a93a1b]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 8 taps - Blackman\n","truncated":false}}
%---
%[output:5afd2d25]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.8056 dB\n","truncated":false}}
%---
%[output:658b750d]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 12 taps - Rectangular\n","truncated":false}}
%---
%[output:65043433]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.5399 dB\n","truncated":false}}
%---
%[output:5f2d6a51]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 12 taps - Hann\n","truncated":false}}
%---
%[output:6579f99e]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9009\n","truncated":false}}
%---
%[output:19ce1347]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.4185 dB\n","truncated":false}}
%---
%[output:7bd2c451]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 12 taps - Hamming\n","truncated":false}}
%---
%[output:17d6d578]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.8020 dB\n","truncated":false}}
%---
%[output:1b573362]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4585 dB\n","truncated":false}}
%---
%[output:8e722b8e]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 12 taps - Blackman\n","truncated":false}}
%---
%[output:31ba5a88]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9009\n","truncated":false}}
%---
%[output:0396b0c3]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.5506 dB\n","truncated":false}}
%---
%[output:8e4e3347]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 16 taps - Rectangular\n","truncated":false}}
%---
%[output:7b02de8c]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 16 taps - Hann\n","truncated":false}}
%---
%[output:4506e1ad]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.5031 dB\n","truncated":false}}
%---
%[output:01dab60b]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.4066 dB\n","truncated":false}}
%---
%[output:67780741]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9027\n","truncated":false}}
%---
%[output:39e3d777]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 16 taps - Hamming\n","truncated":false}}
%---
%[output:3cafb002]
%   data: {"dataType":"text","outputData":{"text":"\nTesting 16 taps - Blackman\n","truncated":false}}
%---
%[output:1968b996]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 31.9609 dB\n","truncated":false}}
%---
%[output:17f7a426]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4487 dB\n","truncated":false}}
%---
%[output:740d83af]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7441 dB\n","truncated":false}}
%---
%[output:1bbcede5]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9027\n","truncated":false}}
%---
%[output:151d45d6]
%   data: {"dataType":"text","outputData":{"text":"\n*** ORIGINAL 8-TAP HAMMING CHECK ***\n","truncated":false}}
%---
%[output:5a0baef4]
%   data: {"dataType":"text","outputData":{"text":"Expected previous average: 44.58 dB\n","truncated":false}}
%---
%[output:05e225e9]
%   data: {"dataType":"text","outputData":{"text":"Current average: 30.7338 dB\n","truncated":false}}
%---
%[output:082f61d3]
%   data: {"dataType":"text","outputData":{"text":"***********************************\n","truncated":false}}
%---
%[output:9cbbd923]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.3721 dB\n","truncated":false}}
%---
%[output:3b319d37]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4720 dB\n","truncated":false}}
%---
%[output:7e1d0856]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.1764 dB\n","truncated":false}}
%---
%[output:424b7e09]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4690 dB\n","truncated":false}}
%---
%[output:2d9027b6]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.3567 dB\n","truncated":false}}
%---
%[output:14e8b3cf]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7338 dB\n","truncated":false}}
%---
%[output:06b419bf]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4725 dB\n","truncated":false}}
%---
%[output:4759af7d]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.3964 dB\n","truncated":false}}
%---
%[output:5d6bcf19]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4711 dB\n","truncated":false}}
%---
%[output:3c5c8105]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9027\n","truncated":false}}
%---
%[output:8470ec84]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4682 dB\n","truncated":false}}
%---
%[output:704db8b6]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.4035 dB\n","truncated":false}}
%---
%[output:23baeb4e]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.3365 dB\n","truncated":false}}
%---
%[output:657c18c2]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4730 dB\n","truncated":false}}
%---
%[output:2c55653b]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4076 dB\n","truncated":false}}
%---
%[output:39a5bcb5]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9010\n","truncated":false}}
%---
%[output:572a855e]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4675 dB\n","truncated":false}}
%---
%[output:3a8ac5d7]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7700 dB\n","truncated":false}}
%---
%[output:6a1017d2]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4651 dB\n","truncated":false}}
%---
%[output:26eff442]
%   data: {"dataType":"text","outputData":{"text":"PSNR1 = 29.4704 dB\n","truncated":false}}
%---
%[output:2d6c3072]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3960 dB\n","truncated":false}}
%---
%[output:31afbdad]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9023\n","truncated":false}}
%---
%[output:8eaa0aad]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.4190 dB\n","truncated":false}}
%---
%[output:27034e9c]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.2253 dB\n","truncated":false}}
%---
%[output:5e2a5f22]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.5535 dB\n","truncated":false}}
%---
%[output:546995d1]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3748 dB\n","truncated":false}}
%---
%[output:1d8257c9]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.4186 dB\n","truncated":false}}
%---
%[output:83ecdebe]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9023\n","truncated":false}}
%---
%[output:2ea27465]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3686 dB\n","truncated":false}}
%---
%[output:9ef15240]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3894 dB\n","truncated":false}}
%---
%[output:18430226]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.2492 dB\n","truncated":false}}
%---
%[output:0b3bc812]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7092 dB\n","truncated":false}}
%---
%[output:10e2b815]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3599 dB\n","truncated":false}}
%---
%[output:0159f55c]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9025\n","truncated":false}}
%---
%[output:0208280f]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3543 dB\n","truncated":false}}
%---
%[output:2c50958d]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.0388 dB\n","truncated":false}}
%---
%[output:1f4c0936]
%   data: {"dataType":"text","outputData":{"text":"PSNR2 = 30.3711 dB\n","truncated":false}}
%---
%[output:549fa7ef]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.2816 dB\n","truncated":false}}
%---
%[output:967a8cf6]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7015 dB\n","truncated":false}}
%---
%[output:7f1d43fc]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9009\n","truncated":false}}
%---
%[output:2eaefea2]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.2676 dB\n","truncated":false}}
%---
%[output:7bff3441]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.3165 dB\n","truncated":false}}
%---
%[output:2a3e5a3a]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.0720 dB\n","truncated":false}}
%---
%[output:809f507a]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.2495 dB\n","truncated":false}}
%---
%[output:61fcca83]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.2400 dB\n","truncated":false}}
%---
%[output:2de57be9]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9020\n","truncated":false}}
%---
%[output:029eacfe]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7263 dB\n","truncated":false}}
%---
%[output:342100b3]
%   data: {"dataType":"text","outputData":{"text":"PSNR3 = 32.2722 dB\n","truncated":false}}
%---
%[output:3ac264e8]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.5763 dB\n","truncated":false}}
%---
%[output:3d3a4c73]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.6923 dB\n","truncated":false}}
%---
%[output:17b94478]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.6865 dB\n","truncated":false}}
%---
%[output:5858d92c]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9020\n","truncated":false}}
%---
%[output:96132402]
%   data: {"dataType":"text","outputData":{"text":"Average PSNR = 30.7046 dB\n","truncated":false}}
%---
%[output:8a52fd53]
%   data: {"dataType":"text","outputData":{"text":"Average SSIM = 0.9022\n","truncated":false}}
%---
%[output:91963e25]
%   data: {"dataType":"text","outputData":{"text":"\n========= FIR COMPARISON RESULTS =========\n","truncated":false}}
%---
%[output:082289a3]
%   data: {"dataType":"text","outputData":{"text":"    <strong>Taps<\/strong>        <strong>Window<\/strong>         <strong>AveragePSNR_dB<\/strong>    <strong>AverageSSIM<\/strong>\n    <strong>____<\/strong>    <strong>_______________<\/strong>    <strong>______________<\/strong>    <strong>___________<\/strong>\n\n      4     {'Rectangular'}        30.349          0.89984  \n      4     {'Hann'       }        30.802          0.90092  \n      4     {'Hamming'    }        30.806           0.9015  \n      4     {'Blackman'   }        30.802          0.90092  \n      8     {'Rectangular'}        30.503          0.90093  \n      8     {'Hann'       }        30.744          0.90268  \n      8     {'Hamming'    }        30.734           0.9027  \n      8     {'Blackman'   }         30.77          0.90271  \n     12     {'Rectangular'}        30.553          0.90095  \n     12     {'Hann'       }        30.709          0.90231  \n     12     {'Hamming'    }        30.701           0.9023  \n     12     {'Blackman'   }        30.726          0.90251  \n     16     {'Rectangular'}        30.576          0.90094  \n     16     {'Hann'       }        30.692          0.90203  \n     16     {'Hamming'    }        30.686          0.90201  \n     16     {'Blackman'   }        30.705          0.90223  \n\n","truncated":false}}
%---
%[output:820357f3]
%   data: {"dataType":"text","outputData":{"text":"\nBEST PSNR CONFIGURATION\n","truncated":false}}
%---
%[output:2d097538]
%   data: {"dataType":"text","outputData":{"text":"4 taps, Hamming window: 30.8056 dB\n","truncated":false}}
%---
%[output:6e68e2b8]
%   data: {"dataType":"text","outputData":{"text":"\nBEST SSIM CONFIGURATION\n","truncated":false}}
%---
%[output:0ad744ff]
%   data: {"dataType":"text","outputData":{"text":"8 taps, Blackman window: 0.9027\n","truncated":false}}
%---
%[output:25ff1708]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAApsAAAGSCAYAAACsdQoxAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ24VVWZx18QEApwRCi+AhsBaTRBwSDHrLRpGg0arQSpAQGFGPkw5LPUSkmQrxHBGEBAeBoFJ7ORrKamplGHIjEHs1LIDxC4CKgpmIgo87wb12Wfffc5Z+991j5nr7t\/+3nuo9y71tpr\/9e71vqtd301OXr06FHhQQEUQAEUQAEUQAEUQIEUFGgCbKagKkmiAAqgAAqgAAqgAAp4CgCbGAIKoAAKoAAKoAAKoEBqCgCbqUlLwiiAAiiAAiiAAiiAAsAmNoACKIACKIACKIACKJCaAsBmatKSMAqgAAqgAAqgAAqgALCJDaAACqAACqAACqAACqSmALCZmrQkjAIogAIogAIogAIoAGxiAyiAAiiAAiiAAiiAAqkpAGymJi0Jo4D7Cmzbtk1Gjhwpu3fvDv2YQYMGyZw5c6RVq1YFf3\/55Zdl9OjRsmXLloLf33333TJw4MCC3916662ybNky73fBv\/vTmT17tgwZMsQL9+tf\/1qGDRtWUuCwd5WKEJZmse9zv2SLf8Ebb7whM2bMkA0bNsjYsWNl+vTpjflz+TYUQIEqKABsVkFkXoECripQDjb1u\/r06SMrV66Udu3aRQJBPzRqBD9sBuGuEtjUtIPvCisHP1yF\/b1z586yevVq6dmzp6vFGCvfwGYsuQiMAigQQQFgM4JIBEGBvCoQBTb9UFcO3DRsEN78sBkExEphMwjCYeUYfH9YmCjp5NVG+G4UQAEUKKcAsFlOIf6OAjlWwA+bpaa4zXSrHw6DU7D+aWq\/xzEIe36wKwebxbyO69evl5kzZzYA22BR+r8vmFZwKUDw+8PAOvjNwfzr+zVf+pjv1P\/3LzkIpmH0Ua\/v5z73ObnqqqvqPyPMcxu2hCEIy\/7vXrJkiaxYscJb8qDh9N9z585tMI0e9r2lvL7Bcg1bkmBsQtO54447ZNWqVd57\/foYj3mOqyGfjgLOKwBsOl+EfAAKpKdAJbAZdfo5zLMYBq9hazYrhc1yUGq+f\/78+QVrTUt5fIvBcjGP6b59+xqsiS0F48F0\/HBabK1sEN6K5V\/TmjhxYoM1m6U81uUg3Z\/fYNhya2\/xKKdXt0kZBaqpALBZTbV5Fwo4pkDcaXT9PANwwU8ttn7S77nr27ev3HzzzfUeyVNOOaXe6xd3g5C+v9wGH\/+7wzY6hRWXH7z8MOTXKgyW\/aAVtk5V0zUeTj9A+sOG\/d6frtHe\/91+oDPeWX9eg0AXtmYzbNBRzItt8uvPlz9ssbz5f19q05hjVYjsogAKiAiwiRmgAAoUVSAKbIYBXal1kKWmiadNmybjx4\/3pnQ1Xf+\/48JmFM9qEtgs5e01sBc2Re7\/7jAA1EIIy4\/5XRAKiy0xMIUZ5o0Mg83gIKAcbJbStVSe\/FPmZsNVMR1KaUx1RQEUcE8BYNO9MiPHKFA1BcrBZrmjcaJ4OYOApaBpjjWaPHmy\/PznP\/fgMypsxpl6TQKbYdBkCiT4t3Ke2WKbpcK8fEGoD4PCchu0wmCz1FrUYh5W873B8i8FiWF\/K6YlsFm1Ks6LUKAqCgCbVZGZl6CAmwrY7PSLTaUGgU+VMuc8+lUrt2bT7yWLCpxJ1mxmGTbDpuf98F4JbGpZFINZUzbAppv1nFyjQNoKAJtpK0z6KOCwAnFh0+\/JLOUx88NgmHcxzKNaDjZVZj9wlluvqeHj7EaPAlSlptGj5L+SafTBgwfXQ3oxL3ClsBk05eAUv\/7drDsNTs+XmkYPenjj2p3DVYyso0AuFAA2c1HMfCQKJFMgbqdfbto9bPq12FR2qfM3S3kX\/cAb5VD3uOdsJt0gVClsqnalNgh17dq1HjYNaAe9xElhs9jayjDPcJINQsBmsvpJLBRwRQFg05WSIp8oUAMF4sKmZrHYOk2T\/SjrFDVs8BifKLCm8fwwGGWTULl1jppm0EtbCqr9HtW454SW8mwWK34\/gJbTPilsltMoyfFLfk80sFmDys0rUaCKCgCbVRSbV6GAawokgc0wUAzzaJrfldqkU8xLWcqzqen68x1lOt0PPv4yKhU3yaHuwbvdo4C3X59Ro0bJNddcU38uZ5jn1u+pLXZIe6lyLXVdZRjMljvSyuhZ7lB3\/5WgSe3OtfpFflEgLwoAm3kpab4TBVDASQWS7Jh38kPJNAqgQKNVINewGZymC9vBWmy6LKq3pNFaDh+GAihQFQWAzarIzEtQAAVSVCC3sGlAc+DAgTJ9+nRPYm3U9V5e\/3SOTtdNmTKl4HcplgdJowAKoECBAsAmBoECKOC6ArmFTV17tG7dOlm5cqW0a9fOK0cDoEOHDhWztkrDbdy4UaJeZee6QZB\/FECBbCkAbGarPMgNCqBAfAVyC5thUoXBpjb0+hjvZ3yJiYECKIACKIACKIAC+VUA2Hy37M0OzB07dtR7Ow18tm3bVh5++OF6Kyl3RV9+zYkvRwEUQAEUQAEUQIFCBYDNd9dqLlu2zFPGf56e2Rykm4GMZ9MAaLdu3UpOre\/cuVP0hwcFUAAFUAAFUAAF9OIF\/cnjA2z6Sj0MLsOMwpzxFzzo2YRVyJw6daps2rQpjzbFN6MACqAACqAACgQUGDBggMybNy+XwAlsBoxBNwQtXry45O5zA6UTJkyo30jkT8bAqBpVly5dclfhFLIXLVrkVao8fn\/uCrzIB2MHWIIqgB1gB9jB8XpQzEnV2K0E2EwRNvNqVOU8v429UvF9xxTADrAE7AAbMArkvT3I+\/fnFjZ1l\/nu3bsbrLvU36tR6JFIW7dulWHDhjW4F7nc2Zt5N6q8fz\/dC7CJDRxXgPYAa2DQweA7t7AZNhVu7v01d\/2G7VCPsq4z741r3r+frgXYxAaATWygUIG89wt5\/\/7cwqZWg+B1lZ07dw5dq2kOVTZVx8BoscYk70b1\/PPPy5o1a2TEiBFy6qmn0ubmVAHsIKcFH\/hs7AA7UAXybgd554Jcw2ZaTUDejerQoUNSV1cnnTp1kpYtW6YlM+lmXAHsIOMFVKXsYQdVEjrjr8m7HeSdC4DNFCpo3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2ubdqPLeqKRgUk4miR04WWzWM40dWJfUyQTzbgd55wJgM4Vqm3ejynujkoJJOZkkduBksVnPNHZgXVInE8y7HeSdC4DNFKpt3o0q741KCiblZJLYgZPFZj3T2IF1SZ1MMO92kHcuADZTqLZ5N6q8NyopmJSTSWIHThab9UxjB9YldTLBvNtB3rkA2Eyh2rpmVDt37hT9sfF07dpV2rdvL3V1ddKpUyfZv3+\/1bQ1ff9jO+\/B9G1oktc08t655LXcg9+NHWAJqkDe7cA1LrBttcCmbUVFxCWjUlibOnWqbNq0yYoSAwYMkJtvvllOOOEEefvtt+WGG26wmva8efPEAGEaefenb0UQhxKxDe4MOhwqfF9WsYNjYmg7k+fBpy07MDr6YTOPTgiXuCCNlgvYTEFVl4zK5HXqqfukY4sjFamx5UBLWVt3sqxatUq6d+8u27dvl1GjRsk3hh2Vzu0qSloe+5PI8p80kbvvvlsGDhzoJWbyfsGbE6XN0fdV9IK6pk\/Kb1usK0i\/ogQdi5wGuDPocMwIRLxZCAafx8pNB85ZHnzagsEwsLZpB0ZHM\/jMqxPCJS5Io+XKNWy+\/PLLMnr0aNmyZYunbZ8+fWTlypXSrl0hGRkjMQUwe\/ZsGTJkSNHycMmoTF4X9KqTPm0OVWRjCpvXbe3UADaXjT8q\/XpUlLQHm2OXhMPmJYdmSae3z6zoBXUnPCkPtrw+07CZZufCoOOY+bgw6MAOJPeDT5swGAbWttoD44BQJ0Hfvn295VV5dUK4xAUVdaZFIucWNg1oqpds+vTpnjy33nqrbNiwQVavXi09e\/as955NmTKl\/nfbtm2TkSNHyoQJE4oCp0tGBWy+CxkZh81qdS4MOrI96MAOjtXXvA8+Tbv9l3NGyTvvaV8RGzTb\/5S0fOqB0FmjStsD44AIg828OSFc4oKKDArYLFRg\/fr1sm7dugJPpgHQoUOHeiAZBqSaSlhcf+q2jaoanoxKGxX9fjybaVTRY2m60rlgB+nZAHZwXFtg89cybNgwOXj+NDnS\/vSKjK7Z\/qel9SNzgc0QFW3OeNnmgooKvQaRc+vZDNM6CJvGizl\/\/vz6dYIar9jvTZo2jQpPBp4MP2RkvXMBNtNtxU3bgh3ke1mNK3aAZ\/N4e2CTC9JtZdJJHdh8V9c33nhDZsyYITt27Kj3dqpx+KfQTRGUm0q3aVR4tIBNYBOPVnAgC2wCm3g23VpOYZML0sHBdFMFNt9dq7ls2TJP6eBu5zDYDHpAg0VkjGrSpEnejkb\/07FjR9GfqM\/mzZtl+PDhVqdLdLe42dFt0rc5jb506VJvN7p6ZceMGSM21+aE5d3mBiF\/+lHLqBrhsIPCQQd2YG\/6lPagYQ0206e0B5VtHPUvrdINQnv27MlFv6DfqT\/+Z9euXd5JD37GqEbfkZV3AJu+kjAey0GDBnmbhop5NqPCZlghjxgxwoPHqI\/ulJ88ebJV2Fy4cKG3814fk75N2NTd+p07d5bdu3fLzJkzrcJmWN5twqY\/\/ahlVI1w2EEhbGIH9mCT9qA4bNIe2IFN1bF3796yb9++XPQLa9eulTVr1oR2DcBmNXpMB96hm38WL17s7T5\/6aWXKppG1zPaunTpUpFnU4FXz6q0OW2mFaF\/\/\/5evkz6NmFz+fLl3mHIesTFuHHjrMJmWN5twqY\/\/SyZK3ZQCJvYgT3YpD0oDpu0B3ZgU3U844wz5MUXX8xFvxDm2dSLUxYtWoRnM0sday3z4odNzYcec1TLDUKuLAT3vKScs5ma6WIHhbAZdri\/zUFHVr0P2AF2YJwErNlkzWZqHU4KCed2Gl3P1NRp3jlz5kirVq3qpdXfa4Ouh7vro4e+m6OQTKBqHn1E50Ln4lLnwqAjhVbalyTtAe2BS+0Bu9GPV142CB09ejTd5jGbqYftKFeI1DWG\/huCzO+Mp6PcTnR\/Q2DDO0LnQufiUucCbKbb3tEe0B641B4Am8CmUSC3nk0VIHhdpW5q8d8eZEQyDbz5dzWvq6RzoXNxqXMBNoFNowDLatKzBVf6BWAT2AQ202sH6m97wbO53dvcZPPoI9bq2dsYEqalzY1iemyMHoFl7kLGDipvdFyBDAYdlZd1qRRcsQNgE9gENlNsC2yuzXClUaFzSdGgfNdV2jyVANhsWGY2r6dLwyJoD5jpcGmmA9gENoHNNHqCd9MENvFo2TYvIAPIcAkyGHzabgEK03OlPQA2gU1gM8W2ANgENm2blyudC5Bhu+TdhAzsADvw24DOougNQnV1dbldVmOTC9K1rnRSz\/UGoXQkPXZQup6BxppN1mzasjFgE88mns3jtemxP3E3Oudscs6mrf6lGukAmymoDGzi2bRtVsAmsAlsAptGAVfaA6bRmUZnGt02DfjSAzaBTdvm5UrnwvSp7ZJnGj1MUTybx2bPsr5hENgENoHNFPsEYBPYtG1ewCaeTTybeDbxbBa2rC4NOmxyge3+pRrpMY2egso2jQrIADKADCDDNcjAw51CxxIye4Zn0521uza5IF3rSid1YDMFXW0aFbAJbAKbwCaw6a5HK4Uupn4TKrAJbKZhX2mkCWymoCqwyTS6bbNi0MGgg0EHgw7XBh2s2Txusza5wHb\/Uo30gM0UVLZpVEAGkAFkABmuQQbT6Cl0LEyjNxCVNZvp2pnN1IFNm2q+mxawiWfTtlkx6GDQwaCDQYdrgw48m3g2jQLApm0q8N1jzaHuHOpuy7yATWAT2AQ2gc3CFhXPpq0eJv10gM0UNMaziWfTtlkBm8AmsAlsApvApu2+pVrpAZspKA1sApu2zQrYBDaBTWAT2AQ2bfct1UoP2ExBaWAT2LRtVsAmsAlsApvAJrBpu2+pVnrAZgpKA5vApm2zAjaBTWAT2AQ2gU3bfUu10gM2U1Aa2AQ2bZsVsAlsApvAJrAJbNruW6qVHrCZgtLAJrBp26yATWAT2AQ2gU1g03bfUq30gM0UlAY2gU3bZgVsApvAJrAJbAKbtvuWaqUHbKagNLAJbNo2K2AT2AQ2gU1gE9i03bdUKz1gMwWlgU1g07ZZAZvAJrAJbAKbwKbtvqVa6QGbKSgNbAKbts0K2AQ2gU1gE9gENm33LdVKD9hMQWlgE9i0bVbAJrAJbAKbwCawabtvqVZ6wGYKSgObwKZtswI2gU1gE9gENoFN231LtdIDNlNQGtgENm2bFbAJbAKbwCawCWza7luqlR6wmYLSwCawadusgE1gE9gENoFNYNN231Kt9JyFzTfeeEP0p1WrVt5Plh5gE9i0bY\/AJrAJbAKbwCawabtvqVZ6zsDmgQMH5Gc\/+5n84Ac\/kE2bNslbb71Vr1Hz5s1lwIABMmTIELngggukTZs21dIv9D3AJrBp2wCBTWAT2AQ2gU1g03bfUq30Mg+br7\/+uqxevVoWL17sadK3b18555xzpF27dnLaaafJM888Iy+\/\/LIHoE8++aTn5bz66qtlxIgRNYNOYBPYtF2BgU1gE9gENoFNYNN231Kt9DILm0ePHpWHHnpIbrrpJg8qR40a5UFmixYtimqj0+qbN2+Wu+66S55++mn59re\/LR\/\/+MerpWX9e4BNYNO20QGbwCawCWwCm8Cm7b6lWullFjbVW6nezKFDh0qvXr2kSZMmkTVRUN26dausW7dOvvGNb0SOZysgsAls2rIl1zoXze+WAy3luq2dZNUq7AA7wA5s24BLgw7TFtx9993erGRdXZ1s377dcx4tG39U+vWoTJ3H\/iQydkkT0fQHDhzoJWb630sOzZJOb59Z0QvqTnhSHmx5fUH6SRO0yQVJ81DLeJmFzVqKUum7bRoVHi08Wi51LsBmpa1H6fi0B7QHLrUHwObx+myTC9JtZdJJHdhMQVebRkXnQufiUucCbKbQoPiSpD2gPXCpPQA2gU2jgLOwuW\/fPtm2bZv3HT179pQOHTrEbuU1\/siRI2X37t1e3M6dO3ubkTQ98wTDmN8PGjRI5syZE3rsErDJ9GlsYywTAcgAMlyCDAYdtluAwvRcaQ+ATWDTCdjUtZe\/\/\/3vZdmyZXLw4EEZPXq0nHfeed7xRzNnzqw\/\/qhZs2Zy1VVXyYQJEyKfuWkq6+zZs70jk\/RZv369l25w\/ceUKVMaQGippgTYBDZtdzWudC5Ahu2SdxMysAPswG8DrNk8vpbUzxfpWkm2Us+0Z1M72LFjx3qKvf\/975cXX3xRxo0bJ9\/97nflb\/7mb7zNQ\/roRqBf\/vKXctttt8nFF18cSeFbb73V82j6vZO6m33GjBmeh3P69On1ALpx48aiXsywlwGbwGYkI4wRCNjEs4ln83iFcWljSIxqHjmoK+0Bnk08m5n3bB4+fFhuvPFG7xxNhUgFQD3Uffz48d7B7QsXLpS2bdt636EHvl933XXeuZq33HKLnHjiiZErrT+ggU39nYFQhVJ9DHxGSRjYBDaj2EmcMK50Lni04pRq\/LDYAYMOlwYdwCawmXnYfOWVV2TMmDHSv3\/\/etDTYxN0Kl3PzgzCn0KhNsQrV670DnxP8uhxS5q+HqGg6Zt\/K9Q+\/PDD9Umqt7UUfJoOYdKkSd7NRv6nY8eOoj9RHz03dPjw4XLw\/GlypP3pUaOFhmu2\/2lp\/chc70gac0yESX9Brzrp0+ZQRembhmXp0qXSvXt32blzp1eGNo+4CMu7zSMu\/OlXJIblyNhBIWRgB7QHeuQNdpBtO\/Afg6ZHH+3ZsycX\/YJ+p\/74n127dsnUqVOtHKNkuXupSnKZnUYPgp+qEfY7o5IN2NQ0NmzYUL8+02wO0s1ABi5NHrp161Z2g1BYCerNRgqPUZ8tW7bI5MmTrcKmeoX79OnjZcGkbxM2dR2seqJ1mYKugbUJm2F5twmb\/vSjllE1wmEHhbCJHdiDDNqDhjXYnK9Ie1CZE8LApurYu3dv0Y29eegX1q5dK2vWrAntGlizWY0eM8Y7qg2bZnOQf8NQseyWmyY3f583b5506dKlIs+mpqUH4Nr0bGpFUI+xPiZ9m7C5fPly6dq1q3d4r66xtQmbYXm3CZv+9GOYa+pBsYNC2MQO7MEm7UFx2KQ9sAObquMZZ5zh7bvIQ78Q5tnUK7UXLVqEZzP13jLmC6oJm3FAUz\/DeDx197vZye7\/vHIwGkcK1mgVQoYLN0XEKd+oYbED7MAMDocNG2Z18BlWp2wOPrlJKmotjx7OlfaANZvHy9QmF0S3lOyEzPw0unrIDNDpRqD58+d7O9GDkKfAqGsE467Z1KlzPVopikfTFBuwGW7AXFOYXsV2pXNRBbAD7AA7SM8GXBp0AJvAplEg87Cpa9WiProOMQ5sGtAstoai2EhEf1\/q7E2bIxggA4+WS50LkBG1tUoWjvaA9sCl9gDYBDYzD5vvvPOOvPrqq6IHu0d9mjRpIieddJI0bdq0bJSwA9yDkcxRSDt27KiH2LBNQ8F4wCZHH5U1wJgBgAwgwyXIYNARs4LHDO5KewBsApuZh82YdS9WcLMetJjXNOghNR5Q85JyU+7AJrAZyyAjBHalcwEyIhRmBUGwAwYdLg06gE1gM9ewWUFbHykqsAlsRjKUGIGADCDDJchg0BGjcicI6kp7AGwCm5mHzXLex7D6GXfNZoI6HikKsAlsRjKUGIFc6VyAjBiFmiAodsCgw6VBB7AJbGYeNg8ePCj33nuvt27TPHp1pV5Z+ZnPfEZOPfVU79f79++XH\/7wh9KiRQvRA9OvvPJKad26dYJm3F4UYBPYtGdNx1ICMoAM7OB4reJu9F+LC0dgAZvAZuZhM9hZP\/LII7JixQqZNWuWfOADHyj4s14DpXej6zWWX\/nKV0Q3CtXyATaBTdv2B2wCm8AmsGkUcKU9ADaBTadgU72cetTQJz\/5ydBD1PVjdHf5gw8+KEuWLBG9y7yWD7AJbNq2P1c6F\/1uztm0XfoNOyybN4pxqHvD8jLXVWb1akFX2gNgE9h0CjbN+s2rrrpKLrnkktCWXGFz3bp1sc7ZTKtLADaBTdu25UrnAmzaLvnC9LADPNwuebiBTWDTKdg8fPiw3Hjjjd69qnPnzpUOHToUtMB79+6ViRMnSq9eveT666\/31m\/W8gE2gU3b9gdkABkuQQaDDtstgJuDDmAT2HQKNjWzW7dulXHjxsmhQ4dk8ODBcu6553rf8Oijj8oDDzwgLVu2lKVLl3rAWesH2AQ2bdsgsAlsApvHaxUbhNggpNbgkh3Y5ALb\/Us10svsdZVhH683+cybN09++tOfyltvveUFad68uVx66aUyYcIE6dKlSzU0K\/sOm0YFZAAZQAaQYRSgPaA9cKk9wLOJZ9M5z6af8I4cOSIHDhzwftWmTRtp1qxZWQCsZgBgE8+mbXsDMoAMlyBD88pGMdutQENwyfpGMWAT2Mw8bL755pveGZqdO3dOdJSR3qmu8YPrO9Or\/ukYFZABZAAZeDbxbBa23C5Nn6bR57jSLwCb6XBBGjaVdpqZnUZXz+W1114rHTt29NZqdu3aNZIWCpkvvPCCLFu2TPbs2ePtTq\/2g2cTz6Ztm3Olc8GjZbvkC9PDDhh8ujT4BDaBzcx7NjWDr7\/+uqxevdoDxzPPPNO7MeGcc86R97\/\/\/QVT57pbfffu3d4tK\/fff7\/83\/\/9n7eGc+TIkfLe97433dY\/JHVgE9i0bXRABpDhEmQw6LDdArg56AA2gU0nYNNkcufOnfIc7Ex5AAAgAElEQVRv\/\/ZvogfsmrWaYVVZ128OHz5cvvSlL3ke0Vo9wCawadv2gE1gE9g8XquYRmc3ulqDS3Zgkwts9y\/VSC+z0+hhH68ezN\/\/\/veyefNm0YPen3vuOenZs6e3LrNPnz7yoQ99qOZnbPo7BBu3TwAZQAaQAWQYBWgPaA9cag\/wbOLZdMqzWQ3qtvkOmyMYOhc6F5c6F6ZPbbYkDdOiPaA9cKk9ADaBTWAzxT4B2GQa3bZ5ARlAhkuQwaDDdgtQmJ4r7QGwCWwCmym2BcAmsGnbvFzpXIAM2yXvJmRgB9iB3wZ0SVnfvn2lrq5Otm\/fLqNGjZJl449Kvx6V6cSazcr0q2Zsp9ZsVlOYSt4FbAKbldhPWFxgE88mns3jNcMlyLDdFrhkB3g28Wzi2UyjBXg3TWAT2LRtXsAmsOkSZODZtN0CuOnhBjaBTWAzxbYA2AQ2bZsXsAlsApt4No0CrrQHwCaw2ehg85133pHHHntMevfu7d2XXssH2AQ2bdufK50LHi3bJe+mRws7wA78NsCaTfEundGLaWwciZiudaWTeubXbOp5mj\/84Q9l48aNHkhedtll0q1btwI1du3aJd\/61rdk37593vWU7dq1S0etiKnaNCogA48WHi08Wq55tIDNiJ1FwmCu9At4NvFsOuHZ3Lp1q3cvuh7ebh71Wt5xxx1y\/vnny5EjR+QHP\/iB3HrrrfLqq6\/KiBEjZPLkydKqVauEVdhONGATz6YdS2rYUB08f5ocaX96Rck32\/+0tH5kbsEI29jsgl510qfNoYrSNx3MqlXYQUVChkR2BTKATdsl76aHG9gENjMPmwqSCpE\/\/vGPZc6cOfKRj3xE1IM5a9Ysad68udx8882yYsUKufPOO6V79+5e2HPPPVeaNGmSbi2PkDqwCWREMJNYQYAMPNx4uPFwu+bhBjaBzczD5muvvSbjx4\/3AFL\/ayDyySeflKlTp8rZZ58t3\/ve9+TLX\/6yXHvttdK2bdtYnXeagYFNYNO2fQGbwCawWTlsHmiyVw423Vu2emq4h068XSZNmiQDBgwoG77aAdTxov3god6D5Uj73hW9vulf9st7fruq4FtN+sM7vVLRTMeew81k3vMdvLT79esn+\/fvl4MHD8qNN94oYz5T+Tmbu18W+dbdTULzfs7hodLpnTMr0qaYHXTt2lX0J85jkwvivDcrYTO7ZlPXao4ePVomTJggF154Yb1e5vfPPvust05z8ODB0rRp06zo6eXDplEBGUAGkFE5ZERtIOpOeFIebHl9Zhfx0x4kbw8UHLb0+Fd5avdvo5oD4VAgVAEdgMybNy8WcNrkAheLJfOwOX36dBk4cGAD2FQA9Xs8syS+TaOic0neucSxCSDj2E5J1mwCm43VDkwdV0jo0qVLnOaBsChQr8CmTZtk0aJFsQekNrnAxeJwFjaDHs8siW\/TqIBNYBPPJp5NowDtQfL2IOsDyiz1YeSluAJJ+\/ek8RpLWTgLm0GPZ5YKxKZR0bkk71zi2ETWOyLsADtg0FHZoKNYHd+5c6foj+tPknWErn9zLfKftH9PGq8W35jGOzMPm9dcc42cc8459d+uRxxNnDjRWxDs\/70G0E1EJ510Us3XcNo0KiADyAAyKoOMOA0ng47Gu5wirGwVMnWjjU6Nuv4kWUfo+jfXIv9J+\/ek8WrxjWm8M\/OwuWXLlsjf3adPHw51L6EW5ysWFwfIaLyQEbkBERHsoPHaQVjZGgD4yzmj5J33tI9jKpkK22z\/U9LyqQcK1hG+8cYbMmPGDNmwYUNoXm3fZKObd1955RU57bTTMqONHomoj86E2nqSQmPSeLbyXet0MgubejzCvffe6x3WHvVRr+bll18urVu3jhollXA2jQrPJp5NPJt4No0CtAfJ24NSsGnjsoRUOpOIiYY5EgxsnnfeeTJkyJCClGz2UZqwOSUma8vbgM2IBlSFYJmFzSp8e2qvsFmR6VySdy5xChiPVuP1aGEH4QrkbaYD2Cy0A5sgBmyWb2VsckH5t2UvRK5hc9u2bTJy5EjZvXu3VzKdO3eW1atXS8+ePUNHgeaXs2fPbjBS9EewaVTAJrCJZxPPJp7Nws7zsT+JjF3SJPTK1UsOzZJObzc8zBvYPK6h8Xpqn2emmIPT7mHL0tavXy8zZ86sT0j7wr\/7u7\/zzsQ2S97Gjh3rpWn6Ln\/J+ftOA7v692XLlnnBBg0a5N0YaK6c9r9P83PRRReJ9tsaRtfbTps2TebOnVvfZ+vf\/L8LArX+27wr2OcrMF933XVywQUXeDcUFluWl7R\/Txove9iYLEeZhc0333xTnnjiCdH\/Rn1OPPFEOeuss0T\/W+4xBe83fmPY\/rUsGm7KlCn1EGoAVY9eCk5NBDsEG2tigE1gE9gENoFNYLNUn5ZkGt3frxnP5NChQ+v7Ne0P161bV78PQv+9ePHiBn3h\/PnzpVevXh5wmmn0IPRp3oPxDfiZfjKYh2Dfa\/pCA6RxYVPfv3HjxnqYNXCteVN41X\/rN3Tr1q0AeIO6J4XGpPHKsYwrf88sbBrDS2uDkBq6ejT9o6jgaM\/kQQ+V9y8wDlZCW8YYZjTAJrAJbAKbwCawmRQ2wzYIBWfxtJ+55557QvtDXfOpN\/XpZqOw9Z+aryjT6GFex2AfbDyReuJM2Pv8\/XZc2AzTzw+gBjb9wF2qT47rTAI2jx49mkUyjurZ\/MMf\/iD\/8i\/\/4nlAr7zySu8YC+OCj\/tdwZGOGrNOs+vIzX+LkfFuBn+PZ7OlXLe1k6xaxd3ocW2vXHgGHQw6GHRUNuhgGr34VcrB6XF\/e+SfKi+2AagYbAadRn7IDVszan539dVXF3hKTX78cJgUNoPT+8ZTamCz3CanpNCYNF65vsGVv2fWs1lOQIXLNWvWyIIFC6RDhw5y0003ySc+8YmKztgMejKDbnyTp3JT6cao9CxQPfvM\/3Ts2FH0J+qzefNmGT58uNjYLWmmWhQGDTyb9G1eU7h06VLp3r27t6ZmzJgxsmz8UenXI+oXh4cza7TC8l5sjVacN5qOyJ9+nPhph8UOCmETO5gmR9qfXpHZ5a09ADaPmUvYcrGwmT6\/cZXzXAb\/7t8PYZaqlVtPqe9LEzbNd\/uBN8yzGRU2S\/UVe\/bsEf3xP7t27fKcYXE9ohVV8gxFdhI2t27dKjfccIM8+uijcsUVV8i1117rAWeljxq6TjmYTULFYDNsfYv\/3WELo83fR4wY4cFj1EeXEUyePNkqbC5cuNBb\/KyPSd8mbGrjohVap0h0MblN2AzLu03Y9KcftYyqEQ47KIRN7MAebOalPQA2j9UhM4O3Y8eOgvWY\/vWZwTat1DFKGjYIm8H1kRom2J+W8mwmnUYv9o4o6cX1bJbqK9auXes5w8IeYLMaPWaF70jDm2myZEY9\/g1DlcLmvHnzpEuXLgVfHdezqXkYNWqUVdjUitC\/f38vXyZ9m7C5fPly0avTtm\/fLuPGjbMKm2F5twmb\/vQrNFer0bGDQtjEDuzBZl7aA2DzeJMUnJ0L258QXC4W3KvgB9Dgms5gWJOW5sA4c0rBptnN7t\/EFNwgZODQ7Kkw+XnssccavMPApn\/3venzk06jl+orwjybekvVokWL8Gxa7RlTSOx3v\/udzJo1Sx5\/\/HHPmzl+\/Hgr3kzNahhoho3EzGdFnUa3MYJhrV4hZARPChg2bJjYhE0bZZaC+dcfI2JzOUWYljYHHazdtW8JtAfJ2wNgs9Aei+0G92\/KDbaHYUcfmVNZgvB2++231x8zpJCnexz0nWYDTjnY9PfN+v\/Bo490b4Z\/ul7f8bWvfU1WrFhRfxyS\/x3B9aMKmZ\/73Ofkxhtv9OD0lFNOCV0nGqzFSddeJo1nvxWpTYqZn0bX0crKlSu9IxfUmNQwPv7xj1e0NtMvtalwYWdnFtsIxAahcGPdcoANQmlVYyAjOWTEKRMO92+8h\/uXgs1DvQfLkfa945hKpsI2\/ct+ec9vVzV6r5kC7fPPP2\/1+sm4BZkUGpPGi5u\/rIbPNGym6c3UAgmO7IKFVGxtJkcfAZvVrtDAJrBpZlvUm4+H286h7rqJUTdt6BSn649uRtWlW7qEyfUn7ND5ck6ean1zUmhMGq9a35X2ezILm36Xd\/PmzaVHjx7SokWLknpoJdNd6SeffHJZ3cJ25IVFCoYrN4Xu7xBsTMkCGUAGkHG8Zia5OaZsY+ALgGczX55NLXoFTv1x\/dH+rzGApimHsLO2bfSplZZzUmhMGq\/S\/GYlfmZh85VXXvGmzOM0AlFhs9yB8cFrqoK7y7muEs9mtSswgw4GHQw6Kht0ZH0gUe02hfclUyApNCaNlyyX2YuVWdjMnlTRc2TTqIAMIAPIqAwyotdckawDCe1B8vYg62Ubx04JWzsFkvbvSePV7kvtvhnYtKunl5pNo6JzSd65xCnarHdE2AF2wKCjskFHsTrONHqclpKwSfv3pPEai+LOwqaeualXVb7++uvywQ9+0Nup3qRJk0yUi02jAjKADCCjMsiI0ygw6MjXmk0Fza9eebk89mzhbS9xbCYrYRvTBqGsaBqWj6T9e9J4WdYiTt4yD5vPPPOMt2v80KFDsmTJEmnbtq3s3btX9LDXRx55xPvWZs2aeTsK9R5z\/f9aPzaNCtgENoFNYNMoQHuQvD0odfTR1FP3SccWR2rddSR+vx47t7bu5IKjj8rd+hN2zmXiDOQoYtL+PWm8xiJtpmFTQVPv1t6\/f7\/80z\/9k3zlK1+Rli1beqfw6+n9t9xyi5x77rly5513yl133eUd5qpncNb6sWlUdC7JO5c4doBHq\/F6tLCDcAXM3eh5Ody\/FGzauMwgjp3ZDmvOOPaXJbBpW+Vj6SXt35PGS+crqp9qZmHz6NGjnidTjx7SQ91PP\/10Tx29Bmrs2LHyqU99yrtFSKfOdef6NddcI926dfOOPip3RFLaMts0KmAT2MSziWcTz2Zhq53kCCxgs1BDPJvJSCBp\/540XrJcZi9WZmHz4MGDMnnyZOnUqZPccMMN9dPjel2lejv1\/u2zzz67XlGtOJs3b\/Z+H+WczTSLwqZRAZvAJrAJbAKbwGapPsuGZ9P0Nf73+I\/5M3Cqf1+2bJkXzNwrrldHqmNo48aN0rdvX7n55pu9vwePEUyz361W2kn796TxqvVdab8ns7BpzsIcOHBgwdVU69atk3vuucebOu\/QoUMBbGphqhe0Xbt2aetWMn2bRgVsApvAJrAJbAKbacKmXlYybdq0+jvF9V0Kj3pNtN4b3rNnzwY37gVv2DMXoBhADbsFqKYds6WXJ+3fk8azlO2aJ5NZ2NSpcfVg9u\/fvx42Dx8+7B30ro9\/ulyn3NXAn3rqqfpNRLVU1qZRAZvAJrAJbAKbwGZS2NywYUPRqLokbfr06aF\/DwKoejZ3794tc+bMEfVk6uOfig+7xtl4O\/1xatk323h30v49aTwbec5CGpmFTQOWOnpasGCBtGnTRnbt2iVXXXWVXHnllTJkyJB6\/VizGc2U8rYhIJoqx0KxQYgNQtjB8c0PNjbMmKndVatWSffu3WX79u0yatQoWTb+qPTrEad2NgzLms1CTWxMo2uKwdv19EhBv2dTw\/jhNAibOo3uB0tg83g5AZvqFszo84tf\/MLbgX7ZZZfJpz\/9afnBD37geS91Xeapp57q5VrP29QdeOrZvO222+Tiiy+u+dfYNCo8m3g28Wzi2cSziWczqWfzvPPOK3DOmHT8oKheTD06UD2XZho8zLMJbLIbPSlgZdazqR905MgRb1Q1b9487\/9POeUUryJcdNFF9bvQx40bJ7\/5zW\/k8ssvl+uvv15at26dVAtr8YDNbHgy4hQonk08m3g28WzGaTOyFLZSz2aYB1L7MT3PGs9mYUkn7d+TxsuSnVWSl0zDpvkw3Zn+2muvebvMzVoR\/ZsuQP7Od74jvXr18jyfJ554YiVaWItr06jwbOLZxLOJZxPPJp7NND2bwfWWxtOp7wQ2gU0bcOQEbNr40GqmAWzi2bRtbww6GHQw6Khs0ME5m4WtUvCcTf23OdJI12rOnz\/f2wA0dOhQbxo+7FxO1mwOjNzU2+SCyC\/NUMDMw6YuKdV1JM8995y3SehDH\/pQzQ9tL1d+No0KyAAygIzKIKNcffX\/neUUjXc5RSnYHN7pFenT5lAcU8lU2D2Hm8m85zsUXFeZqQw2oswk7d+Txmss0mUaNnX6fNasWXLvvffW6627GufOnetdU5nVx6ZRAZvAJrAJbBoFaA+StwdhsLlz506ZOnWqbNq0KavdSeR8DRgwwNvf0LVr18hxCBhfgaT9e9J48XOYzRiZhk29\/1xh84tf\/KK3JlOPZbj99tu9dZu6Iz2rlcqmUdG5JO9c4lQ5PFqN16OFHYQrkLej0IrVcQVO\/XH90f4wq32i69r685+0f08ar7Fol1nYfP3112XGjBkeWOrVV2bzz2OPPeads7lo0SK58MILM1kONo0K2AQ28Wzi2cSzWdjU2zpnM5MdCJnKtAJJ+\/ek8TItRozMZRY2i11XuW\/fPu9g9yuuuMJbuJzFx6ZRAZvAJrAJbAKb6cEmns0s9qLZzVPS\/j1pvOwqES9nzsFmMQiN99nphrZpVMAmsAlsApvAZjqwyZrNdPvCxph60v49abzGoiGwmUJJ2jQqYBPYBDaBTWAzHdg07es3hh2Vzu1S6AyqlKQuK1j+kyYFu9H1HGpdihblBqEqZbPsa8KOVyobqcoBkvbvSeNV+fNSex2wmYK0No0K2AQ2gU1gE9hMFzZt3NeeQlcSOcmwNawuwmbkD65hwKT9e9J4NfxUq6\/OPGz27dtXJkyYUP\/Rr776qkycOFH69+9f8HsN0KRJEznppJOkadOmVkWKm5hNowI2gU1gE9gENoHNUv0QsBm3l04ePmn\/njRe8pxmK2bmYXPLli2RFevTp4+sXLlS2rWr7XyITaMCNoFNYBPYBDaBzbRh0\/Q1\/vfMnj3buz1IH53iPnDggPezYcMG73f693POOUdGjhzpXb6iNw+Z6y31ystp06bJ4MGDvRNl9DF99IoVK+pvKxo7dqxMnz69\/h36P\/pvc1+7OpyC8U0fb67V1Hfrc8MNN8gDDzzgncXds2fPyOwQJ2DS\/j1pvDh5y3LYzMLmm2++KU888YTof6M+ejzSWWedVfM70m0aFbAJbAKbwCawCWymCZsGDP2QprC3ePHigrvR9TrLu+++WwYOHCimbxo0aJDMmTPHO6ZQgVTBT\/+tm68UQvv16+f9Wx9dQ6qgatIwsKhXY2qawesvZ86c6QGtAq9ZFqBAqzBqNgub6zTNv\/XEGgO8UdkhTrik\/XvSeHHyluWwmYXNLItWLm82jQrYBDaBTWAT2AQ2k8Km8UKGxfd7FYN\/DwKoHyQVLA3cKfgpKOpjvJF+2DQgGfx7WBpB2Fy3bl3BbKU\/ffVgbty4sR50TTs5ZcoUYLMcpNTg78BmCqIDm6tErxXdvn27jBo1Smwsvk9yiHOcouUGIW4QUnvBDhqvHZS6G91GGxWnvbEd1taaTQOQZvmaf1o8uFM8CmzqNHrQW+oHxGAaQdgMwqQfNvU2QX3MFLz+f5iH1rbWSfv3pPFs579W6QGbKShv06jwbOLZxLOJZxPPJp7NpJ7NKEcf+dc+mmnrMM+mH+6AzWPLCaI+Nrkg6juzFA7YTKE0bBoVsAlsApvAJrAJbKYJm36PoU5vmzbHPyWdNc8m0+gpwEuKSQKbKYgLbDKNbtusGHQw6GDQUdmgg2n0wlap1PpI4+nUGGazTdZgUzcMjR492ru2WjcQsUHIdq9jNz1g066e9SPCYcOGFdzmkPQ1QAaQAWRUBhlx6h5rNlmzGcdeshLWxppNhUndba6PrtXUjT36OwNzWYNN9cBy9FFWLLB8PpyDzaNHj3rnfL399tveAe56kPuRI0ekefPm5b+2SiHwbOLZtG1qDDoYdDDoqGzQUcqzOeYzR6VfD9u1tnrp7X5Z5Ft3F15XWb23Z+dNCp+33HKLLFiwILXztpP270njZUfdynLiDGwqZD766KPyjW98Q55++un6w2FbtGghuq7k3HPPlREjRkizZs0qU8RCbJtGBWQAGUBGZZARp0rj2cyXZ1PPgpw6daps2rQpjplkMuyAAQNk3rx50rVr10zmz3amdJ2p\/2ik4Dmctt9n0kvavyeNl9Z3VDtdZ2Dz8ccflyuvvFIuuugi6dGjh\/zsZz\/zzt9SV\/rChQvlu9\/9rjeaufjii6utYYP32TQqYBPYBDaBzWBHd\/D8aXKk\/ekVtXXN9j8trR+ZW7Dcx7Q3C3rVSZ82hypKf8uBlnLd1k6yalU2ZjqKDSQUOPXH9UchMy+gacrKP\/Wvvyt1bqit8k3avyeNZyvftU7HCdg8fPiw3HjjjfLOO+9411YpeKqRmaspdRpdXecvvfRSwQGvccQN242n8YNrQkya\/lsTgu+xaVTAJrAJbAKbwGZhK5vk3N2se63j9FeErZ0CSfv3pPFq96V23+wEbJpdZldddZVccskl3jVZfthUSR588EG58847E92NbowgDCD1b3FvJLBpVMAmsAlsApvAJrBpt+sntaQKJO3fk8ZLms+sxXMKNocPHy6XXnppKGyqZ\/J73\/ueLF++XE4++eTIOvvd8GGwWczjWeoFNo0K2AQ2gU1gE9hMDzaZRo\/cXRJQpP5OeHO\/e1RRbHJB1HdmKZwTsGmm0V9\/\/XXR2w2efPLJAs\/m3r17ZeLEiXLqqafKTTfdJLppKMqjoKn3xuo5Yt\/\/\/vdl9+7dDabhg8c9REnXplEBm8AmsAlsApvpwCYbhKL0aITxK5C0f08ar7Go7wRsqthbt26VcePGebvQdYPQfffd501vP\/\/8897moIMHD8pdd90lZ599dqKyUagMwqaZvm\/btq08\/PDD9emWW4Rs06iATWAT2AQ2gc10YNO0rxe8OVHaHH1for4jC5Hqmj4pv22xrmCzl9mdrQ6V4OP3yoVdO1nJNyVx0FTyvmrHTdq\/J41X7e9L633OwKYK8Lvf\/U5mzZrlHYHkf04\/\/XTPo6nHHyV9wmDTbA7S6fXp06d7SZuK2a1bt6KbkYxRTZo0SfQ4Cv\/TsWNH0Z+oz+bNm0WXD9jcfaq7Q82driZ9m7tPly5dKt27d\/d2eI4ZM0aWja\/8DDuzISAs75ccmiWd3j4zqqSh4czmAX\/6FSVoOTJ2UDjowA7s7UbPS3tQ6pxNG22I5SofK7mwbzOwGbwbPQg9wGYsqeun0Uv1FXv27BH98T+7du3yjtmKO\/0eL3fZDe0UbKqMet7mn\/\/8Z8+j+eabb8ppp50mp5xyijRt2rQilcNgs1iC5UYo5u9h8fUsUIXHqM+WLVtk8uTJVmFTj4pSD7E+Jn2bsKlLHfQGCvUUz5w50ypshuXdRkdhGmt\/+lHLqBrhsINC2MQO7MFmXtoDYPNYHQpCKLAZrwU3\/XupvmLt2rWyZs2a0ISBzXh6N7rQcWDTeDwnTJjg3ckafIwx6gG7Xbp0KfhzXM+mpjVq1CirsKkVoX\/\/\/l6+TPo2YVM3ael5b9u3b\/eWPtj0bIbl3SZs+tPPkpFjB4WwiR3Yg828tAfAZjTYDJt+V+eEOWpQUzGAqoNgffx\/D06jm0246oQ455xzZNq0aTJ48GDvGEN\/3BUrVtRfl+lfqlYuP2YTb9++fRuk2a5dO+vNuOnfS\/UVYZ5NvThg0aJFeDatl4jFBHU95r333iuvvvpq2VTVy\/nRj37U83jG8XamAZs2RjCs2SyEDL+mRhubsGmjzMoaaYIA2AF2YAaHw4YNszr4DKtTNgefWT7U3WYbkqBaW4sSdxr9nnvuqV8GFvRsFrsDXZde6XIyE978Wz\/C33\/efvvt3ndpWP29amxA1Thq+vXr571fnxkzZngbdY0dmjB6N7u+o1x+FDZ1Bk1hVp0\/ad8kVG5ms1ihJo1nzUhqnJAT0+ivvPKKd6i7FpYe3K7PBz\/4Qe+\/zz33XKiEl19+uVx\/\/fXSunXrSBKHwWYx49Dflzp706ZRARlABpBxvAonOcw7UgPwbqCsH\/xNe5C8PcirZzNsg5DfExllGt0PfGqDwXOu\/XXMhDXtlt8jGgRJDRM8XjBufoLXVoalGacNKBc2af+eNF65\/LjydydgU8XU446uvfZaufrqqz0XvF5TqY\/eHqRXVy5evFhuu+027\/ijhx56SL75zW96m1Oiro8Mg00zQtqxY0eDkZl\/01CwsG0aFZ1L8s4lTiUEMhrvndjYQbgCXFd5\/MxEG7MjcezMdtg4ns3gJlft50aPHu15Is3GUQNs6jE0j5naLnf2dKmzqxU2dRp97ty50rNnTy\/pqLBpPJhR8lMuj5Xon7R\/TxqvkrxmKa4TsGmgT488Gj9+vDRp0qRAQ900tGTJEnnhhRe8NRsnnniidx+vFq6uH4zylJpGD96\/atz1xdK1aVTAJrBpPARMn4rg2Tw2KLB5OgXT6MMkT7Bp2hPjndR\/+2HT9HfF1mGWAzkzda7T4DoDOHTo0Pq9DUlgM0l+yuUxChPY7t9tckEl+a9VXCdg04zEdLPJpz\/96VCtfvrTn4oeuWNc9r\/4xS88b+f9999fdW1tGhWwCWwCm8erMLAJbKo1JLGDvE6jB48+KgWbvXr1auDlDK6BjDqNrp5ShT7th\/XiFPVkxoXNKPkJA0tgs+rYU\/aFTsCmbhDS43\/Us6n\/bdasWcGH6VS6HkPwpz\/9yfuvrtNct26d\/OhHPxLdMVbtB9hc5Z2zqbvRdSe9zd3obBCytws5Lx6tOPWf5RSNdzkFsHmsJpQ6+sjAnd8baTyLZhrdOH\/8YfwAqrvK9VHYNO\/Sf+uGID17Oc40epT8AJtxWrjahXUCNlUevU7y61\/\/uvzzP\/+zXHHFFd7ZmvrohiHdWaejJ51C141BTz31lLehSI9C0DjVfoBNYNO2zeHhxsONh7syDxORBJ4AACAASURBVHdeYTNsg5B\/KVhwQ47ZxKPnJOujkKl7IdSBY2YO4xx95L8c5bLLLosFm7qGtFx+dM\/Gxo0bCy5ZwbNpuweqPD1nYFO9l3pIqi4sfuuttwq+vHnz5p4B64HpetC7HqWgIyj1cppd65VLFT0FYBPYjG4t0UICm8AmsJkebJ5zeKh0eqeyW8ii1eR0Qh1oslceOvH23J7hmI6q4akm7d+Txqvmt6X5Lmdg04igxyDp4ah\/\/OMfvV+pm\/0jH\/mIdOjQwfu3wqZ6O9u3by8tWrRIU7uiads0KiADyAAyKoOMOI0A0+j5mkZXp4ReIah9iuuPXo2sF4nohRo86SmQtH9PGi+9L6luys7BZnXlSfY2m0YFbAKbwCawaRSgPUjeHhQbSChw6o\/rj0ImoJl+KSbt35PGS\/+LqvMG52BTNwsdPny4gTo6zb5161ZvnWbUg9zTktimUdG5JO9c4pQvHq3G69HCDsIV4JzNOJZBWBQ4pkDS\/j1pvMaiuzOwuXfvXvna174meqRRsSd4f2utCsmmUQGbwCaeTTybeDYLW\/NKjj6aNGmS6JQzDwokUWDXrl3e0ou4Vxvb5IIk+a51HCdgU72WevyCHmM0cuRIUajUw9rPPfdc7+wu3amum4R0h13nzp0bHPpebZFtGhWwCWwCm8AmsFk5bJpNNDqLwYMClSiQZH2sTS6oJO+1iusEbL722mvezUEKl\/pffRQsTzjhBO8srwMHDsh1110nF110Uf1NBbUStBI3e1iegU1gE9gENoHNymFTU1DgPNh0b9nuoa7pk\/LbFuu8DTddunQpG77aAXRD06JFi+Qv54ySd97TvqLXN9v\/lLR86oGCbzXpTz11n3RscSRx+lsOtJS1dSd7aetxhX\/+85\/lmWeekTvuuEO+MeyodG6XOGkvonq4l\/+kSWjeL3hzorQ5+r6KXlDMDpKsjwU29a7HjD\/mTK8JEybIhRde6OU2eI6WngH2q1\/9quCsrVp9lk2jAjaBTWAT2AQ27cBm1D6BNdx21nArbF63tZM35az7Kerq6nJ72YdNLohqx1kK55Rn85JLLqn3XGrBLViwwJtOP\/nkk721nHqwuzl0tpYi2zQqYBPYBDaBTWAT2PQr4Eq\/AGweLzWbXFBLvkn6bidg06zZfOGFF7zpc4VLvVVAF3rrv8866yxZsmSJPPTQQ\/XwmVQQG\/FsGpUrjYrqZhqWVas41N2GHbnYuWAHtku+MD3aAwafLg0+gU1g0yjgBGxqZvVYo3HjxnmHty9dulRatmzp3RSk11TpAe66DkSv1VIADd6dnm7z3zB1YBPYtG1zQAaQ4RJkMOiw3QK4OegANoFN52BTM6wH7\/7ud7\/zNgLp7UB79uzxdqnrFPrnP\/950TWd6vWs9QNsApu2bRDYBDaBzeO1KsnRR3HqJGs2WbOp9mLTDmxyQRxbzkpYJzybuodJp8hPO+00J25IsGlUQAaQAWQAGUYB2gPaA5faAzybeDad8mzqfehjxoyRL3zhC5k42qjcSAHYxLNZzkbi\/h3IADJcggym0ePW8HjhXWkPgE1g0ynYNOds+nejx6ua1Q0NbAKbti3Olc4FyLBd8oXpYQcMOlwadACbwKZTsKmZ1UNmb7nlFrniiivkYx\/7mLRq1apBq96kSRM56aSTpGnTpum2+GVSBzaBTdsGCGQAGS5BBoMO2y2Am4MOYBPYdAo2zaHuW7ZsKVmDuRu9dAPXbP\/T0vqRuQV3uhqIWdCrTvq0OVRRC8nRRxXJVzIysAlsApvHqwgbhI5t4Dl4\/jQ50v70ihqeNPsFYBPYdAo233zzTXniiSdE\/1vqOfHEE70zN\/W\/tXzwbOLZtG1\/wCawCWwCm0YBV9oDYBPYdAo2bXfcaacHbAKbtm3Mlc6F6VPbJe\/m9Cl2gB34bYDrKkVsckG61pVO6k4cfeT\/9IMHD3q3B+mtQurFPOGEE2T\/\/v3yvve9r+ZrNYOjTq1gAwcOrKjkgAw8Wni08Gi55tECNitq9stGdqVfwLOJZ9M5z6bC5fr16+Wmm26St956S8z6TL0taPz48dK5c2e5\/vrrpXXr1mUratoBbI5gXGlU6FzStSrsgEEHgw4GHa4NOoBNYNM52NRbgvQqyq9+9avSqVMn7w70lStXStu2beX++++XWbNmyXXXXSfDhw9Pt9ePkDqwyTR6BDOJFQTYBDaBTWAT2CxsNl3aKGaTC2J1HhkJ7MQ0+htvvOHdg65T5dOnT5fNmzd711QqbLZr1070hqElS5Z4V1kuXLiw5t5Nm0YFZAAZQAaQ4RpkMNORbg\/vSr+AZxPPplOeTXP0kd59fuGFF3oLbf2wqR+jns\/FixfXA2i6Vb106sAmnk3b9udK5wJk2C75wvSwAwafLg0+gU1g0ynYNNdVfv7zn5ehQ4eGwuaqVavkl7\/8pefh1Kn1Wj7AJrBp2\/6ADCDDJchg0GG7BXBz0AFsAptOwaZuDlJPpk6T33777fLss88WeDb\/+Mc\/epuE1Oup0+y6aaiWD7AJbNq2P2AT2AQ2WU7h2nIKYBPYdAo2NbN79+6ViRMnyquvvio9evSQ3\/zmN\/K5z31OnnnmGXn44Yfl1FNPlWXLlskHP\/hB2\/187PSATWAzttGUiQBsApvAJrAJbBY2lGwQst3TpJeeExuEzOfv27dPli5dKvfdd58cOHDA+3Xz5s3l0ksvFV3P2aVLl\/SUipEysAlsxjCXSEGBTWAT2AQ2gU1gM1KHkcFATsGm0U93nytsvv3223LSSSdl5jD3YEPAoe7bZdSoUbJs\/FHp16My63dpBFvZl4bHBjaBTWAT2AQ2gc00+pdqpOkEbOoGIT1L85Of\/KR07949c3AZLCg8m3g2bVdeYBPYBDaBTWAT2LTdt1QrPWdgc8yYMfLYY495azKHDBkigwcPlve\/\/\/3SpEmTamkV+T3AJrAZ2VgiBgQ2gU1gE9gENoHNiF1G5oI5AZuqmu5I193oemXlj3\/8Y28a\/dxzz\/XAUz2eJ598ckXiarobN26UOXPmSKtWrQrSMh29+eXs2bO99xZ7gE1gsyJjDIkMbAKbwCawCWwCm7b7lmql5wxs+gUJgqfeMPSxj31MvvzlL3v\/jXv0kenIBw0a1AA29W9TpkyR1atXS8+ePWXbtm0ycuRIb0NSMeAENoFN2xUY2AQ2gU1gE9gENm33LdVKz0nY9Ivz+uuvy3\/8x394xx7p1ZXmCsuoAur5nRpXnyBsmpuLBg4c6J3faR71gq5bt67ou4BNYDOq\/UUNB2wCm8AmsAlsAptR+4yshXMSNtWTqfejK2Ru2LDB0\/Siiy6SYcOGiYJhVM+mgqbGV6\/l97\/\/fdm9e3eBZ9N4MefPn++la55ivw82BOxGZze6rQoPbAKbwCawCWwCm7b6lGqn4wxsBgHzrbfe8tZsDh8+XC644AJp06ZNRdopeAZhMziFHoTNYlPpeDbxbFZkjCGRgU1gE9gENoFNYNN231Kt9JyATTOdvWXLFjn99NNlxIgRniezQ4cO1nSKA5smP3pPe9i6TQMGkyZNkgEDBhTksWPHjqI\/UR\/14CpQHzx\/mhxpf3rUaKHhmu1\/Wlo\/Mlf0HnnjqTXpL+hVJ33aHKoofXM1mR68r0dU7dy5U\/QUAZvnbIbl\/ZJDs6TT22dWlPe6E56UB1teX6BNRQlajowdFMImdkB7MHZJk9C2jPYgXuOTZr9g+gStr3379pU9e\/bkol\/Q79Qf\/7Nr1y6ZOnWq2JjxjFfC2QjtBGzqzvMf\/ehHcv7550vnzp2LHnekG4eaNm2a6BzONGAzrIgVlBUeoz4K2JMnT7YKmwsXLpQ+ffp4WTDp24RN3a2v5aSe4pkzZ1qFzbC82+xc\/OlHLaNqhMMOCmETO7AHm7QHDWuwGXzSHlTmhDCwqTr27t1b9BbAPPQLa9eulTVr1oR2DcBmNXrMFN6htwm98MILcs8998jjjz8u3\/nOd7yNQnGfOLBZbke68WzOmzevwRWacT2bmpbewmPTs6kVoX\/\/\/p5EJn2bsLl8+XLp2rWrbN++XcaNG2cVNsPybhM2\/enHtaE0w2MHhbCJHdiDTdqD4rBJe2AHNlXHM844Q1588cVc9Athns1NmzbJokWL8Gym2VGmkbau4Xz44Ye9HeGPPvqo9wr1fGphJjlzMww22SAUr+T8UyY6ja6wyXWV8TQsFpo1m4Ww6fcOGG1sDjqy6n3ADrAD4yTQDbE2nRBhdapSJ4TpEzRtnUavq6vLbb9gcy+HnV6luqk4MY3ul0Td8Pfdd58HmS+99JK38\/wTn\/iEXHPNNfLhD3840RS6ph8Gm8XWZnL0UbiRApvpVV4gA8hwCTI0r7QHtAfA5nEbADZ1Hjrjj67F1A0Susj4l7\/8pXebkG4O+vOf\/yx33nmnd5B7pU8YbGqaCpa67tCM+spNofs7BBveESADyAAyjtfux\/4kohtD8Gzam0ZPw6MFbFbaI5WO70q\/AGwCm0aBTHs21Yt5\/\/33ewtt1f1+yimnyBe\/+EXvp3Xr1t5OZz1s3X8GZtIqXgw2\/R29SZvrKvFsJrWzpPFc6VyAjKQlHC0edsDg06XBJ7AJbGYeNs0Uth6fM3jwYPnsZz\/rTZObA9vN323BZrSmPloom+5yOhc6F5c6F2AzWhuRNBTtAe2BS+0BsAlsOgObzz77rOe5vPTSS+Vv\/\/Zv6w9vBzbjd1fmPDWmzRpqZ446sbH0IX7JlI8BZAAZLkEGg47ydbqSEK60B8AmsJl52NQMHjx4UDZu3Fi\/41y9mro+80tf+pJ3uPv48eOtTaNXUvGDcfFscoOQTXsCMo6ryZrNX3vX8mZ9FzKwabsFKEwP2HRv8GmTC9K1rnRSz\/SaTfPJuodJD4J94IEH5Lvf\/a63ftM8CxYskEGDBkW+Dz0dGcMbAhteMlcaFTqXdC0LO3Cvc0nDIrAD7MClwSeeTTybTng2wxprszNdD4n9+c9\/LnpHeqdOnbwrLHWq3eYVlkk7C5sjGDoXOheXOhcGHUlbjWjxaA9oD1xqD4BNYNNZ2PQ3yXqN5U9+8hPvSKSnn37au4JRz99McoNQtKY+Wihgk2n0aJYSPRSQAWS4BBkMOqLX7SQhXWkPgE1gs1HApvmId955x7uV4L\/\/+78972aSG4SSVPhicYBNYNOmPQEZx9VkzSZrNtUasAM37ADYBDYbFWza7tgrTQ\/YBDYrtaFgfFc8GXi0bJd8YXrYAR5ulwafwCawCWym2CcAm8CmbfMCMoAMlyCDQYftFsDNQQewCWwCmym2BcAmsGnbvIBNYBPYZDmFUcCV9gDYBDaBTds04EsP2AQ2bZuXK50LHi3bJe+mRws7wA78NqDHAPbt29c7tlD3V4waNUqWjT8q\/XpUppNLa3dtckFlqtUmthPnbNZGmuRvtWlUQAYeLTxaeLRc82gBm8n7jygxXekX8Gzi2cSzGaVGJwwDbOLZTGg6RaO50rkAGbZLHs9mmKIuebTSsAhX2gNgE9gENtNoAd5NE9gENm2blyudC7Bpu+SBTWCzoQKutAfAJrAJbKbYJwCbwKZt83KlcwE2bZc8sAlsApvFapVLHm6bXJBuK5NO6qzZTEFXm0YFZBwrIJcalRRMSrAD7EAVwA6wA5fsAM8mnk08m2kQAdPoct3WTt71od27d8\/trsM0TAvIADJcggw83Gm0Ag3B5eD50+RI+9Mrelmz\/U9L60fmiu4YHzhwoJeWaW8W9KqTPm0OJU4f2AQ2gc3E1ad8RDybwGZ5K4kXAtgENoHN43WGmQ6uq3RtxssmF8TrPbIRmmn0FMrBplEBGUAGkAFkGAVoD2gPXGoP8Gzi2cSzmQJkBjsE\/7RE0tfRudC5uNS5aF5NB8NyiqS1vng82gPaA5faA2AT2AQ27fcD9Sni2WQa3bZ5ARlAhkuQwaDDdgtQmJ4r7QGwCWwCmym2BcAmsGnbvFzpXIAM2yXvJmRgB9iB3wa4rvL4pisbM57pWlc6qbNmMwVdgU1g07ZZAZt4NvFsHq9VbBBig5Bag0t2YJMLbPcv1UgP2ExBZZtGBWQAGUAGkGEUoD2gPXCpPWAanWl0ptFTgMxgh2DDXU7nQufiUufC9GmKDQuHuteL65JHKw2LcKVfADaBTWAzjRbg3TTxbDKNbtu8XOlcgE3bJV+YHnbA4NOlwSewCWwCmyn2CcAmsGnbvIAMIMMlyGDQYbsFcHPQAWwCm8Bmim0BsAls2jYvYBPYBDaP1yqm0dkgpNbgkh3Y5ALb\/Us10mODUAoq2zQqIAPIADKADKMA7QHtgUvtAZ5NPJt4NlOAzGCHwAah7TJq1ChZNv6o9OtRmeAujWAr+9Lw2EAGkOESZDCNnkYr0BBcDp4\/TY60P72ilzXb\/7S0fmSu+Psr094s6FUnfdocSpw+sAlsApuJq0\/5iHg2mUYvbyXxQgCbwCawiYfbNQ83sAlsApvx+vpYoYFNYDOWwUQIDGwCm8AmsAlsFjaWLs142eSCCF1G5oKwZjOFIrFpVEAGkAFkABmuQYbm13i1Vq1i8Gm7m3GlX8CziWcTz6bt2u9LD9ikc7FtXq50LkCG7ZIvTA87YPDp0uAT2AQ2gc0U+wRgE9i0bV5ABpDhEmQw6LDdArg56AA2gU1gM2JbsG3bNhk5cqTs3r27IMagQYNkzpw50qpVqwYpAZvAZkTzihwM2AQ2gU2WU7i2nALYBDaBzYjdvHbyU6ZMkdWrV0vPnj0jxQI2gc1IhhIjELAJbAKbwCawWdhoskEoRidS46BsECpTAOvXr5eNGzcW9WKGRQc2gU3b9RrYBDaBTWAT2AQ2bfct1UoP2Cyj9K233uqFmD59euQyATaBzcjGEjEgsAlsApvAJrAJbEbsMjIXDNgsUSQvv\/yyjB49Wtq2bSsPP\/xwfcixY8eWhE8DBpMmTZIBAwYUvKFjx46iP1GfzZs3y\/Dhw8XmTRF6FMnAgQO9LJj0K70pQtMy63OWLl0q3bt3l507d8qYMWOs3iAUlvdLDs2STm+fGVXS0HB1JzwpD7a8XvzpV5Sg5cjYQSFsYgf2bo6hPWhYWWkPjvU7lfYL\/uOv+vbtK3v27MlFv6DfqT\/+Z9euXTJ16tSCm5osdxOZTg7YLFE8ZnOQbgYynk0DoN26dSu7QSgs6REjRniVOOqzZcsWmTx5slXYXLhwofTp08fLgkm\/0kbFD5uzZ8+Wzp07e5uqZs6caRU2w\/JuEzb96Ucto2qEww4KYRM7sAebtAfFYZP2wM51lapj7969Zd++fbnoF9auXStr1qwJ7RpsXGNdjT7H9juAzQSKlpsmN3+fN2+edOnSpSLPpqal94vb9GxqRejfv7+XL5O+Tdhcvny5dO3aVbZv3y7jxo2zCpthebcJm\/70E5hGalGwg0LYxA7swSbtQXHYpD2wA5uq4xlnnCEvvvhiLvqFMM\/mpk2bZNGiRXg2U+slG2HCxuM5YcIEGTJkSIMvLAejcSRhrV4hZPhHhUYbm7CZ1VEndoAdmMHhsGHDrA4+w+qUzcEnNwjFafGjhXWlPeDoo+PlaZMLollJtkLh2UxQHsBmuGhcT5fAmCJGcaVz8S+nADIiFm6MYNgBgw6XBh3AJrBpFAA2SzT0xUYi5c7etDmCoXOhc3GpcwE2Y5BjgqC0B7QHLrUHwCawCWxGaOjfeOMNmTFjhuzYsUNWrlwp7dq1k7BNQ8GkgE2OPopgXrGCABlAhkuQwaAjVvWOHdiV9gDYBDaBzRjVW8\/aXLZsWX0M3W0dtlbTBAA2gc0Y5hUpqCudC5ARqTgTB8IOGHS4NOgANoFNYDNxc18+IrAJbJa3knghgAwgwyXIYNARr37HDe1KewBsApvAZtzaHSM8sAlsxjCXSEFd6VyAjEjFmTgQdsCgw6VBB7AJbAKbiZv78hGBTWCzvJXECwFkABkuQQaDjnj1O25oV9oDYBPYBDbj1u4Y4YFNYDOGuUQK6krnAmREKs7EgbADBh0uDTqATWAT2Ezc3JePCGwCm+WtJF4IIAPIcAkyGHTEq99xQ7vSHgCbwCawGbd2xwgPbAKbMcwlUlBXOhcgI1JxJg6EHTDocGnQAWwCm8Bm4ua+fERgE9gsbyXxQgAZQIZLkMGgI179jhvalfYA2AQ2gc24tTtGeGAT2IxhLpGCutK5ABmRijNxIOyAQYdLgw5gE9gENhM39+UjApvAZnkriRcCyAAyXIIMBh3x6nfc0K60B8AmsAlsxq3dMcIDm8BmDHOJFNSVzgXIiFSciQNhBww6XBp0AJvAJrCZuLkvHxHYBDbLW0m8EEAGkOESZDDoiFe\/44Z2pT0ANoFNYDNu7Y4RHtgENmOYS6SgrnQuQEak4kwcCDtg0OHSoAPYBDaBzcTNffmIwCawWd5K4oUAMoAMlyCDQUe8+h03tCvtAbAJbAKbcWt3jPDAJrAZw1wiBXWlcwEyIhVn4kDYAYMOlwYdwCawCWwmbu7LRwQ2gc3yVhIvBJABZLgEGQw64tXvuKFdaQ+ATWAT2Ixbu2OEBzaBzRjmEimoK50LkBGpOBMHwg4YdLg06AA2gU1gM3FzXz4isAlslreSeCGADCDDJchg0BGvfscN7Up7AGwCm8Bm3NodIzywCWzGMJdIQV3pXICMSMWZOBB2wKDDpUEHsAlsApuJm\/vyEYFNYLO8lcQLAWQAGS5BBoOOePU7bmhX2gNgE9gENuPW7hjhgU1gM4a5RArqSucCZEQqzsSBsAMGHS4NOoBNYBPYTNzcl48IbAKb5a0kXgggA8hwCTIYdMSr33FDu9IeAJvAJrAZt3bHCA9sApsxzCVSUFc6FyAjUnEmDoQdMOhwadABbAKbwGbi5r58RGAT2CxvJfFCABlAhkuQwaAjXv2OG9qV9gDYBDaBzbi1O0Z4YBPYjGEukYK60rkAGZGKM3Eg7IBBh0uDDmAT2AQ2Ezf35SMCm8BmeSuJFwLIADJcggwGHfHqd9zQrrQHwCawCWzGrd0xwgObwGYMc4kU1JXOBciIVJyJA2EHDDpcGnQAm8AmsJm4uS8fEdgENstbSbwQQAaQ4RJkMOiIV7\/jhnalPQA2gU1gM27tjhEe2AQ2Y5hLpKCudC5ARqTiTBwIO2DQ4dKgA9gENoHNxM19+YjAJrBZ3krihQAygAyXIINBR7z6HTe0K+0BsAlsAptxa3eM8MAmsBnDXCIFdaVzATIiFWfiQNgBgw6XBh3AJrAJbCZu7stHBDaBzfJWEi8EkAFkuAQZDDri1e+4oV1pD4BNYBPYjFu7Y4QHNoHNGOYSKagrnQuQEak4EwfCDhh0uDToADaBTWAzcXNfPiKwCWyWt5J4IYAMIMMlyGDQEa9+xw3tSnsAbAKbwGbc2h0jPLAJbMYwl0hBXelcgIxIxZk4EHbAoMOlQQewCWwCmzGae9PAmyizZ8+WIUOGFE0h77Cp+jz11FPSu3dvmTlzpiwbf1T69YgheEjQx\/4kMnZJE7n77rtl4MCBXgij8yWHZkmnt8+s6AV1JzwpD7a8viD9ihK0HNlFyMAOLBuBz+YPnj9NjrQ\/vaIXNNv\/tLR+ZG5onVrQq076tDlUUfoGNLCDimQMjexKe+CHzY4dO8qaNWty2y\/Y5AL7FpV+ik2OHj16NP3XuPsGNZApU6bI6tWrpWfPnrJt2zYZOXKkTJgwoShw2jQqVxoVv0dr0qRJsmjRIjH\/BTYrt3\/sAI+WSx4t2oPK63ypFFxpD\/ywqd8zbNiw3PYLNrkgXetKJ3Vgs4SuL7\/8sowePdrzpE2fPr0+5Pr162XdunWycuVKadeuXYMUbBqVK40KnUs6FdSkih0Am8Dm8TrGTMevPXDLuocb2Dxusza5IN3eJp3Ugc0Suhov5vz58+unbjV4sd8HwcA\/5Zu0+IAMIAPIADIYdBS2oMAmsKkW4ZIdAJtMoxflwOAUuglYbirdGJVOIw8YMCApZ3rxdu3aJVOnTpVDvQfLkfa9K0qr6V\/2y3t+u8qbxjD5MukP7\/RKxWu09hxuJvOe7yCf\/\/zn5b777qv\/75jPVL5mc\/fLIt+6u0lo3s85PFQ6vVPZms0DTfbKQyfeXpB+RWJbjowdHBMUO6A9wA7c6RdMn6B9TteuXb2+zPQPeesXTBtuwwlluXupSnJ4NkvIXAw2zfT60KFDQ9dt7ty506tUmzZtqkoh8hIUQAEUQAEUQIFsK6BOnnnz5nngnbcH2EwBNjVJBU794UEBFEABFEABFEABhcw8gqaWPLCZADbLTaNTpVAABVAABVAABVAABY4pAGyWsISkG4QwLhRAARRAARRAARRAAWCzrA0UW5tZ7uijsgkTAAVQAAVQAAVQAAVyogCezTIFrWCpt+CYHWRMoeekZvCZKIACKIACKIACVhQANiPIGPe6yghJEgQFUAAFUAAFUAAFcqEAsJmLYuYjUQAFUAAFUAAFUKA2CgCbtdGdt6IACqAACqAACqBALhQANnNRzHwkCqAACqAACqAACtRGAWCzNrrzVhRAARRAARRAARTIhQLAZi6KuTYfWe5az9rkirdWQwFT9lu2bPFe16dPH1m5cqW0a9euGq\/nHTVUQE\/smDZtmsydO1d69uxZkJNbb71Vli1bVv+7sWPHyvTp02uYW16dlgKl7MCc6rJ7927ah7QKIGPpApsZK5DGlB3TscyePTv0DvnG9K18y3EFDGgOHDiwHiTUFvRUB4CzcVuKKft9+\/bJ6tWr62HzjTfekBkzZsiOHTvqbcCE7datm8yZM0datWrVuMXJ0dcVswOVwJzuYvqFMNvIkVS5+VRgMzdFXd0P9R8XBWxWV\/tav03Ppl28eHEBbHA+ba1LJf33++t8586dQ8t\/\/vz5ooMQ82icKVOmFIRNP6e8IU0FStmBAUu1D79Hu9htfWnmk7SrqwCwWV29c\/E2M6q9+uqrZcWKFTJ06FA8m7ko+WMfCWzmqLDf\/VS\/GiGaRAAAC\/lJREFUt6p79+6RAdLEM5dm5E+5xvXF5eyg1NR641KCrwkqAGxiE1YV8I9cFTZHjx4NbFpVOPuJFZtG37BhAx6s7BdfxTmM460MG5hUnAESyIQCYXagv9MlNUuWLPHW9GqboM+gQYNYSpGJUksvE8BmetrmMmX\/vfEqALCZSzPwPtq\/GYTOJD92EBU2zdSp2gabhBqffYTZgbn+Wb\/WeLNZs9n4yj7si4DNfJRzVb4yuO6G3ehVkT1zLwluANAMmk6G6dLMFZf1DEWBTdM26MvZNGa9CDKRYCnYDK7jZ013Joos1UwAm6nKm5\/EwxZ+A5v5KX\/zpcYO9N\/BHcbq6dSjTth53LjtohxsApqNu\/zN15WCzeCgs9jGoXwolY+vBDbzUc6pf2Xw3LTgCzlnMfUiyMQLSnUa\/iUWnLeZieJKJROlYNO0Ex06dMCjmYr62Um02JrNYcOG1U+hBwepwV3q2fkaclKpAsBmpQoSv6gCeDbzZxx4NvNX5sEvLgabBjT79euHdzsHZhJmB2GbB1UKjj5q\/AYBbDb+Mq7ZFwKbNZO+pi9mzWZN5a\/5y0tBBge417x4qpaBYoOO4PrtUgPUqmWWF6WuALCZusT5fQGwmd+yDy6rCB7ynV9lGv+Xl9uFHKYAFz80PrsotZzCf\/C7fjmnVTS+8g9+EbDZ+MuYL0QBFEABFEABFECBmikAbNZMel6MAiiAAiiAAiiAAo1fAWCz8ZcxX4gCKIACKIACKIACNVMA2KyZ9LwYBVAABVAABVAABRq\/AsBm4y9jvhAFUAAFUAAFUAAFaqYAsFkz6XkxCqAACqAACqAACjR+BYDNxl\/GfCEKoAAKoAAKoAAK1EwBYLNm0vNiFEABFEABFEABFGj8CgCbjb+M+UIUQAEUQAEUQAEUqJkCwGbNpOfFKIACKIACKIACKND4FQA2G38Z84UokAkFzBV1l112mdx8883SqlWrgnzV6o7kWr03SaHs27dPbrnlFvnRj34kR48elVWrVsn555\/fICkN973vfU+GDh0qJ598cpJXEQcFUAAFrCkAbFqTkoRQAAVKKWBgs1mzZjJnzhxR6PQ\/tYK+Wr03ibWsW7dO9B7xSZMmSa9evaR3797Svn37BkmtX79eNOzKlSulXbt2SV5FHBRAARSwpgCwaU1KEkIBFIgCmxqmR48esnz5cjn11FPro9QK+mr13iTWcuutt4pCezmIBDaTqEscFECBtBQANtNSlnRRAAUKFDCeza9\/\/euyevVq+fjHPy433nijnHjiiV64MOjTOFOmTPHC9+zZsz694O9ffvllGT16tHzhC1\/wpucXLFggOpX8qU99SvR9J5xwgiio6fSzegKvu+46GTx4sKiX1bz3nXfekYsvvlhuv\/12eeaZZ2TgwIFeuLPOOkuaNGlS\/+7XX39d1q5d601hv\/TSS3LGGWfImDFj5B\/+4R+89PQx+Zs+fboH1Tt27JCpU6fKl7\/85VCr0Hf\/7\/\/+r3znO9+Rxx57zPsGTW\/ChAnSpUsXMd+3ZcuW+viDBg3yPMTB5Qj6ncuWLasPN3bsWNF8HDlyRH784x97Wj755JPev9UzetVVV9VroZEMqH7zm9\/00vmv\/\/ov+cAHPiDjxo0TfacpL53Gf+ihh2TRokVeepoPLVMtr27dumH9KIACKFCvALCJMaAAClRFAQObd999t2zfvl1uuOEGue222zzAswWbr732mpx00kkyfPhwOXjwoPzrv\/6rdO3aVd566y0577zz5Oyzz5Yf\/vCH3s+KFSs8ODKw+dOf\/tSLq+DYuXNn0Xwq3ClwKXjqowCrMKXwqJCmaf\/qV7\/yAO4rX\/mKB4cKnPqtmgdNTyGtQ4cOcvrpp3tT38FHoU3XVyoUf\/azn\/V+6urq5M4775SmTZvK0qVLpXv37vLEE094kPuHP\/zBy4PmUUHXAK5J9+mnn5b77rtPfv7zn3vh\/vqv\/9p7r8bV9Z4jR46Uj370o6Ja6Tcq3PrLQWFTw7Vt21YuuOACueiii7xvXLNmjVxzzTXej77zkUce8f7\/H\/\/xHz0dNT0F6+bNm3uadezYsSp2xUtQAAWyrwCwmf0yIoco0CgU8MPmmWeeKTNnzhQFIzOdbsOzqRC0ePHietBR7+OsWbO8dY6XX36556FUkFMvqEKUei7Ne830tOZNnwMHDnh\/V4+d8SAqsCmMKUwpaJrnF7\/4hQfPCogf+tCHPNgcNmyYB5CjRo0q8IwGC\/P555\/38vP3f\/\/3Mnny5Hp43Llzp6hXUiFZPZP6bUmn0RUEZ8yY4XmHDRBrPvbs2VPwDuPZ1LLRcCasArF+u3pe77rrLu8bNS+aR9Xmve99r\/dZCuf6Dd\/+9rfrAb1RGC8fgQIoUJECwGZF8hEZBVAgqgJ+2FRPoYKmQpaZTtepZAUifQzcxZ1GP\/fccz2INdPeCoY6La6eR+NV1GlwfY\/CokKcgU2dHtZd8maaWPOhm2zUs6iA9b73vc+L95e\/\/MXzWurUvHl27dol8+fP9zyJQ4YMqYdN9Rwar2gxnR588EHRKWvNowFdDauAt2TJEm\/qW9dodurUKTFsFnu3+fa\/+qu\/8mBZgVY1U2APLl3Ytm2b5xVVAL\/00ks9+NTlCro84JJLLmHXe9SKQDgUyKECwGYOC51PRoFaKBCETYWpe++9t346\/ZOf\/GTFsKlgpwBpnjBwMoCl09B+2DT\/9mujeb766qs9sNKpbIVj\/7rJoI46rawwFvzWUnprHnWK2gClP2ww\/0k9myZNXaf54osvynPPPeets\/yf\/\/kfbxpd14cawC+2ucisGzUa65KCb33rW946WH10\/aeu6dS1sLrOlAcFUAAFjALAJraAAihQFQXCAEzXVZrpdN1oomss9Unq2UwDNnUqWb18CqMKm8F3hImXNdhUsNd1qjfddJO3qemUU07xTgT4yEc+4oGxrq+MCpt6rqcCtT6a7u7du+UnP\/mJqIfWbBS64447Qs\/\/rIqh8RIUQIHMKQBsZq5IyBAKNE4FigGYmU7XzTuHDh2S97znPQWwaTyL+nfz3H\/\/\/d4UrpnqDXrdTLg4nk1dm6kw1qJFi\/r36DS6rsPU9+jh6LoeUZ+FCxdK69atixZUHNgsN42uu8E1D7rJKKln06wL1SULX\/3qV6VNmzZe3hX29Zv861JVM908pd+su9DNY6bRdXCg0+ZhzwsvvOCld9pppzXQsnFaNV+FAigQRQFgM4pKhEEBFKhYgWIAZqbTFWL08R\/p8\/jjj3vrI3V3tP7eAJKG1b\/ZhE2dTtapbN01rs\/evXtl4sSJ3jE+Zi2nTqfPmzfPW0up4GYenY7WKXnN54UXXhhrGr3cBqF+\/frJ9ddf70FwUtg02mv+\/TcO6ZIA3X2v3lq\/Z1P19W+q0vW0ujlIIV81UvDVb3377be9qXSzzlU3Iils6vFSQXCv2IBIAAVQwFkFgE1ni46Mo4BbCpTy9pnpdPXy+WHzlVde8W7LefbZZ72jhRRi\/v3f\/13efPNN7+xJnXrXHdY2PJsKm3rcz5e+9CVvd7Wuo1R40g1CZnORAVAFXYW0\/v37y29\/+1vvCCE9JkjBSz2ecTybUY4+Mu+PCpuqo25m0iUACsUKg5pfPQJKAVpvFdI86pFLhw8flk984hMNYFOPMNINQTrVrscoff\/73\/cAUs8y1Q1Y+m\/dbf+Zz3zG20mvj75Xz970HxfllpWSWxRAgTQUADbTUJU0UQAFGihQDsB0vZ9ZE+k\/rFx3euu\/9RxMhUE9Ski9c3og\/Ny5c63Bpu72\/vCHP+xNket0sAKYeit1Stj\/KADrkUo6xa7rH3UzjB5zpB5YcwRQuW8NihN2qLueX6lHH+laUfNEhU3dvKNg+J\/\/+Z8evKsXcuvWrZ5XVvNmDmDXDU0\/+9nPvB9zNqZZeqAgec8993jh1durHksFVz37Ux\/Ns6avU+5advoUOwif6oACKJBvBYDNfJc\/X48CKIACBQoUO\/oImVAABVAgqQLAZlLliIcCKIACjVABYLMRFiqfhAI1VgDYrHEB8HoUQAEUyJICwGaWSoO8oEDjUADYbBzlyFegAAqggBUFgE0rMpIICqCAT4H\/B8OkYtIB5o+DAAAAAElFTkSuQmCC","height":0,"width":0}}
%---
%[output:44f4e8c7]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAApsAAAGTCAYAAABnKdmUAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ34FlWZ\/29eUjCwQHB5U7AEaTUhMSEvNdOtrTVwU1eIbSFAIQo15V3RSkmQt+LFCA0I9hLBrWwltc3tZdW18FJbWvafipYo8EMgLCFFRflf9+D5Mc8wz\/PMzHPOPDPPfOa6fpfy+83cc+Z77jn3Z+7z1uLgwYMHhQMFUAAFUAAFUAAFUAAFHCjQAth0oComUQAFUAAFUAAFUAAFPAWATRwBBVAABVAABVAABVDAmQLApjNpMYwCKIACKIACKIACKABs4gMogAIogAIogAIogALOFAA2nUmLYRRAARRAARRAARRAAWATH0ABFEABFEABFEABFHCmALDpTFoMowAKoAAKoAAKoAAKAJv4AAqgAAqgAAqgAAqggDMFgE1n0mIYBRpTgd\/85jcyfPjwkofr16+fLF++XDp27HjEQ+\/Zs0fGjBkjGzduLPnbmjVrZNCgQc2\/e\/3112XatGmyfv16GTx4sMyePdv7m\/md\/n+3bt1k5cqV0rt37+brNm\/eLKNGjZLt27d7v6tUlmDhwp7F3Ltt27aNWYEhT+XXfty4cTJ16tTCPDsPigIo4F4BYNO9xtwBBRpGgdtuu02WLVsW+jxhIBgGc\/6LZ82aJUOHDvV+FQU29Tz\/NfrvdevWyfTp05vNRoFN\/73CHibsWRqmEoHNRq5Kng0FMqkAsJnJaqFQKJA9BYIZxLAS+rOC1YBOr\/dDXVTYDGbeggAcBTYrQbN5rih2sldLlAgFUAAFsqcAsJm9OqFEKJBJBfxZymAXuIE3Pzz6u8+DgOi3ZTKV1WBT4W\/Xrl3SuXPn5i57cw\/9\/fnnny9armqQ6IfmYAYz2OVfqavfVFLw2fw29Nn0MJlXUzb9nX9oQTmAVni\/+OKL5Yorrmj2iWBmV\/8QNlQhqIPJAOvvr7zySpkwYYJnU+999dVXNw9X8Jclil2\/swYhPmxIgql71f7222+XFStWeEMn9KhWd5l8MSgUCqBAVQWAzaoScQIKoIAq4AfEKOMa\/aASpVu6GmzqONG9e\/d6YGIg0JRJIeXCCy+UBQsWVAUWA13lymRgdN68eSVjSitldv2QVG6Mqj9jqnBsxpia3\/shslrmtRoQ+u9lxtIGhxuYcwygm7Gxxnal5whCYaVzgzpXG1oBcNLeoEDjKQBsNl6d8kQo4ESBct3ilUCyHOCEZeeqwaZCkB46ZtRcb+zr33r16uVlEKvBigG5KMBshPSXzW\/fD6BhkObXxg+Q5t5q12Q4\/QDpPzfs9367RgP\/84Rlof11Ecykhk0Q8mcgzaQs\/\/OGwXG5zHa5svl\/73\/mYEbZiUNjFAVQIDUFgM3UpOZGKJB\/Bapl7cIgoVKWzg89UWDz4x\/\/uDcTXiHla1\/7mnzjG9\/wMp3+7moXsOmHrOAz+runNYuoRxhAlhuGEAa\/5neVMojVgN2fudRZ\/5UyupVgU+1EXW0gWKYwYC2nQyWN8\/\/m8AQoUGwFgM1i1z9PjwKJFHCR5YwCm5dccom3zJEe119\/vdx6663e\/2vm7amnnnKW2QyDJiNc8G\/HHXdcM2z64aucjUqwGcy+hkFhtYlYBo6DUOxfpiqO3SBQVoLEsL+V0wHYTPQqchEK5EIBYDMX1UQhUSDbCoRN+KlUYn+GNGxNzbDfBSeynHvuufLII480r8l53333RYLNJGM2swybYd3zuqapWQs1KWya+qu2Tiqwme13k9KhQBYUADazUAuUAQUyrkC5MYum2GFjF\/1jBCvN6jZdtLqIeqVF3U2Xe7Bb3vy+UubOL2+c2egmi5e0G73WzGa1bvQhQ4Y0axZ2L33uWmHTr11Y9tk\/7jRON3pwrC+ZzYw3AhQPBWpQANisQTwuRYEiKVBusk9QAwM3Udbl1GsNLEbpRtedbYKzmaPAVLCM1WZ76\/l+0Es6QahW2PTro\/8fXGKqR48eVXddiqJPWDd6uQxw2HjSaktfhU0QAjaL1HrwrEVXANgsugfw\/CgQUYFqYwPVTHCMYTVAjbOoe7XZ3lEzm1rOKM8SzMZWgmf\/cwfX2TQ7JCUZs1muavwTq6ppnBQ2q00G84N0nGWSGLMZ8YXjNBRoIAWAzQaqTB4FBdJQoNw6ieWWqykHIpWW3yk3ZlMzm2EZUO2CjwObRqe4e6OHQWq1Rd1rgU3VYfTo0fKVr3yleV3OsFno\/kytZmSXLFkic+bM8WbqRxlmUGlv9LAscLm6jruou3+fe7rR03h7uQcK1EcBYLM+unNXFEABFCirQJK1QJETBVAABbKqALAZo2b0y3vKlClexqB3794xruRUFEABFIiuALAZXSvORAEUyL4CwGbEOvLvwezv+ol4OaehAAqgQGQFgM3IUnEiCqBADhQANiNUkn9cV5Q9niOY5BQUQAEUKKsAsIlzoAAKNJICwGaV2jSgqYPye\/bsKZMmTfJ2K6EbvZFeA54FBVAABVAABVDAlQLAZgxlFTyBzRiCcSoKoAAKoAAKoEDhFQA2Y7hAHNjcunWr6A8HCqAACqAACqAACugmDPpTxAPYjFHrUWFTIXPy5MmyYcOGGNY5FQVQAAVQAAVQoFEVGDhwoMydO7eQwAlsxvDqqLBpxnmqU3Xv3j3GHRrjVIXshQsXei9VEZ+\/MWqx9qfAD2rXsBEs4AeNUIu1P0PR\/cA8f7kNEWpXONsWgM0Y9RMXNovqVAa2i\/r8MVyqoU\/FDxq6eiM\/HH4QWaqGPrHoflD05wc2Y7zewGY0sYr+UkVTqfHPwg8av46jPCF+EEWlxj+n6H5Q9OcHNmO848BmNLGK\/lJFU6nxz8IPGr+OozwhfhBFpcY\/p+h+UPTnBzZjvOPAZjSxXnjhBVm1apWMHDlSevXqFe0izmo4BfCDhqvSRA+EHySSreEuKrofAJsHDx5sOK+u8wMV3an2798vTU1N0rVrV2nTpk2da4Pb10sB\/KBeymfrvvhBtuqjXqUpuh8UnQvIbDp484ruVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpVLk\/hBLqvNeqHxA+uS5tJg0f2g6FwAbDp4bYvuVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpVLk\/hBLqvNeqHxA+uS5tJg0f2g6FwAbDp4bYvuVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpVLk\/hBLqvNeqHxA+uS5tJg0f2g6FwAbDp4bYvuVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpVLk\/hBLqvNeqHxA+uS5tJg0f2g6FwAbDp4bYvuVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpVLk\/hBLqvNeqHxA+uS5tJg0f2g6FwAbDp4bYvuVEVvVBy4VC5N4ge5rDbrhcYPrEuaS4NF94OicwGw6eC1LbpTFb1RceBSuTSJH+Sy2qwXGj+wLmkuDRbdD4rOBcCmg9e26E5V9EbFgUvl0iR+kMtqs15o\/MC6pLk0WHQ\/KDoXAJsOXtuiO1XRGxUHLpWKya1bt4r+2Dh69OghnTp1kqamJunatavs3r3bqm217z9slz1o34YmebFhW0v8IC81X1pOW36g75L++ONCEduDonMBsOmgHbDtVLZeen1U8+K7DNQEFwdOJeLBmk0Y9AOV2p08ebJs2LDBSuEHDhwot9xyi7Rq1UrefvttufHGG63anjt3rufLergou9++FUEsGsEPDompPoYf2Ps4dNUemHoycaGo7YFtLrDYpKRiCth0ILNNp3IdSNX+tV+8XJ78ww4rSriEjI+e3kPmL1lTSMhw7QfGZyf32iVdjjpQky9s3NtGVjd1kBUrVkjPnj1ly5YtMnr0aPna8IPSrWNNpuXJ50Tu+GkLWbNmjQwaNMgzZsp+3htXS\/uDx9d0g6aWm+Spo9aW2K\/JoOWL8YNDguIH9j8O\/eBuqz0wbYG+r\/379\/d6OoraHtjkAsvNSirmgE0HMtt0KmPrtTNGyzvHdKqptK13Py1tnr4vNFADGdmGjLT8YH6fJunXfn9NfqYBZuKzXY+AzWUTDsqAk2sy7UHGuCXhsHnR\/pnS9e3TarpBU6tNcn+bGZmFTfzgMGziB8PFdVyotT0wbUEYbBatPbDJBTU1cnW6GNh0ILxNpzK29p0zRQ50OqWm0rbe\/Yy0e3ROKGzW2qhowYCMmqqn4sX4AZDhz+LSHvDRMXz4cMm6HwCbh5t1m1zgLtK4swxsOtDWplMBGUAGkHH4JSWz+RvJA2Tw8ekgsPhM5iUuAJvAplEA2HTQJgCbpWP1itZd4sClmsclZj2TAWS4qP0jAxZ+QGYzDx8dwCawCWw6jAnAJrBp273ykskANm3XfKk9\/ICejjz1dACbwCaw6TAmAJvApm33AjKAjDxBBh8dtluAfH50AJvAJrDpsC0ANoFN2+4FbAKbwObht4qxu\/kYuwtsApvApm0a8NkDNoFN2+4FbAKbwCawaRTIS3sAbAKbwKZtGgA2WfooBZ9iYggTQ\/IwMYRudIeNgW8jg6y3B8AmsAlsOmwLyGyS2bTtXnnJZAAZtms+n2P18AP8wO8DLOp+eKcz\/+5nbr0kW9ZZ+shBfQCbwKZttwI26UanG51udLrRS1vWPI3dtckFtuNLGvaATQcq23QqIAPIADKAjLxBBplNB4ElZKgW3ej5GVZjkwvcepcb68CmA11tOhWwCWwCm8AmsJnfjJaDEJObTR4Ys3m49m1ygQufcm0T2HSgsE2nAjaBTWAT2AQ2gU2\/AnmJC8AmsGkUADaBTW+v5fl9mqRf+\/01qWEalhUrGLNZk5AhF+cluNB9arvmS+3hB3x85unjE9gENoFNhzGBzCawadu9gAwgI0+QwUeH7RYgnx8dwCawCWw6bAuATWDTtnsBm8AmsMlwirwNpwA2gU1g0zYN+OwBm8CmbfcCNoFNYBPYBDZLW1aWPrIdadzZY8ymA22BTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte4WCTROwjbizZs2SoUOHVtV63bp1Mn369Obzql0HbAKbVZ0q5gnAJrAJbAKbwCawGTN0ZOb0wsCmButJkybJypUrpXfv3rJ582YZNWqUXHXVVRWBU0Fz8eLFsa4DNoFN2284sAlsApvAJrAJbNqOLWnZKwRs7tmzR8aMGSODBg2SqVOnNmurILl27VpZvny5dOzY8QjNy1132223iQb\/ctcBm8Cm7RcY2AQ2gU1gE9gENm3HlrTsFQI2TRZz3rx5HnCao9zvzd+BzXhuuHFvG5n4bFdZsQLYjKdc9bOBTWAT2AQ2gU1gs3q0yOYZhYDNYBd6EDYrdaWX60YfPHhwSZbUX71kNoFN2687sAlsApvAJrAJbNqOLWnZKzRsmszlsGHDKo7bNBnQ7du3e\/WyZs2akgxpsLIMGFxzzTUycODAkj936dJF9Cfq8cQTT8iIESNk3zlT5ECnU6JeFnpe693PSLtH53iZR5PhNfbn92mSfu3312TfZDaXLl0qPXv2lK1bt8rYsWNl2YSDMuDkmkzLk8+JjFvSIrTsF+2fKV3fPq2mGzS12iT3t5lRYr8mg5Yvxg9KYTPMh\/GDeE5He1BeL9qDQ3Gn1rjg7+3q37+\/7NixoxBxQZ9Tf\/zHtm3bZPLkyVX5Id5bnJ+zgc0xY6QcbL7++usybdo0efHFF5vHZxpAPfHEE2X27NnStm3bI2o7OOvdf8LIkSO9lzjqsXHjRrnuuuuswuaCBQukX79+XhGM\/VobFc\/Wu93oOlu\/W7duonCus\/htwmZY2W1Cht9+1DpK4zz8oBQ28QN7H5+0B0e+wQY2aQ9qS0KYmKA69u3bV3bt2lWIuLB69WpZtWpVaGiolqxKI57U4x6Fhs1qM9LLdYdXG+tprps7d6507969psym2ho9erRV2NQX4cwzz\/TKZezbhM077rhDevToIVu2bJHx48dbhc2wstuETb\/9eryQ5e6JH5TCJn5gDzZpD8rDJu2BHdhUHU899VR5+eWXCxEXwjKbGzZskIULF5LZzFJgtV2WpBOEyo31rNb9zphNxmza9mHGbJbCpj87YLSx+dGR1ewDfoAfmCTB8OHDrSYhwt6pWpMQJrOptrUbvampyYNNTaDY7PHKQ3tgkwtsx5c07BUis1kODqstfVRrZtNGwCK4EFzyFFz8wylYlcB+E057QHuQp\/YA2DzcBgCbBw8etN8kZs+i2QXIAGC1LnR9glrHbAKbxfyCdeH9QAaQkSfI4KPDRStwJLjYnDhKZrP8cAqbsdyGLbfe5cZ6ITKbRrpq21UGlzky1+ki7suWLWuugXHjxpVd9sgfEGw4FZABZAAZhxs\/sypBHrrNXDTZtAe0B3lqD8hsktk0ChQKNl00\/mE2babLCS4ElzwFFzJablsZ2gPagzy1B8AmsAlsOowJwCYThGy7F5ABZOQJMvjosN0ClNrLS3sAbAKbwKbDtgDYBDZtu1degguQYbvm8wkZ+AF+4PcBZqMfWmZQVxCwMbzOrXe5sU43ugNdbToVkEFGi4zW4ZeUMZuHAlbWJ4YAmw4Ci89kXuICmU0ym2Q2HbYFwCaZTdvulZfgAmTYrnkym2GK8tGRj48OYBPYBDYdxgRgE9i07V7AJhluMtxkuI0CeWkPgE1gE9i0TQMhXRw2xmbkpVEho+XQod7dVpTuUxEyWvnIaNEe0B74fYAxm4zZZMymgzaBzCaZTdtuxUcHmU0ym2Q2yWyWtqx5+vi0yQW240sa9oBNByrbdCogA8gAMoCMvEEGmU0HgSWk9yzrE8XoRqcbnW50h20BsElm07Z78dHBRwcfHXx05O2jA9gENoFN2zQQ8tXJmE32RrflZsAmsAlsApvAJt3otmJK2nboRnegOJlNMpu23QrYBDaBTWAT2AQ2bceWtOwBmw6UBjaBTdtuBWwCm8AmsAlsApu2Y0ta9oBNB0oDm8CmbbcCNoFNYBPYBDaBTduxJS17wKYDpYFNYNO2WwGbwCawCWwCm8Cm7diSlj1g04HSwCawadutgE1gE9gENoFNYNN2bEnLHrDpQGlgE9i07VbAJrAJbAKbwCawaTu2pGUP2HSgNLAJbNp2K2AT2AQ2gU1gE9i0HVvSsgdsOlAa2AQ2bbsVsAlsApvAJrAJbNqOLWnZAzYdKA1sApu23QrYBDaBTWAT2AQ2bceWtOwBmw6UBjaBTdtuBWwCm8AmsAlsApu2Y0ta9oBNB0oDm8CmbbcCNoFNYBPYBDaBTduxJS17wKYDpYFNYNO2WwGbwCawCWwCm8Cm7diSlj1g04HSwCawadutgE1gE9gENoFNYNN2bEnLHrDpQGlgE9i07VbAJrAJbAKbwCawaTu2pGUP2HSgNLAJbNp2K2AT2AQ2gU1gE9i0HVvSsgdsOlAa2AQ2bbsVsAlsApvAJrAJbNqOLWnZAzYdKA1sApu23QrYBDaBTWAT2AQ2bceWtOwBmw6UBjaBTdtuBWwCm8AmsAlsApu2Y0ta9oBNB0oDm8CmbbcCNoFNYBPYBDaBTduxJS17wKYDpYFNYNO2WwGbwCawCWwCm8Cm7diSlj1g04HSwCawadutgE1gE9gENoFNYNN2bEnLHrDpQGlgE9i07VbAJrAJbAKbwCawaTu2pGUvE7D5xhtvyO9+9zvR\/0Y9jj76aDn99NNF\/5u1A9gENm37JLAJbAKbwCawCWzaji1p2csEbO7Zs0fGjBkjGzdujPzc\/fr1k+XLl0vHjh0jX5PWicAmsGnb14BNYBPYBDaBTWDTdmxJy14mYPOVV16RsWPHypNPPimf+tSn5POf\/7z87d\/+rbRq1aqsDi1atJD3ve990rJly7S0inwfYBPYjOwsEU8ENoFNYBPYBDaBzYghI3OnZQI2VZU333xTnnrqKfnBD34g69evl2OPPVb+6Z\/+SS6++GLp3bt3JqGyXG0Cm8Cm7Tcd2AQ2gU1gE9gENm3HlrTsZQY2\/Q\/8+uuvyxNPPCF33323\/PznP5dOnTo1g2fPnj0zD57AJrBp+wUGNoFNYBPYBDaBTduxJS17mYRN\/8Pv27dPHn\/8cbnrrrvkkUcekRNOOEEuu+wyGTJkiHTr1k20Oz1rB7AJbNr2SWAT2AQ2gU1gE9i0HVvSspd52AyC54MPPigzZ86UD3zgA0wQiuklrXc\/I+0enSNr1qyRQYMGeVcbiJnfp0n6td8f02Lp6Rv3tpGJz3aVFSuAzZqEDLkY2AQ2gU1gE9gENm3HlrTsZR42Dx48KC+\/\/LL86le\/8rKb\/\/d\/\/yft27eXf\/mXf5EvfelL0q5du7S0inwfMpvAZmRniXgisAlsApvAJrAJbEYMGZk7LZOwWQ4wL730Uhk8eLB8+MMfltatW2dOzGBD4M8gJi0skAFkABlARt4gQ8tLT0fSVr\/6dXmJC8YHNBb2799fmpqaZMuWLTJ69GhZNuGgDDi5+rNWOuPJ50TGLWkR2lt30f6Z0vXt02q6QVOrTXJ\/mxkl9pMatJmESlqGel6XGdg0gKnd5D\/60Y+aM5if+cxnZOjQoZkHTH8l2nSqvDQqBBe3rzF+wEcHHx18dOTtowPYPOyzNrnAbbRxYz0TsKnrbE6aNEl++ctfel3keQRMYJNMhptX9JBVYBPYxA+ATWCztJUls+ky6ti1nQnY9O8gdNJJJ3lrbFY7evToITfffLN06NCh2qmp\/93mFwyQAWQAGUBG3iCDng63YScvcYHMJplNo0AmYFMzmzfddJNs3bo18hsKbEaWqvlEZqOX18zm2Jz4NVP9irwEFyCjel3WcgZ+wMdnnj4+gU1gM1OwWUvjm8VryWwyG922XwIZQEaeIIOPDtstQKm9vLQHwCawCWw6bAuATWDTtnvlJbgAGbZrPp+QgR\/gB34fYDb64XH3NlapcetdbqxnohvdzaPVzyqwCWza9j5gk8wmmc3Db1WeJobYbgvy5AdkNslsZiqzyZjN8s0RkAFk5Cm4kNFygRZHBqx950yRA51OqelmjOEuLx9juH8jw4cPl1p3lgM2gc1MwaZ\/NnrU1rNfv35sVxlVrHfPI7gQXFQB\/AA\/wA8qN57AJrCpHmLTD2z2eMYM\/Zk4nW50B9Vg06nIbJLZJLNJ96lRgPaA9iBP7QGZTTKbmcpsOuC9upoENhmzadsBgQwgI0+QwXAK2y1Aqb28tAfAJrCZK9jUrSz37t0rBw4ckHbt2slRRx2V6E02L6i5eNasWd5WmNWO4HW6P\/vs2bOlbdu2oZcCm8BmNZ+K+\/e8BBcgI27NxjsfP6jto2Nvi52yr+XOqqLreQ8fvUiuueYaGThwYNXz0z5h27ZtMnnyZNnfd4gc6NS3ptu3fG23HPPUipJnNfZHdH1F+rXfn9j+jjdby9wXOnu2BwwYILt375Z9+\/Z562qP\/XTte6Nv3yPyjTUtQst+xpvDpOs7te2NXs4PdJ1v\/Ylz2OSCOPfNyrmZ6kZ\/9dVXZe3atd7i7lOmTPHAUh3zW9\/6lvzrv\/6rB5vHHXecKCReeOGF0qJFi8g6akXrlpgrV66U3r17y+bNm2XUqFFy1VVXVQTOdevWyfTp08UsV2DGl5544ollgdOmUxFcagsuUR3E5ticqPeMcx5+gB+Q2Tz8xiSZjW7AQd91DhSoRQH9AJk7d24s4LTJBbWUvV7XZgY2d+7cKVdffbU8\/vjj8olPfELmzZsn73\/\/+2X16tXyzW9+U8aOHStnnHGG3HXXXfLEE0\/I97\/\/ffnIRz4SSTcDiIMGDZKpU6c2X6MgqXC7fPly6dix4xG2zHXDhg0rAdIguAYvtOlUQAaQAWTUBhmRGol3T+Kjw87EEJXTdKGuWJGNng5TtwoJ3bt3j+MWnIsCzQps2LBBFi5c2JyAiiqNTS6Ies8snZcZ2FSonD9\/vixatEjOO+88L2upSyJ95Stfkb59+8r1118vrVu3Fk3vK3ieffbZHjjq76odJoupAKvAaY5yvzd\/V+e47bbbYs96t+lUwCawCWwCm\/42SZekYekjkXFLWpQEfNNWXrR\/pnR9+8ju06x\/SFSLY\/w9Gwokje9Jr8vGU9deikzA5l\/\/+leZNm2aNwbylltukaOPPtp7sk2bNsmYMWO8rwgDiTp+U7vRn376aVmyZIkce+yxVVUol4ms1pWumc\/HHnvM69KfMGGCbNy40bvXuHHjSjKkwQIYpwq9c1O3AAAgAElEQVQb79OlSxfRn6iHZnFHjBhhNbhopsHoaezXup6aP5OxdOlS6dmzpzccQj8Mlk2ofWyO6TYLK3u54BJVYz3PBCK\/\/TjXuz4XPyj96MAP7K2zWZT2ANh03UoVw76J75VixY4dO0R\/\/IcZB8sOQnX0k3Ld3Pfee6\/ceeedXmaxa9euzSXUbKNWeLnu7zD484\/XNH8v101u\/q73WbZsmXTr1q15rGecMZthko4cOdKDx6iHAu51111nFTYXLFgguk6pB4jv2rcJm\/oxoJpt377dG+9qEzbDym4TNv32o9ZRGufhB6WwiR\/Yg82itAflYFM\/ivUn70eSSSt5f+Z6lN\/AZqVYoT21q1atCi0esFmPWnv3nmGwqZOBNMv55z\/\/+YiJOAqBmum54447pEOHDlVLXi6zGRU2g85RLR1u\/h42NihuZlNtjR492ips6otw5plneroZ+zZhU+tFG74tW7bI+PHjrcJmWNltwqbfflXHSvEE\/KAUNvEDe7BZlPYgDDYVModdcZVsf\/ZQr1Wej+Cklddff93rMVy\/fn0q0KPxVIe+ffCDH8yMjMoKevjnatRaOBPfK8WKsMxm0rGetZY3K9dnohv9jTfe8MZktmrVqrkbXZdI0C7Yz372sx5smUNnp2umr3379nLrrbc2d7lXEjRpN7o6qr6oZga7uUc1SK0Go3EqnzGbpZDhB\/9qY7Ti6Jz1Ljb8AD8wH4eM2RRJMhs97B0379VrZ4yWd47pFKfJyNS5rXc\/LW2evq9kDKuBTZ3fEFziz2aMUiFMTFSo88+LqLdILmEzbobStub11jbu\/TMBm1poHR954403yg033CCf+cxn5J577vGWO9JZ5x\/60Ie859Lxmg899JA3fnLGjBmRu6OTThDSMi1evBjYjOhVWZt9GrHY3mnAZuPOQsYPwhUo2rallWDTxoSrOH5m+9ywuqwEm3p\/myAGbFavUWBTCS4Dh2YsZ86c6UGmHjrLXL+SdIyj\/r\/+\/etf\/7rcd999XhewThrq3LlzpJKXy0RWW\/qo3AQilj4Klx3YjOSOiU4is0lmk8zm4VfHdmazaLBpQFTH1psu5mC3u47rD86LMOtOm5rQ8fmf\/OQnvYm8wQm0ps3yN3j+jVQM7OrfdW6EHsENU\/z30\/Lo+toal3VTFbMe95w5c7y1s\/XQv+mEXvO7IFCbeRimTMH5GBMnTvRWw9EhfGHP738HyWzGC2WZyWxqsXWcps4y1\/EOOu6jV69ezQu3K2zq+JPjjz\/eW4g9ylhNvxTBxdmrzUQ31wa70stNZvLfy+YXDJABZAAZtUFGnCaRDHfjZrjJbB5+E4IJk7CETDAZE+zp8\/cY9unTxwNO040ehD69c\/B6A37BDVPMutbBMppYaIA0Lmya1WXM7n8GrrVs+jv9tz5DpQ1bgM04rWnpuZmCzeSPEe3K4JdWcLvKct3mwa+5qEsfxf3yCXsKYBPYBDaBTaMA7UHy9qCosBk2Qcif0TPty913310yGdffDT9kyBAv2RM2\/lOvj9KNHpZ11BVL\/Fs\/m0ykbvASdj\/9u7kmLmyGxVc\/gBrYDG7iErwuaTIp6XXR6Cb7ZxUKNtOqDptORXBJHlzi1DcZrcbNaOEH4QowZvPQahy2JlzF8TPb50Yds1kuNgUTKv7y+bvKy00AKgeb5vemi90PuWFjRs3vrrzyypJMqSmPHw6TwmYw6WQypQY2q01yShrfk15n21fqZS9TsKnd6LqkkTqROoAu7q6\/+\/GPf+wNZv7Tn\/7kzXTTsZuats\/qYdOpbDaGRQsucfwD2AQ21V\/wg8b1g6JmNoPZyOCQMvV7f8ZQN1cJHtUyl8G\/my52zUKaHsRq4ylNOfS\/LmDTPLcfeMMym8BmnMgZ\/dzMwKZ\/gpB\/YO4DDzwgX\/3qVz3IPPXUU+WnP\/2ptGzZ0ltjM0vrefklBzazsRdy9NcAyDA+a3O91azsiY0fkNks9yFh82M+jp\/ZPjdqZlPva7rHX3zxxeYJQNUmy1ab2R6EzeD4SL1vcAxmpcxm0m70cveIYo\/Mpm2vLLWXGdjUcSU6E0yXPrr88su9rSvNmpqa4dSvo3bt2skzzzzjpdd13TBdAkn3UM\/aAWwCm7Z90mZQJMNdvnbIbJLZtP3upmEvDmxqeYITZMMmvgaXDAwCaaUxncFzjS29t1m3uhJsanax2gQhA4eaiNLzTXmefPLJI+5hYNM\/+95kOulGT8NDRTIBm2ZRd90jfd68eR5U6vHss8\/KqFGjvGUILrjgAu93ZmehpqYm0e2izLnpyBXtLsAmsBnNU6KfBWwe0irJkjfRVSbD3cgZbrrRS9+EcrPBzfhKPTs4yTVs6SOzYHwQ3hYtWtS8pJFCnsZ2vaeZgFMNNvX+lZY+0oSUv7te76Gbw+gW12FLHwXHjypkXnzxxXLTTTd5cHrccceFjhMNth9J43vS6+K0X1k+NxOwWW45ofvvv1++9a1veY5wwgknNOsYd2\/0tCvAplMBGUCGKoAf4Af4weGWPMlHRyXY3N93iBzo1DftUGHtfi1f2y3HPLXiCDi0doOMGFL4fOGFF6xuPxn30ZLG96TXxS1fVs\/PLGzqWvO6t\/hzzz13RAYT2EzmTnSflteN7tPG7T6N87bgB43rB+X2Rp88ebLovtV5P4J7o+f5ecIWnS+3E2Daz5kUGpNel\/bzubpfJmBTu891Ta2OHTt6W1bqjkGvvvqqNybzrLPO8v5rjuBCrGEz51yJFdWuTacio0VGi4xWbRmtqO+tngdsFgs2tc519RP9yfvRo0cP0Z9GOYLd3vpcNtaurlWfpPE96XW1ljcr12cCNlUMnV1+++23i47zOPfcc0W70HX\/c53ROmDAgGa9fvvb38oXv\/hFb2kEP4RmRVA\/GNh4MYBNYBPYBDaNArQHyduDrH9IZCmGUZbyCiSFxqTXNUpdZAY2d+7cKTpj7PHHH2\/WdsSIETJ9+nRvvU2dRHTXXXfJ0qVLva0qFU51O8ssHjadiuCSPLjE8Y2sByL8AD\/go6O2j46sv+Nx2ivOrZ8CSeN70uvq96R275wZ2NTH0q5zXQJpy5Yt8pGPfMSbga6gaf6mmUyFTp2dzqLu8R2BMZvlNct6IAI2gU1g0w1s0o0eP5YU+Yqk0Jj0ukbROlOwWUlUnTD02muvyTHHHJPJtTX9ZbfpVEAGkAFk1AYZcRprPjqKNWZTQZMJQnHeEM5NGt+TXtcoiucGNvMkuE2nAjaBTWAT2DQK0B4kbw8qLX00udcu6XLUgTyFmZKybtzbRlY3dSiZQFNt15+wdS5zK0CKBU8a35Nel+KjOb0VsOlAXptORXBJHlziVC0ZrcbNaOEH4QoUbVhNJdi0sU1rHD+zfa7C5sRnuwKbtoUNsZc0vie9LoVHSuUWwKYDmW06FbAJbJLZJLNJZrO0oba9qDuw6SAQNqjJpPE96XWNIiOw6aAmbToVsAlsApvAJrAJbFYKVTYymybW+O8za9YsMdtRmm53\/fuyZcu808y+4rrete7u89hjj0n\/\/v29Sbx69OvXT5YvX+6tod0oR9L4nvS6RtEN2HRQkzadCtgENoFNYBPYBDZdwqbuzjNlypTmPcX1XgqPixcv9raL7t27t7evuUKmWT\/aLLpu9jo3+5gbQA3bBchBuE3dZNL4nvS61B\/Q0Q2BTQfC2nQqYBPYBDaBTWAT2EwKm7qcYLlj3LhxZfcZDwKowub27dtl9uzZYnbu808yUthcu3ZtSSbTZDv91zgIuamaTBrfk16X6sM5vFlmYXPbtm3eV9WDDz4oH\/jABzwH1jU358yZ462\/ed5552V2CSSbTgVsApvAJrAJbAKbSWHz7LPPbu4K99sIm40e3CKyW7duJZlNvX7q1KnNZoKwqd3ofrAENg8rbpMLHDKhM9OZhM3nn39exo4dK0cddZScfPLJ3r61Cpvvec97ZOLEibJhwwYvnT9o0CBnwtRi2KZTAZvAJrAJbAKbwKZL2NQs5qhRo7zMpekGD8tsApsiSeN70utqYZEsXZs52Dxw4IA3NkR3EZo7d678\/ve\/9\/5tBhnv27fP28Lyve99r9x8880ekGbtsOlUwCawCWwCm8AmsOkSNsMykBp7Jk2aRGYzIHzS+J70uqzxTdLyZA42X3nlFS+redlll3mpf60gP2zqg957772yevXqzM5ys+lUwCawCWwCm8AmsOkaNv3jLU2mU+\/pnyBEZpPMZsPAphkzcsUVV8hFF10UCpv333+\/fO973wM2Y9Z60RZxjiMPi7qzqLv6C37QuH7Aou6lLWJwzKaZba5n6VjNefPmeYkeM9s8bIwnYzajD+WzmYSKE9uycm7mMptmuYTjjjtOrr\/+enniiSdCu9G1+\/zWW2\/1Jg1l7bDpVGQ2yWyS2SSzSWbTbWZzRNdXpF\/7\/VkLJZHLs+PN1jL3hc4lOwhFvpgTYymQNL4nvS5W4TJ8cuZgU7V69NFH5dprr5Urr7xSTjjhBPnOd74jixYtkpdfflnuvPNOeeSRR+S73\/2uNys9i4dNpwI2gU1gE9gENt3Apk4+nTx5sjfpNO\/HwIEDvXkOPXr0yPujZLr8SeN70usyLUaMwmUSNg8ePCg\/+clPvAlAf\/rTn0oeR2ekz5w5Uy699FJp2bJljEdN71SbTgVsApvAJrAJbLqBTbWqwKk\/eT8UMgFN97WYNL4nvc79E6Vzh0zCpnl0nXmus9H\/53\/+R9544w1v6yvdCqt9+\/bpqJPwLjadCtgENoFNYBPYdAebCZt5LiuoAknje9LrGkXmTMNmXkW26VTAJrAJbAKbwCawmdd42GjlThrfk17XKPplDjY1m3nPPffIX\/7yl6oa6ySij33sY\/LBD34wU13qNp0K2AQ2gU1gE9h0B5t0o1cNtZzgUyBpfE96XaOInznY1HU2b7rpJm\/JIzNe86STTvL0\/uMf\/xiq++WXXy4zZsyQdu3aZaJebDoVsAlsApvAJrDpBjaZIJSJkJmrQiSN70mvy5U4FQqbOdjUsm7atEm++tWverPRhwwZIm3btvUeQXcXeuihh7w907\/97W9Lr1695OGHH5avf\/3r3kLwI0aMyES92HQqYBPYBDaBTWDTDWya9vVrww9Kt46ZCB+JCvHkcyJ3\/LRFydJHZhnBOHujJ7q5xYvC1vK0aN6KqaTxPel1VgqdASOZg03zguie6BMmTJAWLVqUyKQz1ZcsWSIvvfSS3HLLLd46mytWrPAyoXfccUcGJE2+w0BY4YFNYBPYBDaBTbewuWzCQRlwcibCR6JCKGyOW5J\/2Ez08ClflBQak16X8uM5u13mYNPsIDR+\/Hj51Kc+FfrgP\/vZz2Tp0qXNOwj94he\/8LKduo1lFg6bTgVsApvAJrAJbAKblWIbsJle5E8a35Nel96Tub1T5mBTJwhdd911oplN\/W\/r1q1LFNCu9AULFshzzz3n\/VfHaeqerg888IC3X3oWDptOBWwCm8AmsAlsApuuYdPEGv99Zs2aJUOHDvV+pV3ce\/fu9X7Wr1\/v\/U7\/fsYZZ8ioUaNk+\/bt3jaXZi913V99ypQp3lA47YXUQ5cvXL58ubc5y7Jly7zfjRs3TqZOndp8D\/0f\/fe6devkscce85Y7DF7fseOhMQ9mD3e9tx433nij3HfffTJnzhzp3bu3ExxIGt+TXufkIepgNHOwqRr86Ec\/khtuuEG+\/OUvy+c\/\/3nRWed66IShu+++28tiqvPpxKCnn37am1CkDqnXZOGw6VTAJrAJbAKbwCaw6RI2DRj6IU1hT2OtgUezd\/qaNWtk0KBB3tC14cOHy+DBg2X27Nne3Ao9R8FP\/62TrxRCBwwY4P1bj2nTpnmgamwYWNR92NVmcK\/16dOne0CrwGuG2CnQKoyaXlCzd7v5965du5rL7IIHksb3pNe5eIZ62MwkbGr2ctWqVd7XyVtvvVWii+4gpF9LI0eO9BZ6V+dVp9Ysp5m1Xg8h\/fe06VTAJrAJbAKbwCawmRQ2TRYy7Hp\/VjH49yCA+kFSwdLAnYKfgqIeJhvph00DksG\/h9kIwqb2Wmom1GQy\/fY1g6mZTwO6pp2cNGkSsFlvCAq5fyZh05RTl0HSPWt1FyE9+vTpI2eddZZ07tzZ+7fCpmY7O3XqJEcddVRm5AU2V0jPnj1ly5YtMnr0aLEx+D5sTJLR+aL9M6Xr26fVVP9NrTbJ\/W1mlMzmrMmg5Yv56OCjg4+O2j46wt5x817ZaKMsv\/KxzNkas2kAcuPGjd79\/d3iwZniUWBTE0PBbKkfEIM2grAZhEk\/bC5atMgro+mC1\/8Py9DGEjLCyUnje9LrIhQpF6dkGjZzoWBIIW06FZABZAAZtUFGnHaEj45DXaPz+zRJv\/b740h3xLkb97aRic929VYLycLHJ7BZWkV+sPOPfTTd1mGZTT\/cAZuHhhNEPWxyQdR7Zum8TMOmThZ68803j9BLu9mfffZZb5xmVhZy9xfSplMBm8AmsAlsGgVoD5K3B8Bmedj0ZwzNutbqa\/4u6axlNulGzxJKVi9LJmFz586dcv3114suaVTuMLPazFiO6o+a3hnAZjYyGXFqnIxW42a08INwBVrvfkbaPTqnZOiIabvIbMbxmvqfW2s3usKmf3ykyXTqk\/knCGUps6kThsaMGSNMEKq\/\/0UpQeZgU7OW+gWlyxjpTDaFSl2s\/aMf\/ai3lIHOVNdJQprq1\/EkwUXfozy063OATWDTto+R0Uqe0YpTF3x0NO5HB5nN8plN\/YuZba7\/r7FVJ\/bo7wzMZS2zqRlYlj6K07rV99zMwearr77q7RykcKn\/1UPBslWrVt5AYF3ja+LEiXLhhRc2r\/9VXwmPvDuwCWza9klgE9hUBfCD5H5QCTbHfjrfOwht3yPyjTWlOwjZboPyYE\/h89Zbb5X58+c3z2C3Xe6k8T3pdbbLXy97mYNNM+j4qquukgsuuMDTJTieRNP9v\/71r0uWPKiXgGH3telUBJfkwSWOT5DRatyMFn4QrgDd6OItmzd58mRv1ZO8HwMHDpS5c+dKjx498v4okcof7PoPrsMZyUiCk5LG96TXJShiJi\/JHGyazOZFF13UnLnUStIvFe1O79ChgzeWUxeb9a+\/lSV1bToVsAlsktE6\/HazBNahj4J950yRA51OqanZAzYPyafAqT95PxQyiwKapq78Xf\/6u0rrhtqq36TxPel1tspdbzuZg00zZvOll17yus8VLjU1fs0113j\/Pv3002XJkiXy8MMPN8NnvUUM3t+mUwGbwCawCWwaBWgPkrcHWe+9yFocozzhCiSN70mva5R6yBxsqrC6rNH48eO9xduXLl0qbdq08XYK0gVedQH3559\/3vuCUQAN7p2ehYqx6VQEl+TBJY4vZD0Q4Qf4AR8dtX10ZP0dj9NecW79FEga35NeV78ntXvnTMKm6dr43\/\/9X28ikO4OtGPHDm9mnHahX3rppaJjOjXrmcXDplMBGUAGkFEbZMRpI7IOJLQHyduDcnVLN3qcN4Rzk8b3pNc1iuKZhM0XX3zRy2qaxWXzJrZNpyK4JA8ucfwGyGCCkPoLftC4fhBWtwqaY4d9VZ7e\/lSc5iKT5xZtglC9KiFpfE96Xb2e0\/Z9MwebZoLQ+eef7+2rncfDplMBm8AmmU0ym0YB2oPk7UGlpY\/Oe+NqaX\/w+DyGG6\/MTS03yVNHrS1ZoN\/Mzl6\/fv0Rz7VmzeGtFsO2naxFiOB6nLXYyuK1SeN70uuyqEGSMmUONl955RUZO3asfPGLXxSdkZ7Hw6ZTEVySB5c4vkNGq3EzWvhBuALMRj+8bulF+2dK17dPi+MqmTo3rP0ysHn22WeXrEkdjE\/AZryqTBrfk14Xr3TZPTtzsHnw4EG599575Z577pEbbrhBPvShD2VyElClKrXpVMAmsElmk8wmmc3SFjfJEliVMptFgs0ghAKb8QAtaXxPel280mX37MzBpmY2b7zxRnnkkUe83YLKHeyNHt+pipbJiKMQmU0ym153ZKtNcn+bGSXdkXH8yPW5fHwm\/\/gENg9pVw02w7rfg\/HWAOrGjRs9m\/6\/B7vRzVqYunThGWecIVOmTJEhQ4bILbfcUnLtnXfeKcuWLfN+518vs1p5zKYv\/fv3P8Jmx44drb+SSaEx6XXWH6BOBjMHm\/v27fOymn\/5y18qSvK+971PLr\/8cmnXrl2dpCt\/W5tORXBJHlziOAaQAWwCm4e7lef3aZJ+7ffHeYWOOHfj3jYy8dmusmJFNravBTYPVZHGlLvvvrt5B75gZrPcHuiDBg3ytow255t\/q029Zvv27Z7NRYsWeffRc\/X3ej+zAYvZy3zAgAHeuXrosoY6rtSMIzXn6N7seo9q5VHYnD59urcO99ChQ5thWvd31zLYPpLG96TX2S5\/vexlDjZdCmEq29zDOGecewa3yAq71qZTAZvApgkQ7BwjkqT7NM77zUdH4350FBU2wyYI+TORUbrR\/cCnMUn\/XW4HP3Ouabf85wVBUs8JbkcdtzxhMTloM04bUO3cpPE96XXVypOXv2caNjXLqc6puwrpzkGtWrWS3bt3y\/HHHy8tW7aMpbFW9KRJk2TlypXSu3dvz+6oUaO89Tr1ayjKYa7RZZkqbZVp06mATWAT2Dz8dgKbbFep3pDED4oKm8EJQgbmTjzxRC+7qN3UY8aM8bKAmkk0h8kYmn+bru1qIOffQnLw4MHNGVS1ozFUu9HnzJnjxeE4sBmnPNXKGCXelzsnaXxPel0tZc3StZmETYVLdZabb75Z3nrrrebxILpb0IQJE0TT4zNmzIjchR6W9jdOvnbt2kh7rPvHjVQbL2rTqYBNYBPYBDaNArQHydsDYPPwe+TPTupv\/bBpYLHcOMxqIGe6zrUbXBM8w4YNa07oJIHNJOWpVsZaICxpfE96XS1lzdK1mYRN3SVIt6K89tprpWvXrt4e6JpJPPbYY72Z6jNnzpSJEyfKiBEjImkZlro3X1ma3TRjQyoZ8w9Cvu+++8hshoiVtTFakZzj3ZPoPm3c7lP8IFyBok0YBDarw2afPn2OyHKaRIsZAxm1G10zpRo3Fy9eXNKjGCezGaU8YWAJbMZp9dI5N3OwaRxbu8rVWZ944omS8SG6NNKSJUtEt7JcsGBBpOxmsAvdSBu1K91\/\/VNPPSXVsqHmC0aBWXd18B9dunQR\/Yl66PMrVO87Z4oc6HRK1MtCzzPBRQfsm+4SY9\/mhADdz75nz57i7c4xdqwsm3BQBpxcU9Gbu83Cym5j2RITiPz2ayux3avxg9KMFn5AezBuSQtv8lGwLSvXHgCbh96hSrPRDdz5s5Ems2i60U1Pof8cP4DqrHI9NH6be+m\/tcteY0IS2KxUnnrBZqVYodtr64\/\/2LZtm0yePDmzK13YjVhHWsscbBpH1rGUF1xwgTeTLTgYWTOf+rVUadyk\/1HLwWbYSxOUKHhOnAlCYZU3cuTIyBlZvV6XlrjuuuuswqZCunaR+O3bhE2deKVfwTo7UWcJ2oTNsLLbhE2\/fdcvXxz7+EEpbOIH9mCzKO1BJdg8481h0vWd\/C7qvrfFTnn46EWRdxDyT44NTsgxSRhtv\/VQyOzVq1dJkiXO0kfGno7fvOSSS2LBpn5IVCvPQw89JI899ljJ2NA0MpuVYsXq1atl1apVoU28f\/emODEg7+dmDjbNDkKXXnqpN9YjDDb1i+JXv\/qVl+HUrvVqRy2w6V\/SQfdqjwObc+fOle7du5cUL25mU8uu23bazGzqi3DmmWd65TL2bcKmDnvo0aOHbNmyRcaPH28VNsPKbhM2\/far+VWaf8cPSmETP7AHm0VpD8rtja7Zpg0bNqT5Oju5F3ujO5H1CKOm57JSrAjLbKqPLVy4kMxmOtVU\/S46OUgBT7vJdb2uP\/zhDyWZzd\/\/\/vfeJCHNemqaXicNVTuSdqOHXRcHNm18wTAhoBQy\/JoabWzCpo06q+aPSf6OH+AH5uOQJbDszUZXTbVrV3\/yfugHvv5wuFUg6USfpNe5fZr0rGcus6mPvnPnTrn66qu9hd1PPvlkefzxx+Xiiy+W559\/3ttZSFP6utPASSedFEmppBOE\/Es4hN2o3DqdNp0KyAAygIzDb1+SJW8iNRLvnsREscadKJb1uo3jp5xbPwWSxvek19XvSe3eOZOwqY+4a9cu0YkmP\/zhD5u3rXzPe94jn\/vc57y1MYPd05VkKTc2M0qWMmg3yjU2nQrYBDaBTWDTKEB7kLw9ADbtwkNRrSWN70mvaxSdMwubRmCdfa57pL\/99tuiW1TGXczd2DELwga3xIqzqLvaAjbLuz5LH7lrFoCM5JARp1ayDiT4QXI\/MHUbtkpIHB\/h3GIrkHRWObCpNJehQycI6Vqan\/jEJ7zlc5LCZdgjmco2fwt2gwfXBAuzAWwCm\/V4XYCM5JARp76AzcbtRjcztrWOOVCgFgWSTMYCNjMIm7o245NPPumNydStJIcMGSJ\/8zd\/Iy1atKjFP1K71qZTARlAhiqAH+AH+MHhJjzp2F0Fzn0td1aNBU0tN8lTR62VsBVFql6cwglmZvNrZ4yWd47pVNMdW+9+Wto8fV\/Jsxr7k3vtki5HHUhsX3u7Vjd18Gwfd9xx8uc\/\/9mbe3H77bfL14YflG4dE5v2LlQ\/uOOnLULLft4bV0v7g8fXdINyfpBkMpZNLqjpoep0cSa70XVGus5G1yzigw8+6HWjf\/SjH\/XAUzOeHTp0qJNc0W5r06mADCADyKgdMqK9uSJkNhs3sxnVB\/Q8\/MCOH5ihVTp8rX\/\/\/tLU1OQtiafL+dlcfzkPq5TY5II4vpyVczMJm35xguCpOxKce+658oUvfMH7b5Slj9IW26ZTAZvAJrAJbBoFaA9oD\/LUHgCbh9sum1yQNtPYuF\/mYdP\/kH\/961\/l3\/\/9371ljzp27Bh5ByEbQsWxYdOpCC4ElzwFFy0rE8XitBbxzqU9oOujEsUAACAASURBVD3IU3sAbAKbRoHMw6ZmMnVfaIXM9evXe+W+8MILRRc21q2syGxGD1Zmb\/SwLgebOwjpDk86uauo3SXRayT6mUAGkJEnyOCjI\/q7neTMvLQHwCawmWnYDALmW2+95Y3ZHDFihJx33nnSvn37JO9nateQ2QQ2bTtbXoILkGG75kvt4Qd8dOTpowPYBDYzC5tmAfaNGzfKKaecIiNHjvQymZ07d3bbilu0DmwCmxbdyTMFZAAZ+MHhtyrpbPSo7yUThJggpL5i0w9sckFUP87SeZnrRteZ5w888ICcc8450q1bt9wsd+SvVJtOBWQAGUAGkGEUoD2gPchTe0Bmk8xmZjOb1Uhcu9h1f\/Sf\/exncv3113sThbJ2AJtkNm37JJABZOQJMrSsTBSz3QocCS77zpkiBzqdUtONXI7lBzaBzVzBpq47\/9JLL8ndd98tOrlFs5\/9+vVjNnrMJsZlo0JwiVkZMU8HNoFNYJMMd94y3MAmsJkL2NQ1Nh9\/\/HFZsmSJN2ZNj65du8qXv\/xlb1ehrE4UIrNJZjMmS1Y9HdgENoFNYBPYLG0q8zR21yYXVA0YGTwhc2M2VaNdu3Z5+6OvWrXK23HAHPPnz5fBgwdncrkjf93adCogA8gAMoCMvEEGPR1uo31e4gKZTTKbmctsahZT19PUNRp\/9atfif7bLHd06qmnyrXXXitTp0711tbM+gFsktm07aN5CS5Ahu2aL7WHH\/DxmaePT2AT2MwUbL7yyisyfvx4r8tcwfKSSy6Rz372s83LHZnlkIDN7A4EBzKADKMAE0Pc+QKwCWwCm\/ns6bCZhHLXwriznIludAOTb7zxhlx22WXyyU9+Urp37y4tW7b0nhzYHC5Zn3UIbLp7SfMUXPAD\/ICPDrc+kKf2gMwmmc1MZTbfeecd2bx5s\/z4xz9unm1+0kkneeCpE4GOPvpoufLKK+lGz\/ASF0CG2wBDRouMVp4gg\/aA9sDvA7qKTP\/+\/b05GEXdxpjMpq4rlKFD19H89a9\/LXfddZe3nqaO3TzhhBNkx44d8r3vfU\/OPffcDJU2vCg2nQrIADKAjHx2m7loqGgPaA\/y1B6Q2SSzmanMZrlGOTgrvXXr1nL++efL6NGj5cwzz8zsrHRgkwlCtkEDyAAy8gQZZDZttwCl9vLSHgCbwGYuYNMUUrvZn3nmGW8pJF0S6a233pKzzjpLli5dKh06dHD7ViewDmwCmwncpuIleQkuQIbtms8nZOAH+AHd6OHvrg4pyMOqOrY9OBMThOI8lO4e9PDDD8svf\/lLtquMI5yIsINQecGaWm2S+9vM8MYMZ7EhADbJbJLZPPz+5mkx75jNdKTT89IekNkks5mrzGakty9DJ5HZJLNp2x3zElzIaNmueTKbYYoCm7+R4cOzv0oJsAlsApsOYwKwCWzadi9gk8wmmU0ym0aBvLQHwCawCWzapgGfPWAT2LTtXnkJLmQ2bdc8mU0ym0cqkJf2ANgENoFNhzEB2AQ2bbtXXoILsGm75oFNYBPYLPdW5Wk4hU0ucNvKuLGeuwlCbmSwa9WmUwEZdJ\/SfUr3ad66T\/nosBtTgtbyEhfIbJLZJLPpsC0ANsls2navvAQXIMN2zZPZJLNJZpPMptt2JQ3rZDYdqAxsApu23QrYJMNNhpsMd94y3GQ2yWyS2bRNAz57wCawadu9gE1gE9gENoHN0paVMZu2I407e2Q2HWgLbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsFQo2TcA24s6aNUuGDh1aUevXX39dpk2bJuvXr28+r9p1wCawafsFBjaBTWAT2AQ2gU3bsSUte4WBTQ3WkyZNkpUrV0rv3r1l8+bNMmrUKLnqqqvKAueePXtkzJgxcuKJJ8rs2bOlbdu2zdcNHjxYpk6dGlpPwCawafsFBjaBTWAT2AQ2gU3bsSUte4WATQONgwYNKgHEdevWydq1a2X58uXSsWPHIzQPAqo5Icp1w4cPlzVr1ojes5YDyAAygAwgI2+QoeXduLeNTHy2q6xYwcdnLTEg7Nq8xAXjAxoL+\/fvL01NTbJlyxYZPXq0LJtwUAacXJsyTz4nMm5Ji5JYa7S5aP9M6fr2aTXdoKnVJrm\/zQyrsdwGF9T0UHW6uBCwabKY8+bNK4G\/cr+vVhcKm4sXL27OkgbPJ7NJcKnmQ3H\/npfgAmTErdl45+MHfHzm6eMT2Dz8ftvkgnitRjbOLgRslstQRulKD6um2267TdRmpYyoZjavueYaGThwYImJLl26iP5EPZ544gkZMWKE7DtnihzodErUy0LPa737GWn36Bwv02Ayrsb+\/D5N0q\/9\/prsm4Zl6dKl0rNnT9m6dauMHTvW6hdsWNltfsH67dckhuWL8YNSyMAPaA80o4UfZNsP\/NltzWzu2LGjEHFBn1N\/\/Me2bdtk8uTJVrKklsNLKuYKDZume33YsGFVJwqZ2jBfJ5UmCQUnIvlrcuTIkR48Rj02btwo1113nVXYXLBggfTr188rgrFvEzZVm27dusn27dtl+vTpVmEzrOw2YdNvP2odpXEeflAKm\/iBPcigPTjyDTbdp7QHtSUhDGyqjn379pVdu3YVIi6sXr1aVq1aFRoa6EZPI2LW6R7lMptxYdNkQgcMGNA8YSjskQxszp07V7p3715TZlNt6fgWm5lNfRHOPPNMr1zGvk3YvOOOO6RHjx7e2Jzx48dbhc2wstuETb\/9Orlr6G3xg1LYxA\/swSbtQXnYpD2wA5uq46mnniovv\/xyIeJCWGZzw4YNsnDhQjKbWQqststioxs9KmgagGOCUM\/CDgS37b9+n7L50eH\/wjYfSDY\/OpgYYt8TGLNZ+tER5sM2Pz6zmoXKix8wZvNwG8CYzYMHD9pvErNlsdYJQsZJdLkjswRSpSe06VR5aVS8LnlmnzpzfPwAyOCj4\/DrladZyC4ahby0B8AmsGkUKMSYzXLd5dWWMPI37uPGjSu7rmawMQE2mY1uO8DkJbjw0WG75kvt4Qd8dOTpowPYBDYLBZv6sAqWOlnFdItEmYluzqm0gHtYaAE2gU3byAFkABl5ggw+Omy3APn86AA2gc3Cwaa\/oTYPH5xRHlw\/U5c4WrZsWdlWo9x4HmAT2LQdaoBNYBPYpBvdKJCX9gDYBDYLCZu2AaCcPWAT2LTta3kJLmS0bNd8PjNa+AF+4PcBdhA6tPKLrYnDbr3LjfVCjNl0I115qzadCsggo0VGi4xW3jJawKbbqJOXuEBmk8wmmU2HbQGwSWbTtnvlJbgAGbZrnsxmmKLMRj+UJcv6UmjAJrAJbDqMCcAmsGnbvYBNMtxkuMlw5y3DDWwCm8CmbRrw2QM2gU3b7gVsApvAJrAJbJa2rHnKcNvkAtvxJQ17jNl0oLJNpwIygAwgA8jIG2QwnMJBYAlJaNCNLjJuSYuSLSBNzMzaTlI2ucCtd7mxDmw60NWmUwGbwCawCWwCm\/nNaDkIMc0zm4FNYNOFf7mwCWw6UBXYpBvdtlvx0cFHBx8dfHTk7aODMZuHfdYmF9iOL2nYAzYdqGzTqYAMIAPIADLyBhl0ozsILHSjHyEqYzbd+plN68CmTTXftQVsktm07VZ8dPDRwUcHHx15++ggs0lm0ygAbNqmArG7UwCQAWQAGUBG3iCDzKaDwEJmk8ymW7dyah3YdCAvmU0ym7bdio8OPjr46OCjI28fHWQ2yWyS2bRNAyFfnbof7KBBg2q6E5ABZAAZQEbeIIPMZk3NftWL8xIXgE1gE9is+jonP4HMJpnN5N4TfmVegguQYbvmS+3hB3x85unjE9gENoFNhzEB2AQ2bbsXkAFk5Aky+Oiw3QLk86MD2AQ2gU2HbQGwCWzadi9gE9gENhlOkbfhFMAmsAls2qYBnz1gE9i07V7AJrAJbAKbwGZpy8o6m7YjjTt7zEZ3oC2wCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaATQdKA5vApm23AjaBTWAT2AQ2gU3bsSUte8CmA6WBTWDTtlsBm8AmsAlsApvApu3YkpY9YNOB0sAmsGnbrYBNYBPYBDaBTWDTdmxJyx6w6UBpYBPYtO1WwCawCWwCm8AmsGk7tqRlD9h0oDSwCWzaditgE9gENoFNYBPYtB1b0rIHbDpQGtgENm27FbAJbAKbwCawCWzaji1p2QM2HSgNbAKbtt0K2AQ2gU1gE9gENm3HlrTsAZsOlAY2gU3bbgVsApvAJrAJbAKbtmNLWvaAzQhKm0BvTp01a5YMHTq07JVFh03V5+mnn5a+ffvK9OnTZdmEgzLg5AhCVzjlyedExi1pIWvWrJFBgwZ5ZxqdL9o\/U7q+fVpNN2hqtUnubzOjxH5NBi1fnEfYxA8sO4HP5\/edM0UOdDqlphu03v2MtHt0Tug7Nb9Pk\/Rrv78m+xv3tpGJz3YV\/KAmGUMvzkt7YHxA2+0uXbrIqlWrChsXbHKBfY9ybxHYrKKxOsikSZNk5cqV0rt3b9m8ebOMGjVKrrrqqrLAadOp8tKoqIymYbnmmmtk4cKFYv4LbNb+IuMHZDbzmtmkPaj9\/Q9ayEt74IdNfYbhw4cXNi7Y5AL7HuXeIrBZQeM9e\/bImDFjvEza1KlTm89ct26drF27VpYvXy4dO3Y8woJNp8pLowJsun1Z8QNgE9g8\/I7R0\/EbD9yynuEGNg\/7rE0ucBtt3FgHNivoarKY8+bNa+661dPL\/d6YsulUQAaQAWQAGcG2JeuQwcenm4CdNz8ANoFNowCwWaFNCHahm1OrdaUbQNTuo4EDB9bU6mzbtk0mT54s+\/sOkQOd+tZkq+Vru+WYp1Z43RimXMb+iK6v1DxGa8ebrWXuC53l0ksvlR\/+8IfN\/x376drHbG7fI\/KNNS1Cy37Gm8Ok6zu1jdnc22KnPHz0ohL7NYlt+WL84JCg+AHtAX4gkpf2wMQEjTk9evTwYpmJD0WLC6bO\/PMOLIeJTJsDNhPApuleHzZsWOi4za1bt3ov1YYNGzJd+RQOBVAABVAABVAgHQU0yTN37lwPvIt2AJsOYFNNKnDqDwcKoAAKoAAKoAAKKGQWETS15oHNBLBZrRudVwoFUAAFUAAFUAAFUOCQAsBmBU9IOkEI50IBFEABFEABFEABFAA2q\/pAubGZ1ZY+qmqYE1AABVAABVAABVCgIAqQ2axS0QqWuguOmUFGF3pB3gweEwVQAAVQAAVQwIoCwGYEGeNuVxnBJKegAAqgAAqgAAqgQCEUADYLUc08JAqgAAqgAAqgAArURwFgsz66c1cUQAEUQAEUQAEUKIQCwGYhqpmHRAEUQAEUQAEUQIH6KABs1kd37ooCKIACKIACKIAChVAA2CxENdfnIatt61mfUnHXNBQwdb9x40bvdv369ZPly5dLx44d07g996ijArpix5QpU2TOnDnSu3fvkpLcdtttsmzZsubfjRs3TqZOnVrH0nJrVwpU8gOzqsv27dtpH1xVQMbsApsZq5BGKo4JLLNmzQrdQ76RnpVnOayAAc1BgwY1g4T6gq7qAHA2tqeYut+1a5esXLmyGTZff\/11mTZtmrz44ovNPmDOPfHEE2X27NnStm3bxhanQE9Xzg9UArO6i4kLYb5RIKkK86jAZmGqOt0H9S8XBWymq32976Zr0y5evLgENliftt614v7+\/ne+W7duofU\/b9480Y8Qc+g1kyZNKjnXfUm5g0sFKvmBAUv1D39Gu9xufS7Lie10FQA209W7EHczX7VXXnml3HnnnTJs2DAym4Wo+UMPCWwWqLLffVR\/tqpnz56RAdJcZzbNKJ5yjfXE1fygUtd6YynB0wQVADbxCasK+L9cFTbHjBkDbFpVOPvGynWjr1+\/ngxW9quv5hLGyVaGfZjUXAAMZEKBMD\/Q3+mQmiVLlnhjerVN0GPw4MEMpchErbkrBLDpTttCWvbvG68CAJuFdAPvof2TQQgmxfGDqLBpuk7VN5gk1Hj+EeYHZvtnfVqTzWbMZuPVfdgTAZvFqOdUnjI47obZ6KnInrmbBCcAaAFNkKG7NHPVZb1AUWDTtA16cyaNWa+CTBisBJvBcfyM6c5ElTktBLDpVN7iGA8b+A1sFqf+zZMaP9B\/B2cYa6ZTlzph5nFj+0U12AQ0G7v+zdNVgs3gR2e5iUPFUKoYTwlsFqOenT9lcN204A1ZZ9F5FWTiBpWChn+IBettZqK6nBSiEmyadqJz585kNJ2onx2j5cZsDh8+vLkLPfiRGpylnp2noSS1KgBs1qog15dVgMxm8ZyDzGbx6jz4xOVg04DmgAEDyG4XwE3C\/CBs8qBKwdJHje8QwGbj13HdnhDYrJv0db0xYzbrKn\/db14JMljAve7Vk1oByn10BMdvV\/pATa2w3Mi5AsCmc4mLewNgs7h1HxxWEVzku7jKNP6TV5uFHKYAGz80nl9UGk7hX\/hdn5zVKhqv\/oNPBGw2fh3zhCiAAiiAAiiAAihQNwWAzbpJz41RAAVQAAVQAAVQoPEVADYbv455QhRAARRAARRAARSomwLAZt2k58YogAIogAIogAIo0PgKAJuNX8c8IQqgAAqgAAqgAArUTQFgs27Sc2MUQAEUQAEUQAEUaHwFgM3Gr2OeEAVQAAVQAAVQAAXqpgCwWTfpuTEKoAAKoAAKoAAKNL4CwGbj1zFPiAIogAIogAIogAJ1UwDYrJv03BgFUAAFUAAFUAAFGl8BYLPx65gnRIFMKGC2qLvkkkvklltukbZt25aUq157JNfrvkkqZdeuXXLrrbfKAw88IAcPHpQVK1bIOeecc4QpPe8HP\/iBDBs2TDp06JDkVlyDAiiAAtYUADatSYkhFECBSgoY2GzdurXMnj1bFDr9R72gr173TeIta9euFd1H\/JprrpE+ffpI3759pVOnTkeYWrdunei5y5cvl44dOya5FdegAAqggDUFgE1rUmIIBVAgCmzqOSeffLLccccd0qtXr+ZL6gV99bpvEm+57bbbRKG9GkQCm0nU5RoUQAFXCgCbrpTFLgqgQIkCJrN5ww03yMqVK+XjH\/+43HTTTXL00Ud754VBn14zadIk7\/zevXs32wv+fs+ePTJmzBi57LLLvO75+fPni3Yl\/93f\/Z3o\/Vq1aiUKatr9rJnAiRMnypAhQ0SzrOa+77zzjvzDP\/yDLFq0SJ5\/\/nkZNGiQd97pp58uLVq0aL73X\/\/6V1m9erXXhf2nP\/1JTj31VBk7dqx85jOf8ezpYco3depUD6pffPFFmTx5snzhC18I9Qq993\/\/93\/Ld77zHXnyySe9Z1B7V111lXTv3l3M823cuLH5+sGDB3sZ4uBwBH3OZcuWNZ83btw40XIcOHBAHnzwQU\/LTZs2ef\/WzOgVV1zRrIVeZED161\/\/umfnP\/\/zP+WEE06Q8ePHi97T1Jd24z\/88MOycOFCz56WQ+tU6+vEE0\/E+1EABVCgWQFgE2dAARRIRQEDm2vWrJEtW7bIjTfeKN\/+9rc9wLMFm6+++qq8733vkxEjRsi+ffvku9\/9rvTo0UPeeustOfvss+UjH\/mI\/OQnP\/F+7rzzTg+ODGz+7Gc\/865VcOzWrZtoORXuFLgUPPVQgFWYUnhUSFPbv\/71rz2A+9KXvuTBoQKnPquWQe0ppHXu3FlOOeUUr+s7eCi06fhKheLPfvaz3k9TU5N873vfk5YtW8rSpUulZ8+e8rvf\/c6D3P\/3\/\/6fVwYto4KuAVxj95lnnpEf\/vCH8vOf\/9w77wMf+IB3X71Wx3uOGjVKPvaxj4lqpc+ocOuvB4VNPe\/YY4+V8847Ty688ELvGVetWiVf+cpXvB+956OPPur9\/z\/+4z96Oqo9Bev3vOc9nmZdunRJxa+4CQqgQPYVADazX0eUEAUaQgE\/bJ522mkyffp0UTAy3ek2MpsKQYsXL24GHc0+zpw50xvnePnll3sZSgU5zYIqRGnm0tzXdE9r2fTYu3ev93fN2JkMogKbwpjClIKmOX7xi1948KyA+KEPfciDzeHDh3sAOXr06JLMaLAyX3jhBa88f\/\/3fy\/XXXddMzxu3bpVNCupkKyZSX22pN3oCoLTpk3zssMGiLUcO3bsKLmHyWxq3eh55lwFYn12zbx+\/\/vf955Ry6JlVG3e+973eo+lcK7P8M1vfrMZ0BvCeXkIFECBmhQANmuSj4tRAAWiKuCHTc0UKmgqZJnudO1KViDSw8Bd3G70j370ox7Emm5vBUPtFtfMo8kqaje43kdhUSHOwKZ2D+ssedNNrOXQSTaaWVTAOv74473rXnvtNS9rqV3z5ti2bZvMmzfPyyQOHTq0GTY1c2iyouV0uv\/++0W7rLWMBnT1XAW8JUuWeF3fOkaza9euiWGz3L3Ns7\/\/\/e\/3YFmBVjVTYA8OXdi8ebOXFVUA\/9znPufBpw5X0OEBF110EbPeo74InIcCBVQA2CxgpfPIKFAPBYKwqTB1zz33NHenf+ITn6gZNhXsFCDNEQZOBrC0G9oPm+bffm20zFdeeaUHVtqVrXDsHzcZ1FG7lRXGgs9aSW8to3ZRG6D0nxssf9LMprGp4zRffvll+eMf\/+iNs\/yv\/\/ovrxtdx4cawC83uciMGzUa65CCb3zjG944WD10\/KeO6dSxsDrOlAMFUAAFjALAJr6AAiiQigJhAKbjKk13uk400TGWeiTNbLqATe1K1iyfwqjCZvAeYeJlDTYV7HWc6s033+xNajruuOO8FQHOOussD4x1fGVU2NR1PRWo9VC727dvl5\/+9KeiGVozUej2228PXf8zFUfjJiiAAplTANjMXJVQIBRoTAXKAZjpTtfJO\/v375djjjmmBDZNZlH\/bo57773X68I1Xb3BrJs5L05mU8dmKowdddRRzffRbnQdh6n30cXRdTyiHgsWLJB27dqVrag4sFmtG11ng2sZdJJR0symGReqQxauvfZaad++vVd2hX19Jv+4VNVMJ0\/pM+ssdHOYbnT9ONBu87DjpZde8ux98IMfPELLxvRqngoFUCCKAsBmFJU4BwVQoGYFygGY6U5XiNHDv6TPb3\/7W298pM6O1t8bQNJz9W82YVO7k7UrW2eN67Fz5065+uqrvWV8zFhO7U6fO3euN5ZSwc0c2h2tXfJazgsuuCBWN3q1CUIDBgyQGTNmeBCcFDaN9lp+\/45DOiRAZ99rttaf2VR9\/ZOqdDytTg5SyFeNFHz1Wd9++22vK92M3+4LUAAAAf1JREFUc9WJSAqburxUENxrdiAMoAAK5FYBYDO3VUfBUSBfClTK9pnudM3y+WHzlVde8XbL+cMf\/uAtLaQQ82\/\/9m\/yxhtveGtPate7zrC2kdlU2NTlfv75n\/\/Zm12t4ygVnnSCkJlcZABUQVch7cwzz5SnnnrKW0JIlwlS8NKMZ5zMZpSlj8z9o8Km6qiTmXQIgEKxwqCWV5eAUoDWXYW0jLrk0ptvvinnn3\/+EbCpSxjphCDtatdllH70ox95AKlrmeoELP23zrb\/9Kc\/7c2k10Pvq2tv+peLypeXUloUQAEXCgCbLlTFJgqgwBEKVAMwHe9nxkT6FyvXmd76b10HU2FQlxLS7JwuCD9nzhxrsKmzvT\/84Q97XeTaHawAptlK7RL2HwrAuqSSdrHr+EedDKPLHGkG1iwBVO1Zg+KELequ61fq0kc6VtQcUWFTJ+8oGP7Hf\/yHB++ahXz22We9rKyWzSzArhOaHnroIe\/HrI1phh4oSN59993e+Zrt1Yylgquu\/amHllnta5e71p0e5RbC53VAARQotgLAZrHrn6dHARRAgRIFyi19hEwogAIokFQBYDOpclyHAiiAAg2oALDZgJXKI6FAnRUANutcAdweBVAABbKkALCZpdqgLCjQGAoAm41RjzwFCqAAClhRANi0IiNGUAAFfAr8fy0UF\/yDldCsAAAAAElFTkSuQmCC","height":0,"width":0}}
%---
%[output:1cdea7d8]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAApsAAAGSCAYAAACsdQoxAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQv0FlW5\/x+QawEeQJJbgAd\/aB3zpwZB5qU0Wx050DJPglYgoBB\/uXgQBEypgAC5FYJxfshFOIXgSU3IatWpY+KiSNCDWSmoAXILUFNALqL817Npv847zLx7PzPvZc9+v7PWb6Hv7L1nz\/d59p7PPPsy9U6ePHmScEABKAAFoAAUgAJQAApAgRIoUA+wWQJVUSQUgAJQAApAASgABaCAUgCwCUeAAlAACkABKAAFoAAUKJkCgM2SSYuCoQAUgAJQAApAASgABQCb8AEoAAWgABSAAlAACkCBkikA2CyZtCgYCkABKAAFoAAUgAJQALAJH4ACUAAKQAEoAAWgABQomQKAzZJJi4KhABSAAlAACkABKAAFAJvwASgABaAAFIACUAAKQIGSKQDYLJm0KDjrCmzdupUGDRpEu3fvjryVPn360IwZM6hp06Z559944w0aMmQIbd68Oe\/3lStXUq9evfJ+u\/fee6murk79Fj4fLGf69OnUr18\/le73v\/893XTTTQXljbpWOEPw\/mzSJ7HnkSNHaMKECbR27VoK6xW8fvv27WnZsmVUU1OTu4zWpra2lpYsWUJbtmxR9x2VNknddJ44ndOUmaW8Wuc4f9b3EvTVQvcX9NVS6qDbQbH9odR1Zh3Zn1u1apXXnrN0H6XUCGX7qQBg00+74q6KoIAJNvkSGoTCD464y4cfxMEHePhhnwY2+fqmh345YJPrEYbGOK2C9Q1C6rBhw2j8+PE5yC72QxmweeqFB7BZhE6jQBGrV6+miRMnxvYZxfbr0t4NSocCMgUAmzK9kLqKFLCBzSDUBQEpTqbwAyUcLQoCV1rYDINwuE7lgk39kDXdexB2gnUzQXNalwRsAjbT+pBNfsCmjUpI46sCgE1fLYv7Sq1AIRgLAoqOvEX9pisRHPoOwlMYNoOAaILNuEhIHNyZYPO3v\/1tbkg\/WI9COtgAa1SaIJhffvnltG7duryIT9QQqem3+++\/n5YuXaqG7KOizvr+g5rzfc6ePZvGjh2rpj2EwTbqBULbm8uLisDy73GwbDv0G\/WiU2gaAk+DCNovzjfCUzCC+UyRzaD\/xN13lMb6t+BUjbBvb9u2Led7thE+Wy2D9S70chdlzyuvvDJvykrUi0\/YVnyf27dvz0UxFyxYQDNnzsz5pa6P1pv9Tk8PsfXh1J0bCoACZVYAsFlmwXG57CiQBjZtH5hR8+Ci4DVqzmYxYTPKKsHydT2DoMV54qI1wfKiIFxry+nuuusumjZtmpobq4EkqtxCsBnnVUFotok8B3UuFNkOlqvrGoQ1\/RvXK\/i7zfzIQteNi\/6a7Be0VZxWxYLNQnM7tX3j5jUH62aaRyyBzUK2j3t5iNNJMn+a\/cQWNothl+z0rqhptSkA2Kw2i+N+rRWQDqMXeqDHDQUH4eOiiy6iKVOm5BbAtG7dOrfQSPKAC0dOwguY9Pm4BTrB38PRlzh4C0NoWOQwZK1ZsyYy8sP32bdv38hFRSbYjII6roeGlqjochhCtM7B3+OivPqeg+CsFzmFo6e8IITtoBdLFZoaEAX2UdHqoJ2i4JfvXV8nCHc2UGxqJKaILufXWphGAYI2iqtnVH0ksBmlX1T+sD9o34lqE1wnbc+g\/kE\/i7JLoXneUbaxfXE12QznoUAlFQBsVlJ9XNtpBWxgMyoaVCiyE4ayIITdeeedNGLECDWcy+UG\/18KmzYPqELzIsORRTaUXmEf9QA2RaHC5T3wwANq2FTrEaeDKaIbfLAH6xAVlY5bqBSlQ6GodvhewhB5zTXXnLYbAdeNXx54d4MgiJkaQDgCGLRrnP2ipl\/E6VRot4BCdTMNo+u8wQgv\/yaZcmLyYVvYjKtr8Peol4xw2w6\/MO3cuTO3W0Xcwj8pbJp82OQvOA8FXFUAsOmqZVCviitggk1TNC\/8oNU3FDVnMxxB5LRjxoyhX\/\/616fNJSy09ZFpUVBQ1EJAFfUgD0fcooaP44wWrDMPKzJsMlSHo45cf155znMog8PqXK5pzmZw66RCsGmz6r8QyJi0+fKXv6wgpE2bNtSpUyc1V49t3rlzZzU3zzRcXWiIOQ42C82H5C2zCt2PzdB+2K6FYDPO7+NgMwxqNvOA4\/whyv9shux1Wy50X4VgM\/yyFTUNRLpAyFaHineUqAAUsFAAsGkhEpJUpwLF7OzjhgfDDzBWWg\/NBVWXRPhsgVMKmxpYwvPQbFaLB+9fLwgKglN4qJWjnnHDjcF8cRBVbtgManP11VfT3LlzVRRPLzBhwGzevLmC60IvKeFhXJ3WNIzuCmwGXyq0nYLTQUzzkdnnbdudbWQTsFmd\/Tfu2i0FAJtu2QO1cUgB24eernIwohOOdMTNAYyKKkVFVE2wGYz08H+bomfhh3oYGKOiMMF74Ajdk08+qW49vBl7lAmjFmgE62g6X8zIZhhi0w6j876hUUDDPtCtW7fIIfXw5v5as2A5QR8qFmzydYLlFnsYPSrabZqzGYZvW4i0TWc75M\/aJI1sFmMYPTxtQNr\/ONR1oipQ4DQFAJtwCigQo4C0szcNu+vLBB+ucUOYhbZoKfSQDQKvKeJos0AobvW5vhcbqI2Ccf4tDnB1+vD5tMPowahb1LBpsE6SBUK6vkGbaXDo2LFjXqTaFHU2gVnSYfRguXGLiSS2jIOyKCgO+mRUZDMIwEkWCBXqwPQ96QVpJv0ksMnXLeYCIcAmHkU+KwDY9Nm6uLdUCkhhky9WaL4an4\/b2LzQPMIwmBWCzeDD0rTAwgTHUfnDeUxAGzRA1B6PwQhfuOxwdDgtbBZz66MoMAveX9yKb9M8X9ar0AIzEyxx\/rj9WU2+WQzYNPlUHGxGNVTTorNCc5fDL0Ns+6hPyHK6uAh7ocV8+jO1cXUotHdsuG7BfTZN845TdWjIDAUqqABgs4Li49JuK5AENsMP++AdRoFGocUZcVFK0\/Bh1DYtUdsfhe8vblP34D1IYDZs3bitenS6ICRFgW5a2IyLQPKm7myHJJu6B+8xeH9BW8etBI\/z\/jAUx+1MEOefhb6IFIYjflnQG6oXAzb5nqJeKoKbnPM2UHxo+OM68MGfcox6IYvTSQKb2v\/DIB9uk5LIpi4zaoTg2WefPe3TlGG76gjzli1bcpu6AzbdfiagdskVAGwm1w45oUDVKZB0jl\/VCYUbLqiA758ItfnYAVwEClSTAlUNm+FJ\/VHzqeKGhSRRgGpyKNyr3woEo0mSIXS\/VcHdSRXwATbj5sFKFiRJdUN6KJBVBaoWNnVHwXPGeF8\/PniIhffECw5l8MOV9\/yzWXGbVSdAvaGASYHwkKVpoYupPJyvbgV8gE39zOBtuuIO07zT6vYC3H01KVC1sMnDHKtWrSKeP8Rbl\/ChO8D+\/fsTb4TMB6dbv3496Qnh1eQcuFcooBWI2j+xpqYGAkGBRAr4Apv6GaHnm2oxTIvzEomGTFAgwwpULWxG2SwKNjnayYeOfmbY1qg6FIACUAAKQAEoAAXKrgBg8x+S63k2O3bsyEU7NXy2aNGC1q1blzOOzfYlZbckLggFoAAUgAJQAApAAQcVAGyG9rULzrHRi4N4MZCObGoA5W8eFxpa37lzJ\/EfDigABaAAFIACUAAK8Ece+K8aD8BmwOpRcBnlFHr+Wtzkb4bMcePG0YYNG6rRp3DPUAAKQAEoAAWgQEiBnj170qxZs6oSOAGbIWfgBUHz588vuPpcQ+nIkSNzC4mCxWgYZafq0KEDGlyZFGC4nzdvnmrM0L1MohfhMrBbEUSsQBGwWwVEL8IlYbciiJigCK17te5QANgsIWxWq1MlaIdFyWKKOBflIiik6ArAbkWXtCwFwm5lkbnoF4Hdii6pVYHVrnvVwiavMt+9e\/dp8y75d3YK3hJJf0Ys6hvNhfberHansmp5JUgE3UsgahmKhN3KIHIJLgG7lUDUMhQJu5VB5IhLVLvuVQubUUPh+hNj+ssoUSvUbeZ1VrtTVaYpf\/BNZkSUK2WBZNdFe0mmW6VzwW6VtkCy68NuyXRLm6vada9a2GTHCX+uMm4jXo52Br8SYfpMX7U7VdpGmTT\/tm3baPny5TRw4EDq0qVL0mKQr8wKwG5lFrxIl4PdiiRkmYuB3cos+D8uV+1cUNWwWSqXq3anKpWupnKPHj1Ke\/bsoXbt2lGTJk1MyXHeEQVgN0cMIawG7CYUzJHksFtlDFHtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVbtTlUBSqyLRiVrJ5Fwi2M05k1hVCHazksm5RLBbZUxS7VwA2CyB31W7U5VAUqsi0YlayeRcItjNOZNYVQh2s5LJuUSwW2VMUu1cANgsgd9Vu1OVQFKrItGJWsnkXCLYzTmTWFUIdrOSyblEsFtlTFLtXADYLIHfVYNT7dy5k\/gv6dGxY0fiv2Ie6ETNasJuZo1cTAG7uWiVbNYJ\/WRl7FYNXFBIWcBmCfzOd6fiB9+4ceNow4YNidXr2bMnzZo1KwecxXiYnnXWWbRnzx5q164dHThwoKgwXIz6FRuupeLDblLFSPkn7Ha6bqwJ2pvcn1zIAdisjBV85wKTqoBNk0IJzvvuVPr+3rlkML3\/obPECjU48CI1eXENrVy5knr16qWgcGTvT9Hmg03EZekMDK9TpkyhM844g9577z265557igbDpYC0xDeaImMp7FaMlw7YrbBRYTe504dfZuUl+JujlLDpw0t5qSzvOxeYdKtq2HzjjTdoyJAhtHnzZqVTbW0tLVmyhFq1apWnm3YS\/eP06dOpX79+sdr67lT6\/g5ddiedOOs8k4+ddr7BgZeo2dMzc7CpyxvXZT+1bXRCXB5D6oo9LWnp0qXUuXNn2r59Ow0ePJi+ddNJap9vSquyN71MtOgX9U6r3xXHRlHzkx+xKiOYaE\/9F+jZRqty5YkLKFIG2E0mJOwWrRfam8yPipm6GDBXqoi0Ly\/lxbRXsCzfucCkW9XCpgZNjqyNHz9e6XTvvffS2rVradmyZVRTU6N+YwcZO3Zs7retW7fSoEGDaOTIkbHA6btTlQpa5nTbQ7XNj5p89rTz\/PC7Y0u702CzbsRJ+uS54uKIYXPYgtNhs\/fRqdTuvQvEBe454wV6osnd3sIm7CZ2CVEGtDeRXORKe5PV2py6WDBXzJGEHhd2pDkLVqqpJtpPs\/5SbrZEshS+c4FJlaqFzdWrV9OqVavyIpkaQPv3769AMgpIWdCovNX0BoOHn6lZ5Z935eEHu8Fu\/HKt\/QAvCTJ\/SJK6GJFIPWc4KyNAWX8pT2JnmzyAzZMnT9oIVQ1pwrCpo5izZ89Wcwv1Efe7Pu+7UwFaZK0BsBmtFyLSdn6E9mank07lSnsrRiSSpymtf\/SBvMghXhJk\/uBKat+5wKRz1UY2w8IcOXKEJkyYQDt27MhFO8ND6GHYjBtK992p8PAzNStENm0UAmzaqHRqKs9NN91ExZ4jDWix0z9pqmIv7EJEOqkl3MjnOxeYVAZs\/mOuZl1dndJKr5Dm\/46DzXAENCyydqrRo0cTr4oMHm3btiX+y\/KxceNGGjBgQOqHHy\/o4YixLi\/tw2\/hwoVqgRBHFIYOHUpp52yG65d2eEiXVynbw24y5XWEDHbL102\/JKC9FfYntLdstjdZraNT7927l\/gveOzatUttGRhkjGJcKytlADYDltLD43369FGLhtLCZpQTDBw4UIFalg9evT9mzJjUsDl37ly1A4AuLy1s8i4B7du3p927d9PEiRNTw2a4fmlhU5dXKdvDbjLlNWzCbtGwifZW2J\/Q3rLZ3mS1jk69YsUKWr58eeRJwGYxFPagDF78M3\/+fLX6\/PXXX89bia5vz7QiXUc2edPyDh06eBfZ5PvjrYXSDutxg+zevbuCei4vLWwuWrRIzW3irY+GDx+eGjbD9UsLm7q8SjUT2E2mvIZN2C0aNtHeCvsT2ls225us1vaRTf4Iyrx58xDZLIbAPpQRhE2+H97mCAuE8i2LOWQyT3dlwQLsBrthNbrMB9KkRnuTqedKPymrtX1qzNms0tXovKcmD7fOmDGDmjZtmvMY\/p2dgjd354M3fddbIelE2PoICxbsuxhyZt8\/PPwkVoPd4tTCwi47P0J7s9NJpwJsyvTKWuqqnbMZNRTOEMlz\/YJfCNK\/6XkWpiF0dgDf32DQicqauSudKOwGuyGyKfOBNKnR3mTqudJPymptn9p3LjApUbWwycKEP1fJi0uCXw\/S4mkn0f+Pz1UismlqWMHzrnSiePhJrIbIJiKbMn8Jp0Z7k+nnSj8pq7V9asBmlQ6j27uIPKXvToVOVOYTrnSisBvshsimzAfSpEZ7k6nnSj8pq7V9at+5wKREVUc2TeIkPe+7U6ETlXmGK50o7Aa7ATZlPpAmNdqbTD1X+klZre1T+84FJiUAmyaFEpz33anQicqcwpVOFHaD3QCbMh9IkxrtTaaeK\/2krNb2qX3nApMSgE2TQgnO++5U6ERlTuFKJwq7wW6ATZkPpEmN9iZTz5V+UlZr+9S+c4FJCcCmSaEE5313KnSiMqdwpROF3WA3wKbMB9KkRnuTqedKPymrtX1q37nApARg06RQgvO+OxU6UZlTuNKJwm6wG2BT5gNpUqO9ydRzpZ+U1do+te9cYFICsGlSKMF5350KnajMKVzpRGE32A2wKfOBNKnR3mTqudJPymptn9p3LjApAdg0KZTgvO9OhU5U5hSudKKwG+wG2JT5QJrUaG8y9VzpJ2W1tk\/tOxeYlABsmhRKcN53p0InKnMKVzpR2A12A2zKfCBNarQ3mXqu9JOyWtun9p0LTEoANk0KJTjvu1OhE5U5hSudKOwGuwE2ZT6QJjXam0w9V\/pJWa3tU\/vOBSYlAJsmhRKc992p0InKnMKVThR2g90AmzIfSJMa7U2mniv9pKzW9ql95wKTEoBNk0IJzvvuVOhEZU7hSicKu8FugE2ZD6RJjfYmU8+VflJWa\/vUvnOBSQnApkmhBOd9dyp0ojKncKUThd1gN8CmzAfSpEZ7k6nnSj8pq7V9at+5wKQEYNOkUILzvjsVOlGZU7jSicJusBtgU+YDaVKjvcnUc6WflNXaPrXvXGBSArBpUijBed+dCp2ozClc6URhN9gNsCnzgTSp0d5k6rnST8pqbZ\/ady4wKQHYNCmU4LzvToVOVOYUrnSisBvsBtiU+UCa1GhvMvVc6SdltbZP7TsXmJQAbJoUSnDed6dCJypzClc6UdgNdgNsynwgTWq0N5l6rvSTslrbp\/adC0xKADZNCiU477tToROVOYUrnSjsBrsBNmU+kCY12ptMPVf6SVmt7VP7zgUmJQCbJoUSnPfdqdCJypzClU4UdoPdAJsyH0iTGu1Npp4r\/aSs1vapfecCkxKATZNCCc777lToRGVO4UonCrvBboBNmQ+kSY32JlPPlX5SVmv71L5zgUkJwKZJoQTnfXcqdKIyp3ClE4XdYDfApswH0qRGe5Op50o\/Kau1fWrfucCkBGDTpFCC8747FTpRmVO40onCbrAbYFPmA2lSo73J1HOln5TV2j6171xgUgKwaVIowXnfnQqdqMwpXOlEYTfYDbAp84E0qdHeZOq50k\/Kam2f2ncuMCkB2DQplOC8706FTlTmFK50orAb7AbYlPlAmtRobzL1XOknZbW2T+07F5iUAGyaFEpw3nenQicqcwpXOlHYDXYDbMp8IE1qtDeZeq70k7Ja26f2nQtMSgA2TQolOO+7U6ETlTmFK50o7Aa7ATZlPpAmNdqbTD1X+klZre1T+84FJiUAmyaFEpz33anQicqcwpVOFHaD3QCbMh9IkxrtTaaeK\/2krNb2qX3nApMSgE2TQgnO++5U6ERlTuFKJwq7wW6ATZkPpEmN9iZTz5V+UlZr+9S+c4FJiczC5pEjR4j\/mjZtqv5cOnx3KnSiMm9zpROF3WA3wKbMB9KkRnuTqedKPymrtX1q37nApERmYPPgwYP0q1\/9in7yk5\/Qhg0b6N13383dW8OGDalnz57Ur18\/uuKKK6h58+am+y7ped+dCp2ozH1c6URhN9gNsCnzgTSp0d5k6rnST8pqbZ\/ady4wKeE8bB4+fJiWLVtG8+fPV\/dy0UUX0SWXXEKtWrWirl270iuvvEJvvPGGAtAXXnhBRTlvvfVWGjhwYMWg03enQidqalb5513pRGE32A2wKfOBNKnR3mTqudJPymptn9p3LjAp4Sxsnjx5kp566imaPHmygsrBgwcryGzUqFHsPfGw+saNG+nBBx+kl156ib773e\/SlVdeadKg6Od9dyp0ojKXcaUThd1gN8CmzAfSpEZ7k6nnSj8pq7V9at+5wKSEs7DJ0UqOZvbv35+6detG9erVM91L7jyD6pYtW2jVqlX0rW99yzpfsRL67lToRGWe4konCrvBboBNmQ+kSY32JlPPlX5SVmv71L5zgUkJZ2HTVHGXz\/vuVOhEZd7nSicKu8FugE2ZD6RJjfYmU8+VflJWa\/vUvnOBSQnApkmhBOd9dyp0ojKncKUThd1gN8CmzAfSpEZ7k6nnSj8pq7V9at+5wKREZmFz\/\/79tHXrVnV\/NTU11KZNG9O9nnae8w8aNIh2796tzrVv314tRuLy9BFOo3\/v06cPzZgxI3LbJd+dCp2ozNVc6URhN9gNsCnzgTSp0d5k6rnST8pqbZ\/ady4wKeE0bPLcyz\/96U9UV1dHhw4doiFDhtCll16qtj+aOHFibvujBg0a0C233EIjR4603nNTG3769OlqyyQ+Vq9ercpduXIlcafMB6cbO3bsaRBaSFjfnQqdqKlZ5Z93pROF3WA3wKbMB9KkRnuTqedKPymrtX1q37nApITTsMnGGTZsmLqHs88+m\/72t7\/R8OHD6Yc\/\/CF9\/OMfV4uH+OCFQE8++SR9\/\/vfp2uvvdZ0z+r8vffeqyKawegkr2afMGGCinCOHz8+B6Dr16+PjWJGXcx3p0InauViuUSudKKwG+wG2JT5QJrUaG8y9VzpJ2W1tk\/tOxeYlHAWNo8fP06TJk1S+2gyRDIA8qbuI0aMUBu3z507l1q0aKHujzd8v+OOO9S+mtOmTaPGjRub7jvyvIZNPqkhlKGUDw2fNgX77lToRG284IM0rnSisBvsBtiU+UCa1GhvMvVc6SdltbZP7TsXmJRwFjbffPNNGjp0KHXv3j0Henv27FFD6bx3Zhj+GArZmEuWLFEbvic5eLslLp87ZC5f\/z9D7bp163JFcrS1EHxqpxo9erT6slHwaNu2LfFflg\/ey3TAgAF06LI76cRZ54lvpcGBl6jZ0zNp6dKlSmtd3pxue6i2+VFxeZsPNqE7trSjhQsXUufOnWnnzp3Kd+pGnKRPnisujja9TDRsQb3T6tf76FRq994F4gJ1J6rvV1xAkTLAbjIhYbdovdDe7PwI7c1OJ53KlfYmq3V06r179xL\/BY9du3bRuHHj8qbpFeNaWSnDWdgMgx8LGvWbFroYsMllrF27Njc\/Uy8O4sVAGi51HTp16mRcIBTlBPxlIwa1LB+bN29f4OAMAAAgAElEQVSmMWPGpIZNjk7X1taSLi8tbPL8W46A8\/QInnubFjbD9UsLm7q8StkedpMprx9+sFu+bho20d4K+xPaWzbbm6zW0alXrFhBy5cvjzwZXBNSjGtlpQzA5j8spRcHBRcMxRnRFA7X52fNmkUdOnTIK8aHyCbfH3\/RKW1kkxskR651eWlhc9GiRdSxY0favn27mtubFjbD9UsLm7q8SnUOsJtMeQ2bsFs0bKK9FfYntLdstjdZre0jm\/xJ7Xnz5iGyWQyBi1lGOSObEtDke9QRT179rleyB+\/dBKPF1KkSZWEukkx1V+YiwW6wG+ZsynwgTWq0N5l6rvSTslrbp\/adC0xKOB\/Z5EiVBjpeCDR79my1Ej0MeQyMPFdPOmeTh855ayWbiKYWE7D5e7rppptSRzb1cIJuhGkjmzwnkudscmSTI69pI5vh+qWNbFZ6+AQPP1N3mH\/elYcf7Aa74SVB5gMupgZs8maWDh46ssnzXmwPnv8ngU0NmnEQEOcc\/HuhvTd9dyo8\/Gw98lQ6QEu0XnruH14SCvsT2hvaG2BT5gMupvadC0yaOxvZfP\/99+mtt94iCQvXq1ePzjzzTKpfv77pviM3cA9n0lsh7dixIwexUYuGwvl8dyo8\/IzulZcAsAnYlHlMfmq0N5l6aG9obzKPKU9q37nApKKzsGmqeJrzpqhpOEKqI6D6mqYhd9+dCg8\/mffh4YeHn8xjAJs8TQfTVk5tEYfpRmlajzt5fecCk9JVCZsmUdKe992pAJsyDwFsAjZlHgPYBGwS6f2IAZtpWo87eX3nApPSzsKmKfoYdWPSOZsmcZKe992pAJsyzwBsAjZlHgPYBGwCNtO0GRfz+s4FJs2dhc1Dhw7Rww8\/rOZt6oM\/XcmfrPziF79IXbp0UT8fOHCAfvrTn1KjRo2IN0y\/+eabqVmzZqb7Lul5350KsClzH8AmYFPmMYBNwCZgM02bcTGv71xg0txZ2AxX\/Omnn6YHHniApk6dSh\/96EfzTvNnoPjb6PwZy2984xvEC4UqefjuVIBNmXcBNgGbMo8BbAI2AZtp2oyLeX3nApPmmYBNjnLyVkOf+9znIjdR55vkfTafeOIJWrBgAfG3zCt5+O5UgE2ZdwE2AZsyjwFsAjYBm2najIt5fecCk+aZgE09f\/OWW26h3r17R94Tw+aqVatE+2yaxEl63nenAmzKPAOwCdiUeQxgE7AJ2EzTZlzM6zsXmDTPBGweP36cJk2aRH\/7299o5syZ1KZNm7z72rdvH40aNYq6detGd999t5q\/WcnDd6cCbMq8C7AJ2JR5DGATsAnYTNNmXMzrOxeYNM8EbPJNbNmyhYYPH05Hjx6lvn37Uo8ePdS9PfPMM7RmzRpq0qQJLVy4UAFnpQ\/fnQqwKfMwwCZgU+YxgE3AJmAzTZtxMa\/vXGDSPDOwyTfCX\/KZNWsW\/fKXv6R3331X3VvDhg3puuuuo5EjR1KHDh1M91uW8747FWBT5kaATcCmzGMAm4BNwGaaNuNiXt+5wKR5pmBT38yJEyfo4MGD6n+bN29ODRo0MN1nWc\/77lSATZk7ATYBmzKPAWwCNgGbadqMi3l95wKT5s7C5rFjx9Qemu3bt0+0lRF\/U53zh+d3mgQpxnnfnQqwKfMSwCZgU+YxgE3AJmAzTZtxMa\/vXGDS3FnY5Mjl7bffTm3btlVzNTt27Gi6F3WeIfO1116juro62rt3r1qdXu7Dd6cCbMo8CrAJ2JR5DGATsAnYTNNmXMzrOxeYNHcWNrnihw8fpmXLlilwvOCCC4g7oEsuuYTOPvvsvKFzXq2+e\/duYmM+9thj9H\/\/939qDuegQYPowx\/+sEmDop\/33akAmzKXAWwCNmUeA9gEbAI207QZF\/P6zgUmzZ2GTV35nTt30o9+9CNauXJlbq5m1I3x\/M0BAwbQV7\/6VRURrdThu1MBNmWeBdgEbMo8BrAJ2ARspmkzLub1nQtMmmcCNvVNcATzT3\/6E23cuJF4o\/e\/\/vWvVFNTo+Zl1tbW0sc+9rGK77HJdfXdqQCbpmaVfx6wCdiUeQxgE7AJ2EzTZlzM6zsXmDTPFGyabsaV8747FWBT5mmATcCmzGMAm4BNwGaaNuNiXt+5wKQ5YNOkUILzvjsVYFPmFIBNwKbMYwCbgE3AZpo242Je37nApDlg06RQgvO+OxVgU+YUgE3ApsxjAJuATcBmmjbjYl7fucCkOWDTpFCC8747FWBT5hSATcCmzGMAm4BNwGaaNuNiXt+5wKQ5YNOkUILzvjsVYFPmFIBNwKbMYwCbgE3AZpo242Je37nApDlg06RQgvO+OxVgU+YUgE3ApsxjAJuATcBmmjbjYl7fucCkuTew+f7779OmTZvo\/PPPV99Lr+Thu1MBNmXeBdgEbMo8BrAJ2ARspmkzLub1nQtMmjsPm7yf5k9\/+lNav369Askvf\/nL1KlTp7z72rVrF33nO9+h\/fv3q89TtmrVynTfJT3vu1MBNmXuA9gEbMo8BrAJ2ARspmkzLub1nQtMmjsNm1u2bFHfRefN2\/XBUcv777+fLrvsMjpx4gT95Cc\/oXvvvZfeeustGjhwII0ZM4aaNm1quu+SnvfdqQCbMvcBbAI2ZR4D2ARsAjbTtBkX8\/rOBSbNnYVNBkmGyJ\/\/\/Oc0Y8YM+tSnPkUcwZw6dSo1bNiQpkyZQg888AAtXryYOnfurNL26NGD6tWrZ7rnkp\/33akAmzIXAmwCNmUeA9gEbAI207QZF\/P6zgUmzZ2FzbfffptGjBihAJL\/1RD5wgsv0Lhx4+jiiy+mH\/\/4x\/S1r32Nbr\/9dmrRooXpXst23nenAmzKXAmwCdiUeQxgs9pgs\/47B6j+O6\/nGZ5\/+9CzS2n06NHUs2dPFWzhZ9+Adm9SbfOjYpfae7wBzdrWhm677Tbq2rUrHTp0iCZNmkRDv3iSPnmuuDja\/QbRd1bWO61+lxzvT+3ev0Bc4MF6++ipxvflyhMXUOEMHTt2JP6LO3znApP8zsImz9UcMmQIjRw5kq666qrcfejfX331VTVPs2\/fvlS\/fn3TfZb1vO9OBdiUuRNgE7Ap8xjAZjXBJkPl+dtW0u4tm9O4CfJWWAF+IZg1a1YscPrOBSb5nYfN8ePHU69evU6DTQbQYMTTdKPlPO+7UwE2Zd4E2ARsyjwGsFlNsNngwEvU7OmZClQ6dOiQxlWQt0IKbNiwgebNm0crV67M45VgdXznApP0mYXNcMTTdKPlPO+7UwE2Zd4E2ARsyjwGsFmNsFkIVNL4D\/KWXgGbZ75NmtLXtHJXyCxshiOelZPw9Cv77lSATZm3ATYBmzKPAWwCNol27typ\/rJ8mOYxZvnepFFL37nAZEvnYZMnM19yySW5++AtjkaNGqUmEQd\/5wS8iOjMM8+s+BxO350KsGlqVvnnAZuATZnHADarHTYZMnkxEA\/PZvkwzWPM8r0BNmXWcx42N2+2nzRdW1uLTd1l9k+UGrApkw2wCdiUeQxgs9phU\/ex71wymN7\/0Flp3KdieRsceJGavLim4DxGrpxe9GsarXzllVeoZcuWFf9oS5SgNgEmmzQVM1YZLuwsbPK2DA8\/\/LDarN324KjmDTfcQM2aNbPNUpJ0vjsVYFPmNoBNwKbMYwCbgM3fE2tw6LI76cRZ56Vxn4rl1QufTHNReY\/suro64+IaTufCFwIBm8lcylnYTHY7buQCbBa2Q7gT0nrN6bYn0f5xmw82oTu2tKOlS5eqDf63b99OgwcPproRyfaP2\/Qy0bAF9XKdn65f76NTqd178v3jAJuAzTQ9E17uZOplrb1FQVlam8sUK01qG9jk+2SI5E9Nz549u+BKbsBmaexUrlKrGja3bt1KgwYNot27dyu927dvT8uWLaOampo8\/XXD1z9Onz6d+vXrF2sjwCZgM6hA1h5+cdbDS0K5umVENhHZ9D+yyc\/fO++8k3j4nEEybhg9\/PzVkdLw79xq9LNZD83zPtwcDdXP+GCUVafRU\/XSTMOzeebbpKlMD1OeqzoLm8eOHaPnn3+e+F\/bo3HjxnThhRcS\/2s6tOGD4Lh69WqaOHFiXjif040dOzYHoRpQeeulOOD03anSvnUDWkzeWZrzsJtMV7wkICIt85hkLwnVGNk8cuQITZgwgS699FK65ppr1AdcCs3Z1BFQPYyuQXXmzJm54BA\/v+fPn6+e1a1bt1ZlcsRUB5CCz3JeJa+vr5\/jDLx8cD2kh80z3yaN9LpZSu8sbIbfOmxElbyZsGPx2w5\/d71p06aqeN0AOMLJDqfrwJvKBx2QnXrVqlWx80d8dypAi403fpAG0AJokXlMMmhBRPqUAllrb9UIm\/wMXb9+vXr+8nNXCptRvh4EUA2b\/fv3zwsKaaC89dZb1TXD55O2U5tnvk2apNfPQj5nYdM2svnnP\/+Zvve976kI6M0336y2i9DwKDWAhk3Ox42At5\/gYfbwXBId3YybY+K7UwE2ZZ6VtYcfoCWb0AK7ZdNu1Qab4aikzWr0cGRT+3o4KKWnwmnYDEdLg5C7Zs0aNZLJhyRQFdXObJ75NmlkT5ZspXYWNk0yMlwuX76c5syZQ23atKHJkyfTZz\/72VR7bIYjmeEhdF0n01C6direC5T3GQsebdu2Jf7L8rFx40YaMGBA4pWSunPlBT0cNdblpV0gtHDhQrVAiF8Shg4dmnqBULh+aRcI6fIqZXvYTaa8fkmA3fJ10wvy0N4K+5Nte6s22NTT1aLUGzZsWOQwdtQwul5voafCRUU2C8GmDkoFgTUpdOpnvu4r9u7dS\/wXPHbt2qWCYabV+bJeKjupMwmbW7ZsoXvuuYeeeeYZuvHGG+n2229XwJn24BD72rVrI+d4BBcNaeeMC8FHTVzWdRs4cKACtSwfPKF6zJgxqWFz7ty56o1Sl5cWNrnT4Tdbnh7Bb6xpV6OH65cWNnV5lbI97CZTXsMm7BYNm2hvhf3Jtr1VG2yGVUsS2QxGKDU0BoNDpmH0qHmZNvWIs7h+5uu+YsWKFSoYFnUANmX9cEVSlyKaqW9Ev20FFwzFRTZtYXPWrFnUoUMH7yKbrAtvLZR0DzjduXKD7N69O+ny0sLmokWLiCd+89ZHw4cPTw2b4fqlhU1dXkUaD1FOZ9jNzgIaNmG3aNhEeyvsR7b9JGDzDeOczfDQe3jdhB5tZIvYLBCKgtG4oXqb3kLDpu4roiKb\/DWoefPmIbJpI2gl0\/zxj3+kqVOn0nPPPaeimSNGjChKNJPvKQo0+fe0w+i+vsFgzqasJWDOZrRe2B\/Vzo\/Q3ux00qmy1t4Am2bY1OspeORRB4T0ZvBsdx7R4jUU\/BuPOOoV7sGtj8JbG9pufWjjfTbzMW3S2Fwrq2mcH0ZnJ+PtDnhLA3aWSZMm0ZVXXplqbmbQWNpho\/bOjFsIhAVC6faAw9ZHlekuAC0y3bMGLXF3h\/Yms3uxUtu2t0KwefT8vnTirPOLVaWyllP\/nQP0oWeXViSSl2ZIPIlINiBpkybJtbOSx2nYLGU0kw1k+kxW3HA5tj4CbEoaOKAFkU2Jv4TT2kILYPOUAllrb1GwyYsceTEJD71m+eAFsjydjKc3lfMAbJZTbbtrOQubwRViDRs2pHPPPZcaNWpU8K7YoXlVesuWLY13H7WBe1SmcDrTSnQuw\/c3GDz8jO6VlyBrDz9ASzahBXbLpt3iPuvIwMl\/WT74mVxu0GS9AJvueY2zsPnmm2+qIXNJY7OFTdOG8eHtD8Kry\/G5SkQ2JU0ZsInIpsRfENk81b+kXZBX6Tnzti\/lNt8QT+M\/yFt6BWwCTDZpSl\/Tyl3BWdisnCTpr+y7U9l2ooi0ZDPSArvBboV6QSzssntG2PaTgE07PV1OZfPMt0nj8j2mrRtgM62CEfl9dyrbThTQAmgBtKTvYNDeZBpmbSQBw+gy+7qY2uaZb5PGxXsrVp0yC5u85yZ\/qvLw4cN0zjnnqJXq9erVK5Yuqcrx3anw8JO5R9YefnhJwEsCXhJkbTwqtW0\/GbdA6D9uvoE2vZr\/FZr0tSpvCZ\/857b0vQcfrsi8zXLeqc0z3yZNOetc7ms5D5uvvPKKWjV+9OhRWrBgAbVo0YL27dtHY8eOpaefflrp1aBBA7Vyjz9fxf9d6cN3p7LtRAEtgBZAS\/reCO1NpmHWXu4KbX00rst+atvohEwAR1LzdIsVe1rGbn0U3CfT5jORzAK8+LdVq1aO3OEH1bB55tukce7Gilghp2GTnYu\/cX3gwAH6+te\/Tt\/4xjeoSZMmahd+3ql\/2rRp1KNHD1q8eDE9+OCD9MADD6g9OCt9+O5UePjJPCxrDz+8JOAlAS8JsjZe7Mim7mOTflUtfe3Tl6Dn9kYt1ApvH8jgyZ8ZnjFjBunPTwZrwHpwGt5zG7CZ3jaVKMFZ2Dx58qSKZLJTsoOdd955Sh\/+DNSwYcPo85\/\/vPqKEA+d88r12267jTp16qS2PjJtkVRqoQGbhRXGJtOl9sDo8vGSINMdLwnRemGBkJ0f2ba3QpFNH2FTfw3o0ksvpX79+ikxTVsVATbtfM7lVM7C5qFDh2jMmDHUrl07uueee3LD4\/y5So528nd5L7744py2\/NazceNG9bvNPpulNApgE7AZVADQAmhJ09\/YQgsi0tmMSAM2C8Om9n\/t3zpSGv6dz+ttCTW8Bj9XyeeDUdbwFog2Q\/lxbczmmW+TJk0\/4XpeZ2FTO0KvXr1o\/PjxOR1XrVpFDz30kBo6b9OmTR5ssjFdCLP77lR4+MmaNWATsCnzmPzUaG8y9bLW3qoNNtma4WF0m6\/yBYfR+eMqd955J82cOZNqamqUg3AZ\/FnrZcuWUevWrWnIkCG0f\/9+9f+chtsRr\/Xg\/+c9uSdMmEDB6CqXz0eQN2w9z+aZb5PG9npZTOcsbPLQOEcwu3fvnjP+8ePH1UbvfASHy3nInd9oXnzxxdwiokoaw3enwsNP5l1Ze\/ghQpbNCBnslk27VSNsajicOHGiMlqfPn1i52vyeZth9CCAatjs379\/bqiey9FAeeuttyoYDZ+X9ewfpLZ55tukSXr9LORzFjY1WHKEc86cOdS8eXPatWsX3XLLLXTzzTfnORDmbJbX1QCbMr0Bm4hsyjwGkc1q\/4KQzwuE9JxN3q5QRxFNIBYHm+GhcC4zGNnk8nl0VB8c\/Vy\/fr0C2zVr1pCG3TRD6BqG2WcLfbXKdI9p+ogs5HUWNlm83\/zmN2oF+pe\/\/GX6whe+QD\/5yU9U9JLnZXbp0kXpy\/ttsoE5svn973+frr322orr7rtTATZlLgbYBGzKPAawCdg89clOHxcIxYFjoWHscB6OYvJWh7yCXc\/TjIpsFoJNveo9CKxJodPmmW+TJk0\/4Xpep2HzxIkT6i1l1qxZxP\/NoXF2rKuvvjq3Cn348OH0hz\/8gW644Qa6++67qVmzZhXX3HenAmzKXAywCdiUeQxgE7AJ2Ay2gjBsBiOUGhqDczJNw+hR8zJNK+ILtWGbZ75NmjT9hOt5nYZNLR6vTH\/77bfVKvPgHlwcjv\/BD35A3bp1U5HPxo0bO6G3704F2JS5GWATsCnzGMAmYNNf2EwyjB5eEBReUKQjndxybBYIRcGozbzQuHZs88y3SZOmn3A9byZg03URw\/Xz3akAmzKPBGwCNmUeA9gEbPoLm+zdGjjXrl2rnF3PtdQry8PtJZheD5sHv0DE+WfPnq0WAPGin2uuuUYtAApufRS+RnAo3qYOiGym6cWInIdNXmnO8zL++te\/qkVCH\/vYxyq+abtJcsBmYYWwqbvJg0pzHi8JMl3xkoCXBJnHJHtJKLQafUC7N6m2+dE01ahY3r3HG9CsbW0KLpopVeXSDIknqZPNM98mTZJrZyWP07DJw+dTp06lhx9+OKdn586d1d5a\/JlKVw\/fnQrQIvM8QAugReYxyaAl7hp4uUujfvK8tv1kFGzu3LmTxo0bRxs2bEheAQdy9uzZU6254H0ty3kANsuptt21nIZN\/v45w+ZXvvIVNSeTHei+++5T8zZ5RXq5HdhO0lN7gpm2QbAty8V0tp0oHn6nFABsAjbTtGO0N5l6WWtvUbDJd8zAyX9ZPvgZXYnnNGDTPa9xFjYPHz6sdvhnsJwyZUpu8c+mTZvUPpvz5s2jq666yj1F\/7EBLWAz3jSItFTGbQEtMt2zBi14ucvmy10cbMq8FakrqYBNgMkmTSXvodTXdhY24z5XyZ+f4o3db7zxRjUR2MXDd6cCtMi8DtCCyKbMY\/JTo73J1Mtae0NkU2ZfF1PbPPNt0rh4b8WqU+ZgMw5CiyVIMcrx3anw8JN5SdYefoiQZTNCBrtl026YsynrT11MbfPMt0nj4r0Vq06AzWIpGSjHd6cCbMqcBrCJyKbMYxDZxNZHp+b9f+umk9S+VRrvqVzeTS8TLfpFPeNqdN4zc9u2bblPV+oaB7c24t8KfQqSz3Mgij9d3bVr17LftM0z3yZN2StexgsCNksgtu9OBdiUOQ1gE7Ap8xjAJmDzFGzWjThJnzw3jfdULi\/D5rAFhWGTQZO\/Tz5s2LA82GTQ5C0P+RvmvG5D74nJe2kGv3Wu767cC4LCqto8823SVM5apb+y87B50UUX0ciRI3NKvPXWWzRq1Cjq3r173u+coF69enTmmWdS\/fr1S69cgSv47lSATZl7ATYBmzKPAWwCNv2GTb1JOy\/4bdOmjQJI\/QnJOHAs9O10wGaaHqY8eZ2Hzc2bN1srUVtbS0uWLKFWrSo77gDYLGwyrEa3dumiJsRLgkxOvCTgJUHmMcleEgpt6u5rZJP7ooceekhFLnk7Qz6ivlceVDQONjVoalbQUdLwF4K4rGAElcs7ePCg+tNfMgpHWHXkVddDf70IkU15y3AWNo8dO0bPP\/888b+2B38b\/cILL6z4N9IBm4DNoAKAFkCLbR8WlQ4vCTL1stbeqhE2bSAymEYDJe9A069fv9McIhzZjIp0hp\/Lek6oBsjwNTg9p9EBrPD32YOVsHnm26SReXq2UjsLm9mSMfqN1jShOav3iIefzHJZe\/jF3R0i0jK7Fys12ptMyay1N8DmvQUjm3rInRPpOZxhj7AZRg+nCc8L5TKDgPmrX\/2KVq1aZTVaagOSNmlknp6t1IDNEtjLd6fCw0\/mNFl7+AE2TykAuyEiLWvp0UGHQ5fdSSfOOi+2KMBmPGxq0NyxY0dB6IuDTZ1fD5OzEXQQKGpYPhi9bN26NQ0ZMoT08HzcELqGVNOHXHznAlNbAWyaFEpw3nenAmzKnALQAmiReUwyaMFLQjZfEgCb0bBpC5ps9bhhdAZFPQ8zKrLJeYNzReOGyoNzN6Og0+aZb5MmTT\/hel7AZgks5LtTATZlTgPYBGzKPAawidXofq9GD3p4VITRZug8WEYYJMPzLTltePsk0zB61EJjhs7169efNpxv88y3SZOmn3A9L2CzBBby3akAmzKnAWwCNmUeA9gEbFY3bEaBYKE2pOH00ksvVQuI+Bk1duxYWrZsGdXU1OQinxzpDA6j19XVUdwCIQbL4JzN8DWC9bF55tukSdNPuJ4XsFkCC\/nuVIBNmdMANgGbMo8BbAI2qxc2o7Ys0i2iT58+sYuE9FC3TrNmzRq1Ybw+Fi9eTI8\/\/ji1b99eDZ1HbX0UHiIPf8UovDWSLtvmmW+TJk0\/4XrezMHmyZMn1b5Y7733ntrAnTdyP3HiBDVs2NAZrX13KsCmzNUAm4BNmccANgGbp2Bz6Bez+wWh3W8QfWel+XOVadpGmryFNomXlmvzzLdJI71ultJnBjYZMp955hn61re+RS+99BLpDdwbNWqkwuU9evSggQMHUoMGDSquv+9OBdiUuRhgE7Ap8xjAZrXD5s6dO2ncuHG0YcOGNK5T8bw9LuxIcxaspI4dO1a8LuEKADbLa5LMwOZzzz1HN998M1199dV07rnnEu+BxZut8ndT586dSz\/84Q9pzpw5dO2115ZXwYirATYLmwD7NVbGRfGSINMdLwl4SZB5TLKXhKjV6FwSAyf\/ZflgyHQRNFlTwGZ5PSsTsHn8+HGaNGkSvf\/++zRlyhRi8Azu7M\/D6NOmTaPXX389dj6HSda4VWZx80cKzR0BbAI2gwoAWgAtpv6n0Hm8JMjUy1p7i4NN2V0jdSUVsHnm26Sp5D2U+tqZgE29rcEtt9xCvXv3ztvlX29P8MQTTxBPAE7ybXTtBFEAGV7VZmMQ350KDz8bL\/ggTdYefnF3h4i0zO7FSo32JlMya+0NsCmzr4upbZ75NmlcvLdi1SlTsDlgwAC67rrrImGTI5M\/\/vGPadGiRdSyZUtrfYKrzaJgMy7iaROJwOcqo1UCtFi7Z1ETAlpkcmYNWvCScEqBrNkNw+iyduliahuQtEnj4r0Vq06ZgE09jH748GG1J9YLL7yQN4y+b98+GjVqFHXp0oUmT55MvGjI5mDQ5M9Y8V5cjz76KO3evfu0Yfgk8zp8dypAi413IbJpUmnzwSZ0x5Z2tHTpUurcuTNt376dBg8eTHUjkq3A3fQy0bAFH6x+1X7a++hUavfeBabqnHY+a9AC2PQHNn1ZINSzZ0+aNWuWs\/M2xZ1CTAabZ75NmmLVx8VyMgGbLNyWLVto+PDhahU6LxB65JFH1Cr0bdu2qcVBhw4dogcffJAuvvjiRDpHbSKrh+9btGhB69aty5Ubt9eWTuC7UwE2ZS4GaInWC7Bp50dob3Y66VRZa2+FPld5xbFR1PzkR2QCOJJ6T\/0X6NlGq3KbqMdVi0cP+Tke\/Gwkpw2vlyj0bXJOz8\/rN998k7p27Vp2BWye+TZpyl7xMl4wM7DJmvzxj3+kqVOnqi2Qgsd5552nIpq8\/VHSIwo2tbPz8LpuCBpAO3XqFLsYSTvV6NGjid\/sgkfbtm2J\/7J8bNy4kXhKw6HL7qQTZ50nvhXduXJEq1evXqTLm9NtD9U2PyouT0PLwtNO9HEAACAASURBVIULVYSMowJDhw5NHSEL1y9thEyXJ77BImWA3WRCamiB3fJ1Q3uz8yPb9lYINpP2OXY1LG0qG+jXG7GHAzhhMAt\/ajJc8\/DnKkt7Z6eXruur+4q9e\/cS\/wWPXbt2qe2sfJ1eZ9I8U7DJN8P7bf79739Xb0LHjh1TbzGtW7em+vXrm+614HnJ57FMbyj6fNQFeS9QBrUsH\/zJrzFjxqSGTd6yiiPVury0sMlvvvx1CJ4OwV+OSDscG65f0o5fd7q6vErZHnaTKQ+7ReulYRPtrbA\/2ba3aoRN\/enHTZs2UZs2bVTQQQd09Dn9pR+tcqEpba7Apu7jV6xYQcuXL490EMCmrB\/2LrUENvVb1siRI9V3WMOHhk2eq9KhQ4e80z5ENvn+eG5d2sgmN8ju3burBV9cXlrY5MVhvKcbz\/3jKRdpYTNcv7SwqcurVOOB3WTKa9iE3aIjm2hvhf3Jtr1VI2yyNg899JAaHbzvvvuUkOHRQ\/5\/hlDToUGT4Z4PHSWN2rYwGEGN+lxlOMKqI6+6DnFD+fqZr\/uKqMgmb9A\/b948RDZNBq3keZ6P+fDDD9Nbb71lrAZHOT\/96U+riKck2lkK2PT1DQZzyIxumJfAZjhJVmKy1LCbTDfYrXBkEwu7zLDJX0IyvZRXI2wGlQtHLBkS77zzztz3yzVEFpqzGY5sRkU6wyOSeicaXa7O079\/fxVE4vTB\/bx1vWbOnEk1NTV5xjeNdnJimzSyHipbqTMxjM6TfnlTdzYWb9zOxznnnKP+\/etf\/xqp+A033EB33303NWvWzMoiUbAZ5xz8Oy9O4lXsYaerBqcCtFi5VC4RoAXQIvOY\/NRobzL1stbeAJv3KgPryKaOSPLwut432zSaaDOMHk4T98zXgMlfKVy1apXV3t02IGmTRubp2UqdCdhkSXm7o9tvv51uvfVW6tu3r\/pMJR\/89SB2ivnz59P3v\/99tf3RU089Rd\/+9rfVIhHb+ZFRjqfnjuzYseM0pw8uGgqb3HenwsNP1siz9vCLuzvsjyqze7FSo73JlMxaewNsng6bHNkMRxB5SDsO\/uJgUz\/DeYtDfegRx6g5oMHoJY+SDhkyRK0p4KNQZNXmmW+TRubp2UqdCdjUDsNbHo0YMYLq1auXpzIvGlqwYAG99tpr6nOWjRs3Vnv3sXF5XpHNUWgYPbjxu8npENk0qw1oMWtUihSAFpmqWYMWvCScUiBrdgNs2sFmeFg76O9xw+gMinoeZlRkMxhR5f+OGyoPzt2Mgk4bkLRJI+uhspU6E7CpnYQXfXzhC1+IVPiXv\/wl8dY3Ouz+m9\/8RkU7H3vssbJbxHenArTIXCprDz9ASzahBXbLpt0Am\/mwqYNLN954Y94CoUJf8wuDZBSYhrdPMg2j609hB9tVXB1snvk2aWRPlmylzgRs8gIh3mqHI5v8b4MGDfJU5qF03nLg5ZdfVv\/yPE0Ot\/\/sZz8jXh1W7sN3pwJsyjwKsBmtFzZ1t\/MjtDc7nXSqrLU3wGY+bOrRwfDinEGDBlHcDjAaUC+99NLc4p7guorgivXgMHpdXV1ueDy8QCg8bB++RtArbZ75Nmlknp6t1JmATZaUPyf5zW9+k\/7f\/\/t\/xG88PJ+CD14wxFsocBSTh9B5YdCLL76oFhRddNFFKk+5D9+dCg8\/mUdl7eGHCFk2I2SwWzbtBtg8HTY1cPJqfn2YviCkh7p5PQVvqbRmzRq137I+Fi9eTI8\/\/rjai5kXI0VtfRS+RngKXdzXA22e+TZpZE+WbKXODGxy9JI3SeVJw++++26eyg0bNlRbJfCG6bzR+4QJE9RXZDjKqVetl9MsvjsVYFPmTYBNRDZlHpOfGu1Npl7W2lsh2LzkeH9q9\/4FMgEcSX2w3j56qvF9zu4rWWiTeKmENs98mzTS62YpfWZgU4vK2yDx5qh\/+ctf1E\/dunWjT33qU+orBHwwbHK086yzzqJGjRpVxBa+OxUefjK3ytrDDxGybEbIYLds2i0KNjlYwp825Gddlg\/+XDN\/3IQ\/tuHaAdgsr0UyB5vllSfZ1QCbhXXDavRkfpU2F14SZAriJQERaZnHJItIR8Eml8TAyX9ZPhgyXQRN1hSwWV7Pyhxs8mKh48ePn6YSD7Nv2bJFzdO03ci9VFIDNgGbQQUALYCWNH0NXhJk6mWtvcXBpuyukbqSCtg8823SVPIeSn3tzMDmvn376K677iLe0ijuqK2ttdrtv9Si+u5UePjJPChrD7+4u0NEWmb3YqVGe5MpmbX2ptvV6NGjiYedcWRPgV27dqlpD4U+Ue07F5islgnY5Kglh7x5GyPe\/oChkjdr79Gjh\/pcJK9U50VCvJKMV5qFN303iVDs8747FR5+Mo\/J2sMPsHlKAdgNEWlZS89PbdtP1n\/nAJ2\/bSXt3nLqSzU4sqmAaX6q71xgslomYPPtt99WXw5iuOR\/+WCwPOOMM9QWBgcPHqQ77riDrr76arXHVqUP353KthMFtABaCrVF7LNp11OhvdnppFNl8SWBgbP+O6\/n3WiDAy9SkxfXqAU2HTp0UIuF5s2bR+O67Ke2jU7IRCEibm8r9rSkyZMnq6lmr7zyCt1\/\/\/30rZtOUvtW4uJo08tEi35R77T6XXFsFDU\/+RFxgXvqv0DPNlqVK09cQIUzmOan+s4FJvkzAZt6s1Xe0PWqq65S9xTeyZ83cf\/d736n9tfS30033XypzvvuVHj4yTwniw+\/qDvEMLrM7sVKjfYmUxLtLVovvNzJ\/KjYqX3nApNemYBNHdns3bt3LnLJhpszZ44aTm\/ZsqWay8kbu+vPVZpuvJTnfXcqPPxk3oOHHx5+Mo\/JT432JlMP7Q3tTeYx5UntOxeYVMwEbOo5m6+99poaPme45O+c8oRq\/v8LL7yQFixYQE899VQOPk03XsrzvjsVHn4y78HDDw8\/mccANvnLMb2PTqV278k3NEd7Q3tL095Kldd3LjDplgnY5JvgbY2GDx+uNm9fuHAhNWnSRH0paP369WoDd55\/wp+SYgANfzvdJEKxz\/vuVIBNmcfg4YeHn8xjAJuATSJMW0nTatzL6zsXmBTPDGzyjfAGt3\/84x\/VQiD+OtDevXvVKnUeQr\/++uuJ53Ry1LPSh+9OBdiUeRhgE7Ap8xjAJmATsJmmzbiY13cuMGmeCdg8efKkGiLv2rWrs18jCArtu1MBNk3NKv88YBOwKfMYwCZgE7CZps24mNd3LjBpngnY5O+hDx06lP793\/\/dia2NTKL67lSATZMHADZtFMLqWBuViNDe7HTSqfByh5c7mceUJ7XvXGBSMROwGbUa3XRjlTzvu1Ph4SfzLjz88PCTeQwim4hsIrKZps24mNd3LjBpngnY5JvgDW2nTZtGN954I11++eWRe2nyl4POPPNMql+\/vum+S3red6cCbMrcB7AJ2JR5DGATsAnYTNNmXMzrOxeYNM8EbOpN3TdvLvw5L3wb3WTu4pwHbMp0BGwCNmUeA9gEbAI207QZF\/MCNnn1jePHsWPH6Pnnnyf+t9DRuHFjtecm\/1vJw3enAmzKvAuwCdiUeQxgE7AJ2EzTZlzM6zsXmDTPRGTTdBOunffdqQCbMo8DbAI2ZR4D2ARsAjbTtBkX8\/rOBSbNMwebhw4dUl8P4q8KcRTzjDPOoAMHDtBHPvKRis\/V1GL77lSATVOzyj8P2ARsyjwGsAnYBGymaTMu5vWdC0yaZwY2GS5Xr15NkydPpnfffZf0\/Ez+WtCIESOoffv2dPfdd1OzZs1M91zy8747FWBT5kKATcCmzGMAm4BNwGaaNuNiXt+5wKR5ZmCTvxLEn6L8j\/\/4D2rXrp36BvqSJUuoRYsW9Nhjj9HUqVPpjjvuoAEDBpjuueTnfXcqwKbMhQCbgE2ZxwA2AZuAzTRtxsW8vnOBSfNMwOaRI0fUd9B5qHz8+PG0ceNG9ZlKhs1WrVoRr3FasGCB+pTl3LlzKx7d9N2pAJumZoVhdBuFsKm7jUrY1N1OpQ9S4eUOL3dSnylHet+5wKRhJmBTb33E3z6\/6qqr1Bc1grDJN8mRz\/nz5+cA1HTjpTzvu1MBNmXeg4cfHn4yj0FkE5FNRDbTtBkX8\/rOBSbNMwGb+nOV119\/PfXv3z8SNpcuXUpPPvmkinDy0HolD9+dCrAp8y7AJmBT5jGATcAmYDNNm3Exr+9cYNI8E7DJi4M4ksnD5Pfddx+9+uqreZHNv\/zlL2qREEc9eZidFw1V8vDdqQCbMu8CbAI2ZR4D2ARsAjbTtBkX8\/rOBSbNMwGbfBP79u2jUaNG0VtvvUXnnnsu\/eEPf6AvfelL9Morr9C6deuoS5cuVFdXR+ecc47pnkt+3nenAmzKXAiwCdiUeQxgE7AJ2EzTZlzM6zsXmDTPDGzyjezfv58WLlxIjzzyCB08eFDdW8OGDem6664jns\/ZoUMH0\/2W5bzvTgXYlLkRYBOwKfMYwCZgE7CZps24mNd3LjBpninY1DfDq88ZNt977z0688wzndnMXdfPd6cCbJqaVf55wCZgU+YxgE3AJmAzTZtxMa\/vXGDSPBOwyQuEeC\/Nz33uc9S5c2fn4DIssu9OBdg0NSvApo1C2PrIRiVsfWSn0gep8HKHlzupz5Qjve9cYNIwM7A5dOhQ2rRpk5qT2a9fP+rbty+dffbZVK9ePdM9lv28704F2JS5FB5+ePjJPAaRTUQ2EdlM02ZczOs7F5g0zwRs8k3winRejc6frPz5z3+uhtF79OihwJMjni1btjTda8HzXO769etpxowZ1LRp07y02kn0j9OnT1fXjTt8dyrApszVAJuATZnHADYBm4DNNG3Gxby+c4FJ88zAZvBGwuDJXxi6\/PLL6Wtf+5r6V7r1kXaCPn36nAabfG7s2LG0bNkyqqmpoa1bt9KgQYPUgqQ44PTdqQCbpmaVfx6wCdiUeQxgE7AJ2EzTZlzM6zsXmDTPJGwGb+rw4cP0+OOPq22P+NOV+hOWphvX53n\/Ts7LRxg29ZeLevXqpfbv1AdHQVetWhV7Ld+dCrBp612n0gE2AZsyjwFsAjYBm2najIt5fecCk+aZhE2OZPL30Rky165dq+7x6quvJu6gGAxtI5sMmpyfo5aPPvoo7d69Oy+yqaOYs2fPVuXqI+53fd53pwJsmpoVIps2CmGBkI1KWCBkp9IHqfByh5c7qc+UI73vXGDSMDOwGQbMd999V83ZHDBgAF1xxRXUvHlz070WPM\/gGYbN8BB6GDbjhtJ9dyrApszV8PDDw0\/mMYhsIrKJyGaaNuNiXt+5wKR5JmBTD2dv3ryZzjvvPBo4cKCKZLZp08Z0f9bnJbCp68PfaY+at6mdavTo0dSzZ8+8OrRt25b4L8sHR5UZ8g9ddiedOOs88a00OPASNXt6JvH37DlirMub020P1TY\/Ki5PR8h4w3\/eGmvnzp3EuxfUjThJnzxXXBxteplo2IJ6p9Wv99Gp1O69C8QFatjU9ysuoEgZYDeZkLBb4ZcEtLfC\/oT2ls32Jqt1dOq9e\/cS\/wWPXbt20bhx42jlypV5I6XFuF4WysgEbPLK85\/97Gd02WWXUfv27WO3O+KFQ\/Xr10+0D2cpYDPKARiUGdSyfDD0jxkzJjVszp07l2pra0mXlxY2eZcA9g+OUE+cODE1bIbrlxY2dXmVsj3sJlNewybslq+bfrlDeyvsT2hv2WxvslpHp16xYgUtX7488iRgsxgKV6AM\/prQa6+9Rg899BA999xz9IMf\/EAtFJIeEtg0rUjXkc1Zs2ad9glNHyKbfH+DBw9ODZvcILt37066vLSwuWjRIurYsSNt376dhg8fnho2w\/VLC5u6PKlvFis97CZTUsMm7BYNm2hvhf0J7S2b7U1Wa\/vI5oYNG2jevHmIbBZD4HKWwXM4161bp1aEP\/PMM+rSHPlkYybZczMKNrFAKNqimLMp83TM2YzWCwuE7PwI7c1OJ50K7Q3tTeYx5UmNOZscGszQsX\/\/fnrkkUcUZL7++utq5flnP\/tZuu222+gTn\/hEoiF0vv0o2Iybm4mtj36vVv6nnbOphxN0I0wb2eQ5kTxnkyObHHlNO2czXL+0kc1KD58AWmQdHaAF0CLzmPzUaG8y9Vxpb7Ja26cGbGYANnkuJk+2Zph48skn1deEeHHQ3\/\/+d1q8eLHayD3tEQWbXCaDJc\/\/06BgGkLnPL47FTpRmbe50onCbrAbL8jDy53MD5KmRnuTKedKPymrtX1q37nApITTC4Q4ivnYY4+pibZ79uyh1q1b01e+8hX116xZM7XimDdbD+6BabrhuPNxsBmER50Xn6tEZFPiZ650onj4SayGzfjj1ML0Bzs\/Qnuz00mncqWflNXaPjVg09HIph7C5m1s+vbtS\/\/2b\/+mhsn1hu36fLFg095lzCl9dyp0omYfCKZwpROF3WA3RDZlPpAmNdqbTD1X+klZre1T+84FJiWcjWxqmHz11VdV5PK6666jz3zmM7nN2wGbJtOW7jw6UZm2rnSisBvsBtiU+UCa1GhvMvVc6SdltbZPDdh0NLLJJjx06BCtX78+t+Kco5o8P\/OrX\/2q2tx9xIgRRRtGt3cZc0rfnQqdqNkHENk0a4ThWLNGnALtzU4n14ZjYbds2k1Wa\/vUvnOBSQlnI5vBijMP80bda9asoR\/+8Idq\/qY+5syZQ3369LH+HrpJkGKc992p0InKvMSVN3bYDXZDZFPmA2lSo73J1HOln5TV2j6171xgUiITsBm8Cb0ynTda\/vWvf038jfR27dqpT1jyUHsxP2FpEi\/uvO9OhU5U5hmudKKwG+wG2JT5QJrUaG8y9VzpJ2W1tk\/tOxeYlMgcbAZviD9j+Ytf\/EJtifTSSy+pTx\/y\/ptJviBkEkpy3nenQicq8Qasao5TC8Podn6E9mank07lCrTAbtm0m6zW9ql95wKTEpmGTX1z77\/\/vtrI+3\/\/939VdDPJF4RMQknO++5U6EQl3gDYBGzK\/CWcGu1Nph9gM1ovvNzJ\/KjYqX3nApNeXsCm6SbLfd53p8LDT+ZRePjh4SfzmPzUaG8y9dDe0N5kHlOe1L5zgUlFwKZJoQTnfXcqPPxkToGHHx5+Mo8BbPLncPF52Jeo2dMzc1+v0\/0uPuubpjVVLq\/vXGBSFrBpUijBed+dCrApcwrAJmBT5jGATcAmUYMDgM007ca1vL5zgUlvwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkmhBOd9dyrApswpAJuATZnHADYBm4DNNG3Gxby+c4FJc8CmSaEE5313KsCmzCkAm4BNmccANgGbgM00bcbFvL5zgUlzwKZJoQTnfXcqwKbMKQCbgE2ZxwA2AZuAzTRtxsW8vnOBSXPApkGhrVu30qBBg2j37t15Kfv06UMzZsygpk2bnlaC704F2DQ1q\/zzgE3ApsxjAJuATcBmmjbjYl7fucCkOWDToBA7yNixY2nZsmVUU1Nj0lOd992pAJtWbpBLBNgEbMo8BrAJ2ARspmkzLub1nQtMmgM2DQqtXr2a1q9fHxvFjMruu1MBNk3NCpFNG4U2H2xCd2xpR0uXLqXOnTvT9u3bafDgwVQ34iR98lybEvLTbHqZaNiCerRy5Urq1atX7qWv99Gp1O69C8QF4iUBLwlipwlkQD8pU8+V9iartX1q37nApARg06DQvffeq1KMHz\/epGXuvO9OhU7U2hVUQlc6UdgNdgtC+Jxue6i2+VGZKESElwQ7ydDe7HTSqVzpJ2W1tk\/tOxeYlABsFlDojTfeoCFDhlCLFi1o3bp1uZTDhg0rCJ\/aqUaPHk09e\/bMu0Lbtm2J\/7J8bNy4kQYMGECHLruTTpx1nvhWGhx4iZo9PVNFtPjhp8tL+\/BbuHChipDt3LmThg4dmjpCFq5f2giZLk8sWJEywG4yIfXDD3bL103DJtpbYX9Ce8tme5PVOjr13r17if+Cx65du2jcuHG5kZdiXCdLZQA2C1hLLw7ixUA6sqkBtFOnTsYFQlFFDxw4UIFalo\/NmzfTmDFjUsPm3Llzqba2lnR5aWFz+vTp1L59e7WYa+LEialhM1y\/tLCpy6uU7WE3mfIaNmG3aNhEeyvsT2hv2WxvslpHp16xYgUtX7488qSe5lOM62SpDMBmAmuZwuH6\/KxZs6hDhw7eRTb5\/nhuXdrIJjfI7t27q7l1XF5a2Fy0aBF17NhRzf0bPnx4atgM1y8tbOryErhcUbLAbjIZNWzCbtGwifZW2J\/Q3rLZ3mS1to9sbtiwgebNm4fIZjEErpYydMRz5MiR1K9fv9Nu2wSjWdcJc5FkFnRlLhLsBrthzqbMB9KkRnuTqedKPymrtX1q37nApAQimyaFIs4DNn9PvDVJ2shmeNVw2sgmVjWbIy2wm32Dd+XhB2ixtxmnhN2i9cLCLpkfFTs1YPPkyZPFFtWX8uKcw7T3pu9OhYefzMPx8MPDT+Yx+anR3mTqob2hvck8pjypfecCk4qIbBZQ6MiRIzRhwgTasWMHLVmyhFq1akVRi4bCRfjuVHj4mZpV\/nk8\/PDwk3kMYBObumNT9zRtxsW8vnOBSXPApkkhIuK9Nuvq6nIpeRVm1FxNncB3pwJsWjhNIAlgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmUagLnTAAAFcNJREFU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNCmU4LzvTgXYlDkFYBOwKfMYwCZgE7CZps24mNd3LjBpDtg0KZTgvO9OBdiUOQVgE7Ap8xjAJmATsJmmzbiY13cuMGkO2DQplOC8704F2JQ5BWATsCnzGMAmYBOwmabNuJjXdy4waQ7YNClERNpJdNLp06dTv379YnP67lSuwibb5cUXX6Tzzz+fJk6cSHUjTtInz7UwcCjJppeJhi2oRytXrqRevXrl7N\/76FRq994F4gIBm4VhE3Yr7FJob7Imh\/aG9ibzmPKk9p0LTCoCNg0KsYOMHTuWli1bRjU1NbR161YaNGgQjRw5MhY4fXcqVx9+o0ePpnnz5pH+F7CZjQgZ7JZN2ITdYDcfX8pN0JT0vO9cYNIFsFlAoTfeeIOGDBmiolvjx4\/PpVy9ejWtWrWKlixZQq1atTqtBN+dCrBpalb55xFpKRxpAbQAWnyEFvST2ewnZbW2T+07F5iUAGwWUEhHMWfPnq2AUx9xv+vzvjsVOlFTswJs2ii0+WATumNLu1wkGhHpaNXQ3my86YM0eLnDy53MY8qT2ncuMKkI2CygUHgIPQybcUPp2qn44dmzZ0+TDTJ3fteuXTRu3Dg6en5fOnHW+eL613\/nAH3o2aUKMlgfXd6Adm9SbfOj4vL2Hm9As7a1oeuvv54eeeSR3L9Dv5hszubuN4i+s7LeafW75Hh\/ave+fM7mwXr76KnG9+XKE99gkTLAbjIhYbdovdDe7PwI7c1OJ53KlfYmq7V9au0Pei2AfU4\/UgI2E8CmHl7v379\/5LzNnTt3KhjbsGGDH16Cu4ACUAAKQAEoAAVSKcDBlVmzZlHHjh1TlZPFzIDNEsAmF8nAyX84oAAUgAJQAApAASjAkFmNoMmWB2wmgE2bFeloVlAACkABKAAFoAAUgAKAzYI+kHSBEBwLCkABKAAFoAAUgAJQ4JQCiGwW8IS4uZmmrY\/gXFAACkABKAAFoAAUgAKATSsfYLDkr9HoFWQYQreSDYmgABSAAlAACkABKIDIpq0PSD9XaVsu0kEBKAAFoAAUgAJQwHcFMIzuu4Vxf1AACkABKAAFoAAUqKACgM0Kio9LQwEoAAWgABSAAlDAdwUAm75bGPcHBaAAFIACUAAKQIEKKgDYrKD4uDQUgAJQAApAASgABXxXALDpu4VLfH96df7u3btPu1JtbS0tWbKEWrVqpc7FfWs+TRXj9kJNU2a15NUL3+K+1Ws674JOpfApF+6r1HW49957VXsMts\/gNU3nS12\/LJbPmtXV1cVWffr06ZGfN057r2zH3\/72tzR+\/Pi0RZUsP9ppyaTNTMGAzcyYys2KFtoKKvzAKkWHA9hM7hcmmDSdT35l5Ky0AiaYNJ2vdP1dvH4hzcJb6BWr\/keOHKEJEyZQ+\/btAZvFEhXllEQBwGZJZK2eQgvBZvgcYNMtvzDBpOm8W3eD2kgUMMGk6bzkWtWStpBmGgpZixkzZlDTpk2LIgtgsygyopAyKADYLIPIPl\/CBjZnz55NvXr1ihxGjxqGDw836Q517dq1Skp+i1+2bBnV1NRQVGRTRxH69OmjOvb77ruPeJj\/oosuoilTpuSV8frrr9NNN92UM1F4SFmXpRMEr82\/6a9J8RDW2LFj1XX4KNWQWTF9yQSTUef1V7U2b96cq8qwYcNyURVtj7vuuoseeOAB0uk4zahRo1QURttR24cfvFrHW2+9lUaMGJFnj9atW9OgQYMitQ2+wHTs2DEX5enSpYv6GEPYX3TB4fvg+vHB9ismDBTTXsUsywSTUedNbUHbgts75w\/aXg\/xBr\/Ktm3bttywc3jKTTHvtVxl2cDmjh07TptaFOx\/gm1C1zuu\/+N2MWTIkJzOUf1icHpTsE+ysZW+fnB6ANuJ2+i0adNyfXDUF\/W0nbnfZ9tHBRpM\/sTn169fT82bN1cfVQn3veWyK65THAUAm8XRsWpLMQ2jszD6QRPucKJAMQw4utPq1KlTDgK4E5o\/f77q7PhgENFAqzuwYMeqO0sNRcHOOww8ulwG2ahONPxACYOtBqfgV6dcdQ4pbEZ9vjVsf\/3\/fM\/6hSD4UQQN8+GHUZSO2m5BEAnXOQo2GWa1\/bWtgw95XUe2vfZNfa2oh72r9ktTLyls2rQFbZsoe2l7BCE\/7AvBNp7m3iqVt5Cmcf1YsJ+Iin6a+r\/gC5b2ZZt+1cZWUW1H5wuCXxLYtPEn3ScEX2YrZVtcN70CgM30GlZ1CYUWCLEwwY4iDJvcOYcjSeFhoahOSae59NJL6ZJLLsnB5vbt21U0KxxVjIvSBMGS6xrspLt166aiBv3798+b1B\/uyIPgy4DKRxSUuegk4S9jxdVRQ0GULThP0I47d+5U9hg5cmROtzBYBiMm2v5r1qzJvUBoHQtFVrVdomCTyw9GJ6P8Lrw4Juqh7qLNilUn02IWvo6GRv5vm7YQZa9we47zhTjfKtb9lqOcQrCp9Y572dL1C7+8mfq\/vn37njZn06ZftbFV3Msol88vdPplUgqbOiKbpG8thx1xjdIoANgsja5VU2qhyGY4ylhozmYYfDSkRnWcQXH19bt27Urr1q3Lg9soqNFzpaIgsdBio\/DDuRCAZQ02k6xGD79k6Iighk0daQ7Ctx5Si4PNVatWRQ4xBusX1jZuGD24MtcmTanm1LnaEUgjm8H7iGsLUe07DjbDoOELbMatRg8PAceBXFgvU\/9nmrMZ16\/a2CrOJuG+Uwqb+mUy2A8EdTO93LraplCvwgoANuEhqRQoBJvhaBHP4eJ5jeH5lsF5jsE3dT3HLxypioJNLuPyyy9XwBmGp6gO2wY2g3OK9MOCrx0etg9DUlz0JpXQJcicdBg9PBcvKrLpGmzyvDj2Cx2xDoNvOEJbrAUcJTBbUYqUwqZNWygEMLoNMxxFRUmj2mNRbrSMhRSa5xoebTGNKgTnOBfq\/6JgM\/wiyNcOR0BtbMVz3cMjAFyXcN4ksGnjTz68gJTR\/Zy\/FGDTeRO5XcFCsBmOFgVhU881Cnekwc5TApsMNzzsxwtQwmUmgU29KCU4HMzlRg2jVwtsxgFKlmBT+0h4qxhENvP7maCteRFdeGpEVFuwAZhqg03WKWoeeaFRHm0JG58Mw2ZcnnA6G1vx1JZw36bvxzS3vdACIf3CnqRvdftpiNoVUgCwCf9IpYBNZDNqRaKetxOOMIUXb5jmCd5yyy00fPjw3AKhqPokgU0WRUfDuI760A8Om2H0qOhZKrGLnFkS2YwD+XD02tVh9KAto6C5GudsRkWttIsFNdqyZYtVW7ABmGqETQ16mzZtyo3qxE21Cf9u6v8mTZpEkydPzu2zGTeqEu5XbWzFwYGoPtBmzmah6+kdQOJ2\/sAwepE7ekeKA2w6YoisVsO0Gj04kTxq7lxwlXBwlbiesxkFAUFI0hHI4LBtuDNMApu63OCK5eDwVLXBJoNzWFf22fAq7izAZqHV6NWy8lUyjK4jm6a2YAMw1QibwShwmzZtcvOSwy+uuj0FXwJM\/Z9ul3qhHZfBozumftXGVlFl6b43uONA+BkQ1Y8Hr6cjmyZ\/wjB6Vqkgut6ATb\/sWfa7KbQa3fS5yvD+cVx5hjj+9Fqww5Xus6k7aC6PP8fH+z2GV73bzNkM3xvfD0MtzzvVUUub+UplN4rlBSWRTR3dDS8O4flgfMRtRcXnbFejl3KBUDhCE7fPJtfX5c\/+WZrWmEwCm\/y5WZu2YAMw1QqbbBDd3oLba4Xnbkr22Qzv2qD7Tz0KofezjepXOVodnD\/PaeKG4INtXu+zyX1q+FPEwf1CFy9eTI8\/\/ngu4hr2DRt\/Amwam3GmEgA2M2UuVBYKQIFSKBDcTqtfv36luATKhAJeKMAQyJutV8PHD7wwmCM3Adh0xBCoBhSAAuVRwHbf1fLUBleBAm4qEDXyUWjalJt3gVq5ogBg0xVLoB5QAAqURYGo6Rs+fC6xLOLhIlWlQHCLIn3jcfvyVpUwuFmxAoBNsWTIAAWgABSAAlAACkABKGCrAGDTVimkgwJQAApAASgABaAAFBArANgUS4YMUAAKQAEoAAWgABSAArYKADZtlUI6KAAFoAAUgAJQAApAAbECgE2xZMgABaAAFIACUAAKQAEoYKsAYNNWKaSDAlAACkABKAAFoAAUECsA2BRLhgxQAApAASgABaAAFIACtgoANm2VQjooAAVKroDeNHrw4MHEf8FDnzt27BgtW7aMLrjggrzzS5cuJf7jc\/xt+yFDhuQ+K5q24npvzvbt21f0c5b8zetf\/OIX9PWvf52aNm2a9raQHwpAAShQFgUAm2WRGReBAlDARoFDhw7RmDFjqHnz5jRt2jRq3LhxLtsTTzyhvuf83nvv0ZQpUyj4Wcnjx4\/TpEmT1HfY586dS\/z\/PsImf\/1o9+7d+FSgjTMhDRSAAs4oANh0xhSoCBSAAqzAggUL6Gc\/+xktWbKE2rVrp0Q5ceKEAsy3335b\/T9D6OTJk6lRo0bq\/\/fv30+33HILXXPNNTRixIiiC+lKZBOwWXTTokAoAAXKoABgswwi4xJQAArYK\/D000\/T8OHDacWKFXTxxRerjAcOHKChQ4fmopnLly\/Pg9HnnnuObr75Zqqrq1ND5xzhDEY2NSwyvH7605+m2bNn05\/+9Cc655xz6LbbbqO+fftSgwYN1LVOnjxJTz31VC4Nl3fXXXfRAw88QOFhdIZcrueqVavo9ddfp3\/5l39R9fzXf\/1XVV64Xlw+A7MGYgbrFi1aqOvyt6hHjx6t7is8RSDqE5vTp08nnlKwaNEiNXWgpqYmJ7KOEJ977rk0cuRImjBhggJ3vvcZM2bQK6+8onS644476MILL6R69erl8h4+fFjdE09JiLone0siJRSAAlDglAKATXgCFIACTinw2muv0aBBg+jWW2\/NwSVDG0PTf\/7nf6q6MkjOmzdPARMf\/A3nIIDGweamTZsU3H31q19V8zr\/+7\/\/m9atW6dA8sorr1RlcVT19ttvp89+9rPUv39\/+vOf\/6zSMMBdccUVuTmb+\/bto1GjRtHOnTvpG9\/4hgJRzrtmzRoFy1zfv\/\/97wo+r7rqqhxg6rmnDLUMid26dVPXZfD8wx\/+oP7VAKoNw5FdhuPFixcr8GZtGC6PHj2qtOJrBacVvPDCC6pOrNHHP\/5xBZt87w0bNlTzPbmu\/I3rzZs35wCdr8XwzFMVeG4oR4o7duxIv\/vd71Q9uTy+joZyp5wGlYECUMBpBQCbTpsHlYMC1aeAjuL90z\/9E91zzz0KbjjK9uSTTyoQ44Mjg5\/61KfUv3q+Jv+uh9YLwSZHDs877zxVjo6Y9uzZU0Hkm2++qSKdHAX8zne+Q82aNVPpfvvb36prfe1rX1PpGBS5Lgy5wfLef\/99evDBBxXA8b8MhDz8z9DJEUVe1MNzT\/k3nnv67W9\/m3r37q1AlueqfuITn1DXCUYagx4QHkbnyCZrxP\/q8jk96\/U\/\/\/M\/dP\/991OTJk0UbLJ+\/P+XXXaZKvLgwYMqslm\/fn0VxeV75Ygm3xPXn0FTH7\/5zW\/UdRh2P\/axj1WfU+KOoQAUSKUAYDOVfMgMBaBAKRQIwiVH4xiWOnToQOPGjVOXmzVrFr388stqMRBH9\/QQu47uxcEmwyBD2Yc\/\/GFVjgZb\/m89vMyRQoY6jkbqQ8Ng165dFWzqoXAehtdArNNu27ZNRV45esor6hkuv\/e976noIAMc153rzHX50Ic+pO6J74UjiXxdHa2N0jVqzubatWtVPj2UruvWo0cPBa58LdZP32NwFTuD5Q9+8AMFxh\/5yEdUunfeeYcGDBhAZ5xxRq4Ku3btUkDKUc9gBLUUtkeZUAAK+KcAYNM\/m+KOoEDmFXjmmWdUpI+jhgxHDIA8b1IDIEfaePU5AxZH9XiuIw8Z67mOcbAZBq4wbPKwMg8Vh7dW0tHTli1bKtjU5ffp0+e0LZrC19bD5hx15Wgs39eXvvQlFc389a9\/rYCZI6ccdQwuirKFTQ23Grh5ygH\/N8\/l5DmvhRY38TzRm266SQ2p83A+QzJrEHdw1JejoTigABSAAhIFAJsStZAWCkCBsiiwZ88eBT4cSeODt0FiAPzoRz+q\/l\/P62QAfeuttxQsMVwxDPJRbNjUq+E5EimFzeBineuvv17dE0cieSEOl8XD8T\/60Y8UNAdX2NvCpgZhjpTy8DzPPw3O\/TTBJkcx+foctS3mdlFlcRRcBApAgUwoANjMhJlQSShQXQpogGrTpo3a5oijg8E5iRqg\/vmf\/1mBJR\/B4eyksMmrtIs1jM6LeHiBER8MlM8\/\/zxde+219Oijj6r\/Z7jkoXO+Hs+VZNDj+ZuFjritjzjS+93vfldFSefMmaMWN+lN8bVWHCEOwywPozOcMsgzqHPUlQ8uR89XrS7Pw91CAShQCgUAm6VQFWVCASiQWgEGIZ6PyAcPn4e\/KKQXwfBCF14pHQS1pLDJEUye58irwXlrIQ1cvB0TDyHzkLPNAiEe0meArK2tVfXn4epvfvObdPbZZ6tFQwzGvECII7NbtmxR8yR5pX1w+yLbyCan27t3Lw0bNowuueQStXJ+\/vz5uYU8GjZ5NXpwMRMvhuLpB23btlURUYZ6rjPPKWUY1qvzuXwe5uf75ghzcC5raiOjACgABapCAcBmVZgZNwkFsqcAzz3kIV4+gntu6jvRcxN5aJtXSQdBLSlscvRPg2X37t3pK1\/5ivpiz8KFC9Wekwx0DF182Gx9pLcJ0tMCXnzxRRV5vO6661QZDMxTp06lz3\/+81bRRIbA\/\/qv\/1KLingFvZ5WwJDMUU\/e7\/Mzn\/lMbnU5XyO4R2fnzp1VNLVVq1ZqURCvxud709sv6XvS2rIGzz77rJqmwNs+BVfoZ8+jUGMoAAUqpQBgs1LK47pQAAoUVEBvS8SJeD7mWWedlZdefzWIo4XhYd80sMnbGvGQN0MhRyQZ6AYOHEjr169Xm8Br2OTKRG3qzguMOPoX3I9STwvgBUHBxUdcPgM179dp8+WjV199le6++25Vr\/BiHY4+8tA9n9eQHoRNntPJ2x6xlrw36Be+8AUFrZ06dcrTlSOeDMF6o3reBYAjulymXsUP14UCUAAKSBQAbErUQlooAAWggKMKMIAyCIejvOEV98Gtjxy9FVQLCkABzxQAbHpmUNwOFIAC1acAD6PzfEpe+a7nX2oVAJvV5w+4YyjgmgKATdcsgvpAASgABSwVYLh8+OGH1TfY+XvuwS8EATYtRUQyKAAFSq4AYLPkEuMCUAAKQIHSKKA\/OblhwwY1hH7DDTec9u1yRDZLoz1KhQJQwF6B\/w\/PW\/yV+cSOrgAAAABJRU5ErkJggg==","height":0,"width":0}}
%---
%[output:4a09c78c]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAApsAAAGSCAYAAACsdQoxAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ3UV1WV\/zdICgYWCMYDKL6BViblG+aolY6taQxaZQkxBQIJUaAlr+LLlJIob4ZghAoELRGdyQpGa+VUpi4bHLGexiZFbSB5S1BLSVHR57\/28X9+3ec+995z9r2\/l33P7\/tbi6Vw9r333O\/eZ9\/PPW+3U1tbWxvhBwWgABSAAlAACkABKAAFaqBAJ8BmDVTFKaEAFIACUAAKQAEoAAWMAoBNBAIUgAJQAApAASgABaBAzRQAbNZMWpwYCkABKAAFoAAUgAJQALCJGIACUAAKQAEoAAWgABSomQKAzZpJixNDASgABaAAFIACUAAKADYRA1AACkABKAAFoAAUgAI1UwCwWTNpcWIoAAWgABSAAlAACkABwCZiAApAASgABaAAFIACUKBmCgA2ayYtThyiAi+88AKNHz+eWltb293e2rVr6fTTT+9wy6+++irNmjWLNmzY0K5s7ty5NGLEiHb\/duedd9Lll19O\/fr1o1WrVtGgQYPI\/ps1jB+XdP60uvA57PmGDBlCK1asoF69elXdTdE6DRs2jK6\/\/nrq1q2buc5TTz1FY8eOpR07drS7T1uJG264gZYvX062fps3b6ZRo0Yl2hapeNSPSb4ocu4yHGt1jvsnXndr57qnemn4X\/\/1XzWJB9f9FSnnOrOO0fZWxvsoogGOhQKATcQAFPBUwD4g0szjD9woWCUdM3HiRJo5c2alyAc2s+DNnqjRsMn1iEOjhdq4hlHNopBqtanVQxmw+TbUAzY9G39Os7SXu1rFdc5q4jAoUHMFAJs1lxgXCEGBtB7K6L1FeySjwJV1\/1Ew9IHNeI9kEgBrgM2ke0nSJAo7UTivdU8ZYBOwWY+8BNish8q4RhkUAGyWwUuoY8MViMJJvEcyCnwWknyHkqPnyoJNBtk+ffqY4fsoTNoeRB5qvv\/++83wtC9sLliwgKZNm1aZEhCtSxT84ufLKrOOSrKJanLWWWfRgw8+WBku557PpN4e17\/dfPPNtHLlyso0hbTpAdHhYLaJ3rvP1ISoNkk9sHzfabDs24uV1BOe1ZPNfvnVr35leij5F3\/Zsb6Iv5BEj3P1bEYbXtp9W5ukIfdo7MQBf8uWLc66xxu+r5bR4+L1yvI3+\/kjH\/mIGaq3v6QXn6imVve777670lt89dVX0zXXXNNh+ozVm9uxnR7iG8MNT4KoABQooABgs4B4OLR5FIg+KNMe6mkPZv73LAC0x2XBJgPSueeeS4sWLSILPtE6XXXVVWZOmC9spnkuCh\/2IZ023J817zMJzi1M8bVnz55N1113Xbv6JvUCZcFm2j1E6+XTIx2FiaypD9Hz2rpG9YrOr03SMQvssq6b1vubdP\/x2IzP+Y0fUy3YzJrbaWM\/bb5ztE6udiKBzSzfp708pMVUNEZ8NPWFTZ922DxZFncasgKAzZC9i3urqgJpD5m0Id+0OZ5xePOFzYsvvpgmT55cmWe3bds2s9iGezx57if3UkpgMw2U7P3Y+qfBW9p9xHu67HXWr19vFkDx+ZYuXUrz5s0zPT98veHDh1cWUkXr5YLNJKiLwr2r15ltk3qjo\/ccBUF7z1Fwtou54r2nDP+8MMouEMuaGpAE9kkvH9G6JMFv9H6icOcDxa7G4urR5eOtFkkvG3HYTIJQF\/xKYDNJv6Tj41Bq6xXV2taLbe0CQR9NXcPorFnSeXxeaF3+QjkU0KQAYFOTN1AX9Qpk9eAkwVdWL0j8werq2bRDv7t37zYP9ccee8zAG59n3Lhx9NWvftUbNuMPsySQiMJB0gPY1QsVf9DeeuutZpjR6hRdET1jxgwD0jy8GIUyF2xG65A0dJ+2UClpyDtrekD8XuIQed5553XYpYDrduihh5oXgiiIuYI8DmVRX6UN1SfNQY2CdlSnrCkeWXVzDaPHX5rs35N64tOmorggyxc20+oa\/XefKS\/xVft2+Dv6QsP\/n6apD2y6YtgVLyiHAmVQALBZBi+hjuoUqEUvpws2o72BPGz+29\/+ttIzeNJJJ1W2FPKdsxnf+ihpO5x4j1vS8HGac6Kww3Vn2IzOOY32nKb1zLrmbNqeNK5DFmzGwT4LzpKAJ6keUW0+85nPVHqZjzjiiIpfBg4caObmuXrssoaY02Azaz4kb6uVBWa+Wx9FfZsFm1kvVUmwmbVzQ1b8+sKmz5C9rVfWfaXBZlKMJGnqgs34eXzmQ6tLhqgQFPBQALDpIRJMoECWAmnDlWnHRB9uSUOhSftsWrv77ruvMhTNPZy2x4z\/a\/evrCZsRoEwPvQd3yc0fr9RXeyCoOi9xYdao\/trxrdKih6XBhz1hs2oNtH5tHaBCQNmjx49zHzdrCkH8WFca+saRtcCm0mLZbhH1w43Azbfnjris88mYBPPmlAVAGyG6lncV1UViPbcxGEuCR7tZuRciaS5erYXJAkss2Azel4+t32IPf\/88yLY9BlGjw8P2hXvFnB50\/msX9ICjWgPn6ucz12tns34w77oMDrDcFLvGcfG4MGDE4fUkzb953tMmq7A\/14t2ORz1XIYPam32zVns1HD6Enxmqdn01dT9GxWNQ3jZCVWALBZYueh6vVTwLVBu61JUi9OVi1diwPiDys+V\/QLRvEFK9VYIBSH6fgQqWtIOHq\/8WPj4O0qLwqb0V63pGHT6MtAWo9z0gIhe4\/RObwW4AcMGNDuq1GurzW5wCzvMHr0vGmLiSS+TIOyrOkfrFNamyiyQMinTdkFaS79JLBZ7QVCGEavXw7HlRqrAGCzsfrj6iVSwLXlSfzBkbYaPXrLvpu6R2EhCjgW3nyH31z3kAQfcdCWbLietMdjtIcvfu446BaFzWpufZSkTfT+0lYnu1btczxkLTxzwVK8d1S6TU\/0c6K+PdVpe7ImHS95AXMtOvNpU0krx+P1Suthj\/sqax5m9JzRfXCTdlOwtkn7bLrmHZcoRaKqUCBVAcAmggMKCBRIW3iQBhRpsJMELq4FQnbOl2R4NX5r0Z7SrE3do8dF78G1Wjh+vbSteqyda\/\/SorCZ1gPJ984gEV\/9zvZJPkvzb1qvZ9pK8LRQi1+T4yNphX7aS0XWF5HigMYwajdUr0bPJt9T0kvF1q1bK\/OLOXb5Z3vluQ78490U+OcbVxLY5B0DkkA+7ktJz6Y9Z7Qe9kXQ7raQNVXE2trpMOjZFCRfmJZaAcBmqd2HykOB2iuQd6uc2tcMVyiTAqF\/IjTPCv8y+Q91hQJFFABsCtTjXgXubeDNqF2LIwSnhSkUUK1A0sboqiuMyqlUIATYTNronXs7Q7g3lUGDSgWjAGDT05U2mdgNtQGbnsLBrLQKxIcsXQtdSnujqHhdFAgByFxzgH2nA9RFcFwECihSALDp4YykfeQAmx7CwaTUCiDuS+0+dZUPATatqEkLuvAypi7kUCFFCgA2Hc6wD1ye0M5fA+HvT0dXDyryJaoCBaAAFIACUAAKQAF1CgA2BS5h8ARsCgSDKRSAAlAACkABKND0CgA2BSEggc1t27YR\/8EPCkABKAAFoAAUgAL8wQf+04w\/wKbA676wyZA5ffp02rhxo+DsMIUCUAAKQAEoAAVCVWDo0KE0f\/78pgROwKYgqn1h087z5KDq37+\/4AowLaIAw\/3ixYtNY4buRZSs77HwW331rtbV4LdqKVnf88Bv9dXbXs3q7vpKVmNqV\/urAjYFGkths1mDSiBpVU0t5EP3qspa85PBbzWXuCYXgN9qImvNTwq\/1VzixAs0u+6ATUHcATYFYjXAtNkbcwMkr8ol4beqyFj3k8BvdZe8KheE36oio\/gkza47YFMQMoBNgVgNMG32xtwAyatySfitKjLW\/STwW90lr8oF4beqyCg+SbPrDtgUhAxgUyBWA0y3bNlCq1evpjFjxtCRRx7ZgBrgknkUgN\/yqNb4Y+C3xvsgTw3gtzyqFT8GsNnW1lZcRpwhqkCzB1WjomHfvn20c+dOamlpoa5duzaqGriuUAH4TSiYEnP4TYkjhNWA34SCVcm82bkAPZtVCiTAZg2EFJ4SSVQomBJz+E2JI4TVgN+Egikxh98a4wjAJno2qx55zR5UVRfU84RIop5CKTOD35Q5xLM68JunUMrM4LfGOKTZuQA9mzWIu2YPqhpI6nVKJFEvmdQZwW\/qXOJVIfjNSyZ1RvBbY1zS7FwA2KxB3DV7UNVAUq9TIol6yaTOCH5T5xKvCsFvXjKpM4LfGuOSZucCwGYN4q7Zg6oGknqdEknUSyZ1RvCbOpd4VQh+85JJnRH81hiXNDsXADZrEHfNHlQ1kNTrlEiiXjKpM4Lf1LnEq0Lwm5dM6ozgt8a4pNm5ALBZg7hr9qCqgaRep0QS9ZJJnRH8ps4lXhWC37xkUmcEvzXGJc3OBYDNGsRdswdVDST1OiWSqJdM6ozgN3Uu8aoQ\/OYlkzoj+K0xLml2LgBs1iDumj2oaiCp1ymRRL1kUmcEv6lziVeF4DcvmdQZwW+NcUmzcwFgswZx1+xBVQNJvU6JJOolkzoj+E2dS7wqBL95yaTOCH5rjEuanQsAmzWIu2YPqhpI6nVKJFEvmdQZwW\/qXOJVIfjNSyZ1RvBbY1zS7FwA2KxB3DV7UNVAUq9TIol6yaTOCH5T5xKvCsFvXjKpM4LfGuOSZucCwGYN4q7Zg6oGknqdEknUSyZ1RvCbOpd4VQh+85JJnRH81hiXNDsXADZrEHfNHlQ1kNTrlEiiXjKpM4Lf1LnEq0Lwm5dM6ozgt8a4pNm5ALBZg7hr9qCqgaRep0QS9ZJJnRH8ps4lXhWC37xkUmcEvzXGJc3OBYDNGsRdswdVDST1OiWSqJdM6ozgN3Uu8aoQ\/OYlkzoj+K0xLml2LgBs1iDumj2oaiCp1ymRRL1kUmcEv6lziVeF4DcvmdQZwW+NcUmzcwFgswZx1+xBVQNJvU6JJOolkzoj+E2dS7wqBL95yaTOCH5rjEuanQsAmzWIu2YPqhpI6nVKJFEvmdQZwW\/qXOJVIfjNSyZ1RvBbY1zS7FwA2KxB3DV7UNVAUq9TIol6yaTOCH5T5xKvCsFvXjKpM4LfGuOSZucCwGYN4q7Zg6oGknqdEknUSyZ1RvCbOpd4VQh+85JJnRH81hiXNDsXADZrEHfNHlQ1kNTrlEiiXjKpM4Lf1LnEq0Lwm5dM6ozgt8a4pNm5ALBZg7hr9qCqgaRep0QS9ZJJnRH8ps4lXhWC37xkUmcEvzXGJc3OBYDNGsRdswdVHkm3bdtG\/Cfvb8CAAdS7d2\/auXMntbS00J49ewqfj89pf9WoX\/R8ee+z6HHVvo9qnA9+K+pV+fHwm1yzUI4AbDbGk83OBYDNGsRdMwRVNR5WFr74XNOnT6eNGzfm9sbQoUPp2muvpQMOOIDefPNNuuqqqwqfb\/78+cR1rFb97Ply32TBA6t9H9U6H\/zmdizam1ujqAXng0a3N65PNf1WrfPh5U4WS9WybgYuyNIKsFmtSIqcJ\/Sg4gR6xmcupi57nsytXvRhYPWafuRu6nvgfvE5W1\/uSmt29qSVK1fSwIEDaevWrTRu3Dj611Ft1K+X+HS06WmiW37aidauXUunn3462fqd\/dol1KPtMPEJd3Z+nB47cF3lfOITVOkAex+vnDSO3jq4t\/isXfY8QV2fWN9BF\/hNLKXogGpBvYUvtDeR\/LmNq+23apzv5KP70tzvrqnaS\/mpJw6ghUvXVu2l\/Ph+J9Et675tzhfaL3QucPkLsOlSKEd56EFVK2hZOHgnDemxT6w4w+bUzS0dYHP55DY6+Vjx6QxsTlzaETbP3zeHWt48QXzCnQc8Tvd0vVINbO49cwbt732c+D745aL7Q\/M6wCb8JpZSdADam0gu0tbe8HLn5z8tL+V+tZVbhc4FLkUAmy6FcpSHHlT2\/gAtfsGh7eEHv8Fvo0aNIrwk+MVBXivkSZlyWvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfotOCwFsyuJBao32JlNMS56U1drfOnQucCkB2HQplKM89KBCEpUFhZYkCr\/Bb4BNWQwUsUZ7k6mnJU\/Kau1vHToXuJQAbLoUylEeelAhicqCQksShd\/gN8CmLAaKWKO9ydTTkidltfa3Dp0LXEoANl0K5SgPPaiQRGVBoSWJwm\/wG2BTFgNFrNHeZOppyZOyWvtbh84FLiUAmy6FcpSHHlRIorKg0JJE4Tf4DbApi4Ei1mhvMvW05ElZrf2tQ+cClxKATZdCOcpDDyokUVlQaEmi8Bv8BtiUxUARa7Q3mXpa8qSs1v7WoXOBSwnApkuhHOWhBxWSqCwotCRR+A1+A2zKYqCINdqbTD0teVJWa3\/r0LnApQRg06VQjvLQgwpJVBYUWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUgKw6VIoR3noQYUkKgsKLUkUfoPfAJuyGChijfYmU09LnpTV2t86dC5wKQHYdCmUozz0oEISlQWFliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lABsuhTKUR56UCGJyoJCSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbYFMWA0Ws0d5k6mnJk7Ja+1uHzgUuJQCbLoVylIceVEiisqDQkkThN\/gNsCmLgSLWaG8y9bTkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/wG2JTFQBFrtDeZelrypKzW\/tahc4FLCcCmS6Ec5aEHFZKoLCi0JFH4DX4DbMpioIg12ptMPS15UlZrf+vQucClBGDTpVCO8tCDCklUFhRakij8Br8BNmUxUMQa7U2mnpY8Kau1v3XoXOBSArDpUihHeehBhSQqCwotSRR+g98Am7IYKGKN9iZTT0uelNXa3zp0LnApAdh0KZSjPPSgQhKVBYWWJAq\/wW+ATVkMFLFGe5OppyVPymrtbx06F7iUAGy6FMpRHnpQIYnKgkJLEoXf4DfApiwGilijvcnU05InZbX2tw6dC1xKADZdCuUoDz2okERlQaElicJv8BtgUxYDRazR3mTqacmTslr7W4fOBS4lAJsuhXKUhx5USKKyoNCSROE3+A2wKYuBItZobzL1tORJWa39rUPnApcSgE2XQjnKQw8qJFFZUGhJovAb\/AbYlMVAEWu0N5l6WvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfgNsymKgiDXam0w9LXlSVmt\/69C5wKUEYNOlUI7y0IMKSVQWFFqSKPwGvwE2ZTFQxBrtTaaeljwpq7W\/dehc4FKiqWDTOtuKMnfuXBoxYoRLI7rzzjvp8ssvr9i5jgs9qJBEnSHTzkBLEoXf4DfApiwGilijvcnU05InZbX2tw6dC1xKNA1ssqOnTZtGq1atokGDBtFTTz1FY8eOpSlTpmQCJ4PmkiVLRMeFHlRIoq5m1b5cSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSjQFbL7wwgs0fvx44kQ7c+bMiiYMkuvWraMVK1ZQr169OmiVdtwNN9xAHDhpx4UeVEiirmYF2PRRqPXlrjR1cwutXLmSBg4cSFu3bqVx48bR8sltdPKxPmdob7PpaaKJSzvR2rVrTVu3cXr+vjnU8uYJ4hNqefihvclcB78l64X2JoujaluHzgUuvZoCNm0v5oIFC8xDyP7S\/t2WAzaTwwcPP1ezAmz6KISHn49KVIHmvWfOoP29j\/M7KGLVZc+T1P2heR0gfOHgnTSkxz7x+eA3P8mQJ\/10slZaXhJktfa3Bmy2tbX5y1VOy\/gQehw2s4bS04bRhw0b1q6XNKqMDapLL72Uhg4d2k60vn37Ev8p8+\/RRx+l0aNHU9GHH\/doMfzb8xV9+C1btsz0kG3bto0mTJhQuIcsXr+iPWT2fI3yPfwmU94+\/OC39rpZ2ER7y44ntLdytjdZrZOtd+3aRfwn+tu+fTtNnz698tJXjeuU6RxN0bOZBpu253LkyJGZ8zZtD+iOHTuMb+0wXZqj4wuRonZjxowxoFbmX2trK1122WWFYXPRokU0ZMgQsucrCpu8cKtfv37EfuIFXUWHY+P1Kwqb9nyN8j38JlPewib8lgybaG\/Z8YT2Vs72Jqt1svWaNWto9erViYUufqjG9TWeA7A5fjylwearr75Ks2bNoj\/96U+V+ZkWUI844gi6\/vrrqVu3bh38amFz\/vz51L9\/\/+B6Nvn+eG5d0Z5NbpCnnHKKGSbk8xWFzVtuuYUGDBhg5v5NmjSpMGzG61cUNu35GpUI4DeZ8hY24bdk2ER7y44ntLdytjdZrf17Njdu3EiLFy92dlZV4\/oaz9HUsOlakZ42x8I11zP0uRmYiyRrylrmIsFv8BtWo8tioIg12ptMPS15UlZrf+vQucClRFPAZt4FQnmH30MPKiRRV7NqX64licJv8BtgUxYDRazR3mTqacmTslr7W4fOBS4lmgI20+ZmurY+Qs9mcvggibqaFWDTRyGsavZRCavR\/VT6u5UWaEGelHlOi99ktfa3Bmw2wWp0Dgf7FSA7Odc1hM7HFJ2zGepEYCRR\/wTDllqSKPwGv6FnUxYDRazR3mTqacmTslr7WwM2mwQ2OSTiq8Tjn52Mb3Nkw4g3cV++fHklqiZOnJi67VH0OoDN5IaIff\/8E1Q1LfHwk6mp5eEHv8FveEmQxYBGa8BmE8FmvQIw9KDCw08WSYCWZL0wjO4XR2hvfjpZK7Q3tDdZxNTHOnQucKnYFHM2XSJUuzz0oMLDTxYxePjh4SeLmPbWaG8y9dDe0N5kEVMf69C5wKUiYNOlUI7y0IMKDz9ZUODhh4efLGIAm6NGjaKi+9o2ehoT8qQs6rXkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/yGuX+yGChijfYmU09LnpTV2t86dC5wKQHYdCmUozz0oEISlQWFliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lABsuhTKUR56UCGJyoJCSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbYFMWA0Ws0d5k6mnJk7Ja+1uHzgUuJQCbLoVylIceVEiisqDQkkThN\/gNsCmLgSLWaG8y9bTkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/wG2JTFQBFrtDeZelrypKzW\/tahc4FLCcCmS6Ec5aEHFZKoLCi0JFH4DX4DbMpioIg12ptMPS15UlZrf+vQucClBGDTpVCO8tCDCklUFhRakij8Br8BNmUxUMQa7U2mnpY8Kau1v3XoXOBSArDpUihHeehBhSQqCwotSRR+g98Am7IYKGKN9iZTT0uelNXa3zp0LnApAdh0KZSjPPSgQhKVBYWWJAq\/wW+ATVkMFLFGe5OppyVPymrtbx06F7iUAGy6FMpRHnpQIYnKgkJLEoXf4DfApiwGilijvcnU05InZbX2tw6dC1xKADZdCuUoDz2okERlQaElicJv8BtgUxYDRazR3mTqacmTslr7W4fOBS4lAJsuhXKUhx5USKKyoNCSROE3+A2wKYuBItZobzL1tORJWa39rUPnApcSgE2XQjnKQw8qJFFZUGhJovAb\/AbYlMVAEWu0N5l6WvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfgNsymKgiDXam0w9LXlSVmt\/69C5wKUEYNOlUI7y0IMKSVQWFFqSKPwGvwE2ZTFQxBrtTaaeljwpq7W\/dehc4FICsOlSKEd56EGFJCoLCi1JFH6D3wCbshgoYo32JlNPS56U1drfOnQucCkB2HQplKM89KBCEpUFhZYkCr\/Bb4BNWQwUsUZ7k6mnJU\/Kau1vHToXuJQAbLoUylEeelAhicqCQksShd\/gN8CmLAaKWKO9ydTTkidltfa3Dp0LXEoANl0K5SgPPaiQRGVBoSWJwm\/wG2BTFgNFrNHeZOppyZOyWvtbh84FLiUAmy6FcpSHHlRIorKg0JJE4Tf4DbApi4Ei1mhvMvW05ElZrf2tQ+cClxKATZdCOcpDDyokUVlQaEmi8Bv8BtiUxUARa7Q3mXpa8qSs1v7WoXOBSwnApkuhHOWhBxWSqCwotCRR+A1+A2zKYqCINdqbTD0teVJWa3\/r0LnApQRg06VQjvLQgwpJVBYUWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUgKw6VIoR3noQYUkKgsKLUkUfoPfAJuyGChijfYmU09LnpTV2t86dC5wKQHYdCmUozz0oEISlQWFliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lABsuhTKUR56UCGJyoJCSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbYFMWA0Ws0d5k6mnJk7Ja+1uHzgUuJQCbLoVylIceVEiisqDQkkThN\/gNsCmLgSLWaG8y9bTkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/wG2JTFQBFrtDeZelrypKzW\/tahc4FLCRWw+dprr9Hvfvc74v\/6\/g466CA68cQTif+r7Rd6UCGJyiJOSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSqiAzRdeeIHGjx9Pra2trvpWyocMGUIrVqygXr16eR9TL8PQgwpJVBZJWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUkIFbL744os0YcIE2rRpE3384x+nz3\/+8\/S+972PDjjggNT6d+rUid71rndR586dXfdY9\/LQgwpJVBZSWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUkIFbHIlX3\/9dXrsscfo3\/\/932nDhg10yCGH0Oc+9zn61Kc+RYMGDVIJlWnihh5USKKuZtW+XEsShd\/gN8CmLAaKWKO9ydTTkidltfa3Dp0LXEqogc2JnAyBAAAgAElEQVRoRV999VV69NFH6Y477qCf\/\/zn1Lt37wp4Dhw4UD14hh5USKKuZgXY9FGo9eWuNHVzC61cuZK4XW\/dupXGjRtHyye30cnH+pyhvc2mp4kmLu1Ea9eupShUnb9vDrW8eYL4hFoefmhvMtfBb8l6ob3J4qja1qFzgUsvlbAZrfTevXvpkUceodtvv50efPBBOvzww+mzn\/0sDR8+nPr160c8nK7tF3pQ4eEnizg8\/PDwk0VMe2u0N5l6aG9ob7KIqY916FzgUlE9bMbB8yc\/+QnNmTOHjj76aCwQcnm3RuV4+MmExcMPDz9ZxAA2R40aReiRfpK6PzSvQ0\/9wsE7aUiPfeKQQs+mWLKqHgDYbGtrq6qiVT4ZV+\/Pf\/4z3X\/\/\/aZ38\/e\/\/z316NGDvvjFL9KXv\/xl6t69e5WvWPx0oQcVYFMWI4BNwKYsYgCbgE2iLnsAm0XajbZjQ+cCl94qezbTAPOCCy6gYcOG0Qc+8AHq0qWL694aVh56UAE2ZaEF2ARsyiIGsAnYBGwWaTMajw2dC1yaq4FNC5g8TH733XdXejA\/8YlP0IgRI9QDZlTo0IMKsOlqVu3LAZuATVnEADYBm4DNIm1G47Ghc4FLcxWwyftsTps2jX75y1+aIfIyAiZg0xVqfy\/H8JC\/VtW0xEuCTE28JOAlQRYxeEkI4SWhiM+zjgVsKpizGf2C0FFHHWX22HT9BgwYQNdccw317NnTZVr38tCDCtAiCylAC6BFFjGAlhCgBXlSFvVa8qSs1v7WoXOBSwk1PZtXX301bdu2zVXfSjlg01uqqhsiicok1ZJE4Tf4DZu6y2KgiDXam0w9LXlSVmt\/a8Cmgp5Nf3eVwzL0oEISlcWhliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lFDRs+mqZNnKQw8qJFFZRGpJovAb\/AbYlMVAEWu0N5l6WvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfgNsymKgiDXam0w9LXlSVmt\/69C5wKWECtjk1eiYs+lylZ5yJFGZL7QkUfgNfgNsymKgiDXam0w9LXlSVmt\/a8Cmgjmb0dXovq4bMmQIPlfpK1aV7ZBEZYJqSaLwG\/wG2JTFQBFrtDeZelrypKzW\/taATQWw6e+ucliGHlRIorI41JJE4Tf4DbApi4Ei1mhvMvW05ElZrf2tQ+cClxIqhtFdlSxbeehBhSQqi0gtSRR+g98Am7IYKGKN9iZTT0uelNXa3zp0LnApUQrY5M7Xl19+mfbv30\/du3enAw880HVfieXW2bZw7ty55lOYrl\/8OP4++\/XXX0\/dunXLvM7atWuJk3toPyRRmUe1JFH4DX4DbMpioIi1pL11fmUPdX7l+XaX4387+LGVdOmll9LQoUNp+\/btNH36dBrd8iIN6bFPXLVdr3eh+Vv60Fe\/+lU65phjaO\/evWatxIR\/aqOTjxWfjna8QPTNtZ061O+k10dSy1sniE\/4cqfn6IGDbqqcT3yCBh\/Ae3\/zn7QfYFPRMPpLL71E69atM5u7z5gxw4AlN4gbb7yRvv\/97xvYPPTQQ4kh8dxzz6VOnTp5hxc7mj+JuWrVKho0aBA99dRTNHbsWJoyZUomcN555510+eWXkwVHO7\/0iCOOSAXO0INKkkSTHITPVXqHbVUN4TeZnHhJSNar9eWuNHVzC61cuZIGDhxIW7dupXHjxtHyyfmgZdPTRBOXdqrkWBun5++bQy1vyqGlbH5jqDx+y1rasblVFqCwVqUAvxDMnz8\/FThD5wKXM9T0bD733HN0ySWX0COPPEIf+9jHaMGCBfTud7+b1qxZQ9\/61rdowoQJdNJJJ9Htt99Ojz76KH3ve9+jD33oQ677M+UWEPmtfubMmZVjGCQZblesWEG9evXqcC573MiRI9sBaRxc4weGHlSAFq+wqxiV7eGXdnd4SZD5vVrWaG8yJcvW3my7YlDp37+\/7GZhrUKBjRs30uLFiysvTEmVCp0LXI5QA5sMlQsXLqSbbrqJzj77bNNryVsicZf\/8ccfT7Nnz6YuXbqYoQQGzzPOOMOAI\/+b62d7MRlgo8Paaf9uz8fBccMNN4hXvYceVHj4uSKufXnZHn6AzbcVgN\/Qsylr6e2tffNk\/CWuyDVxbGMU8Hnm+9g0pvb1uaoK2Pzb3\/5Gs2bNMnMgr732WjrooIPM3T\/++OM0fvx488ZgIZFH\/XkY\/YknnqClS5fSIYcc4lQqrSfSNZTOPZ8PP\/ywGdKfPHkytba+PcwxceLEdj2k8QrYoLJzbaLlffv2Jf5T5h\/3LI8ePZr2njmD9vc+TnwrNrnyMBz71Z5v4eCdueYi2WG9ZcuWmWE9nobBLyRFh\/Xi9Ss6rGfPJxasSgfAbzIhLWzCb+11Q3vziyPf9pYGm5zH+E+Zf655jGW+t2jd7TPf5opdu3YR\/4n+7JzbUNdyuHypAjbThrl\/+MMf0q233mp6FltaWir3wr2N7Ny04e8k+IvO17TlacPktpyvs3z5curXr19lrqdkzmaS+GPGjDGgVuYfQ\/dll11WGDYXLVpEvF+qPV9R2OSXEPbVjh07zDzborAZr19R2LTna5Tv4TeZ8hY24bdk2ER7y44n3\/aWBJsMmbwYiIdny\/xzzWMs870lwabNFTxSu3r16sTbA2w20OtJsMmLgbiX8y9\/+UuHhTgMgfzWeMstt1DPnj2dNU\/r2fSFzXhwuLrDbXnSHJwQejb5\/nhBQNGeTW6Qp5xyinlx4PMVhU2OB36T5gULkyZNKgyb8foVhU17PmfA1sgAfpMJa2ETfkuGTbS37HjybW9JsGmfIa+cNI7eOri3LHCVWHfZ8wR1fWJ95jxGrqp9DvO0uKzdW5555hnzvE9aX9HoW7b+srkiqWfTZ15no++jltdX0bP52muvmTmZBxxwQGUYfc+ePWYo9JOf\/KQBEfvj1encq9ajRw+67rrrKkPuWSLlHUZnqN2wYUOlV9O3R9QFo7V0aD3O7TsXKa0uWGhSDy91vAb8JtMdczaT9cJqdL848m1vWbCZ94Xer4a1tfKdi2pHELN6\/PKun6jtHf797D7PfB+betW3EddRAZt84zw\/8qqrrqIrrriCPvGJT9Bdd91ltjviVefvfe97jTY8X\/O+++4z8yevvPJK7+HovAuEuE5LliwBbMYi0zeJAjbfVgDQAmgpktzR3mTqla29NTNsWojcvXu32YEmrWcTsClrAxqt1cAm91jOmTPHQCb\/eJU5d6vzHEf+fy7\/xje+QevXrzdDr7xoqE+fPl6apg2Xu7Y+SltAlNZTaisT+hsMHn5eYVcxKtvDDy8JeEnIinD0bPq1f9882aywyc9XXnzLz3nu3UwbRrc6WtVtD2j837ncfqjFPvOHDx9u1nbwPH7+RXtPrY1d+MvrB3zXgcQjwOeZ72PjF1nltFIDmywfz9PkVeY834G\/cHDkkUdWNm5n2OQV64cddpjZiN1nrmbUJfHN2V0r0e2x8aH0tMVM0WuFHlS+SRTQAmgBtBR\/MKC9yTQs28tdM8Lmq6++ap7nvIXheeedZ3adyZqzGe\/ZtKA6b94885EW\/kVHIvnjL3xO7jG1H3KJdhLx3H57ffsVQX7W8y+6F7dv5Pk8831sfK9XRjtVsFlrAeNvQvHPVaYNm1tQtfXz3foo1FVnePjJIrVsDz+8JOAlAS8JsjaeZO2bJ5sRNu22gvzZZwZPKWwm6R0FUAub8Q+yWKC8+OKLzTXj5Xm97gOSPjZ5r1+G45oKNuvlkNCDyjeJAloALYCW4lkH7U2mYdle7poNNuO9kj6r0dPmbMaHwu02hRY2472lUcjlKXm8RR7\/igyh8\/E+z3wfG1mkl8taFWzyMDpvacR7jA0bNsysNOd\/+9GPfmTmdDz\/\/PNmAjHP3Rw8eLBapUMPKjz8ZKFXtocfXhLwkoCXBFkbR89mRwXSVqPHRwqjR6aNGiYNo48dO9bMxbQjlEk9m1mwyR+R4V8UWPNCp88z38emeNTpPYMa2IwuEIo6\/N5776Wvfe1rBjLf\/\/73009\/+lPq3Lmz2WOT53Vq\/IUeVIBNWdQBNpP1wkITvzhCe\/PTyVqVrb01W89m3Jt5ejajPZQWGqNzMl3D6EnzMn3qkRaJPs98HxtZpJfLWg1s8n6WU6dONVsfXXjhhebTlXZPTe7h5LeX7t2705NPPmnmWvCkXt4Cib+hru0XelDh4SeLuLI9\/NCziZ5N9GzK2jh6Nv17NvPAZnzoPb6TjF3wy+fmBUGuBUJJMJo2VO8TCT7PfB8bn2uV1UYFbNpN3fkb6bzXFkMl\/zZv3kzcVc5fEjrnnHPMv9kvC+3cuZP401DWVpMDQg8qwKYs2gCb6NmURUx7a7Q3mXpla2\/o2XzBuUDIrl7nTik7bG43g+fo4LmazA78b7zox65wj259FP3sNB9jAdVuixQvl0SdzzPfx0ZyzbLZqoDNtO2E7rnnHrrxxhvNm8rhhx9e0Vb6bfR6OyX0oMLDTxZRZXv4oWcTPZvo2ZS18Vr1bO47fjjt73188co04AydX9lDBz+20vm5ylpUrciQeJ76+DzzfWzyXLssx6iFTf5aEH9b\/Omnn+7QgwnYbGx4ATZl+gM20bMpixj0bI4aNYrO3zeHWt48QSxd2dpbUs8mL5KdPn068fe0y\/wbOnSoeY7zvpb1\/AE266m237VUwCYPn\/MGq7169TKfrOQvBr300ktmTuZpp51m\/mt\/tjud\/857dNnJwX63Wx+r0N9gAJuyOCrbww89m+jZRM+mrI1Xu2eTz8fAyX\/K\/GPIrDdosl6ATX1RowI2WRZeXX7zzTfTTTfdRGeddRbxEDp\/\/3zlypV08sknV5T7zW9+QxdddBHxpqxRCNUkLWAz2xvxN3mr18LBO2lIj31iV2JVs59keEnw08la4SUhWS+0N7848m1vaVsE+V0FVhoU8Hnm+9houJda1UENbD733HN0ySWX0COPPFK519GjR5tNV3k1Oi8iuv3222nZsmXmU5UMp\/w5S42\/0IPKN4mihww9ZOghK56h0N5kGpbtJQGwKfOvRmufZ76PjcZ7q1ad1MAm3xAPnfNqs61bt9KHPvQhswKdQdOWcU8mQyevTsem7tUKAfl58PCTaVa2hx9eEvCSgJcEWRtPsvbNk2mwiWH04j6o1xl8QNLHpl71bcR1VMFmlgC8YOiVV16hgw8+WOXemtG6hx5UvkkU0AJoAbQUT+tobzINy\/Zyl7ZA6OsXXUib\/rhLdvPKrE8+ui\/d+L27GjJvs55S+DzzfWzqWed6X6s0sFlvYYpcL\/SgwsNPFh1le\/jhJQEvCXhJkLXxavds2hw7\/cjd1PfA\/cUr04Az8NzeNTt7pm59FN0n0+czkc8884yZQscLibX9fJ75Pjba7qua9QFsVlPN\/3+u0IMKsCkLGsBmsl5YaOIXR2hvfjpZq7K1t6xN3fMumpQpVhtr277Xrl1rPjcd\/cW\/AMTgyZurp+0ww22AbVasWAHYrI27an5WwGYNJAZsZouK1eg1CDqPUwJaPESKmJQNWtAjXc4e6WaDTbt94RlnnGE+O80\/11ZFgE1Z7tJoDdisgVcAm4DNqAKAFvRsFkkzeEmQqVe29gbYzIZNG\/82CmxPafzfudx+ytLCa\/RzlVwe7WW1Nq2trebUPkP5aZHo88z3sZFFermsAZs18FfoQYWHnyxoyvbwQw9ZOXvI4Ldy+q3ZYJO9FB9Gj\/89Hsvxnk3+rvmMGTNo3rx5NGjQIGPO51iyZIn5vPWhhx5qvre+e\/du83e24XNMmzbN\/J03mucPyUR7V3mYnn8zZ86UJXgic27+6lXSlAF7Mh8b8YVLdABgswbOCj2oAJuyoAFsomdTFjHtrdHeZOqVrb01I2xaOOR9tPk3bNiwzC8C+gyjRwHUwubIkSMrQ\/V8HQuU\/FEYhtF4uSzS\/m7t88z3scl7\/TIcpxY2t2\/fbt5SfvKTn9DRRx9tJgbznpv8JsP7b5599tlqt0AKPajw8JM17bI9\/NBDVs4eMvitnH5rNti0czb79etX6UV0PTPTYDM+FM7njPZsci9ldHES934+\/PDDBmzXr19vPhrDvyJD6Hy8q\/6+NrInS7msVcImb3EwYcIEOvDAA+nYY48134dl2HzHO95BU6dOpY0bN9Ly5cs7rHDTIr1P4Gmpa556ADZlqgE20bMpixj0bPKQ5Pn75lDLmyeIpStbe2s22EwDx6xh7KRh9LFjx5oV7HaeZlLPZhZsduvWzcRWFFjzQqfPM9\/HRhzsJTpAHWzu37\/fdHXzV4Tmz59Pf\/jDH9ptebB3717zNvLOd76TrrnmGgOk2n6hBxVgUxZxZXv4oYesnD1k8Fs5\/QbYfNtvEtiM9lBaaIzOyXQNoyfNy3StiM\/K+j7PfB8b2ZOlXNbqYPPFF180vZqf\/exnzVyLpLegH\/7wh7RmzZpS77lVrjBBT0sz9bQAWsoJLfBbOf3WbLCZZxg9viAovqCIy7mnk38+C4SSYNRnXmhaG\/MBSR+bMnOBq+7qYNO+XXzpS1+i888\/PxE277nnHrrtttsAmy7v1qgcPZsyYdGzmawXNnX3iyO0Nz+drFXZ2luzwSb7yQLnhg0bjNvsXEu7sjzu8ai9HTaPfoGIj1+wYIHpHeVFP+edd55ZABTd+ih+DQuoPBTvUwf0bMraYdxaHWzaoOI3j9mzZ9Ojjz6aOIzOw+fXXXedWTSk7Rf6GwwefrKIK9vDDz1k5ewhg9\/K6bcs2Bzd8iIN6bFPlnCUWO96vQvN39InczugWlW1yJB4njr5PPN9bPJcuyzHqINNFu6hhx6ir3\/968TbExx++OH0ne98h2666Sb685\/\/TLfeeis9+OCD9N3vftesStf4Cz2oAJuyqANsomdTFjHtrdHeZOqVrb0lwSYvip0+fbpZDFvm39ChQ83aC97Xsp4\/wGY91fa7lkrYbGtro\/\/4j\/8wC4Cef\/75dnfCK9LnzJlDF1xwAXXu3NnvLutsBdjMFhyfq6xzQP7\/ywFaZLqXDVrQsxlOzybfCQMn\/ynzjyGz3qDJegE29UWNSti0MvHKc16N\/tvf\/pZee+01sxfWBz\/4QerRo4c+JSM1AmwCNqMKAFrQs1kkYeElQaZe2dpbUs+m7I5h3WgFfJ75PjaNvo9aXl81bNbyxmt57tCDCg8\/WfSU7eGHHrJy9pDBb+X0WxpsomdTlmcbae3zzPexaeQ91Pra6mCTezPvuusu+utf\/+q8d15E9OEPf5iOOeYYVUPqoQcVYNMZmu0MAJvo2ZRFTHtrtDeZemVrb5izKfOvRmufZ76PjcZ7q1ad1MEm77N59dVXmy2P7HzNo446ytzv\/\/3f\/yXe94UXXkhXXnklde\/evVq6FDpP6EGFh58sPMr28EMPWTl7yOC3cvotazX6v45qo369ZPlGi\/Wmp4lu+Wkn52p03jNzy5YtlU9X2vpHtzbif1u7dm3mVwN5nibzA3c+1fvn88z3sal3vet5PXWwyTf\/+OOP09e+9jWzGp33ybJfCOCvC913333mm+nf\/va36cgjj6QHHniAvvGNb5iN4EePHl1P7VKvFXpQATZlYQbYRM+mLGLQs9lMH1HIgs3lk9vo5GOLRE\/jjmXYnLg0GzYZNPmLgBMnTmwHmwyavP8lf8Ocn\/92T0zeSzP6rXN7d\/VeEBRX1eeZ72PTOG\/V\/srqYNPus8nfRJ88eTJ16tSpnQq8Un3p0qX07LPP0rXXXmv22Vy5cqXpCb3llltqr5jHFUIPKsCmRxBETACbgE1ZxAA2AZv\/RaxBqLBpn\/ObNm2iPn36GIC0n5BMA8esz1kCNotkmPocqw42bdBMmjSJPv7xjyeq8LOf\/YyWLVtW+YLQL37xC9PbyZ+x1PADbGZ7AVsfNSZK8ZIg0x0vCXhJkEVMvpeEZuzZ5Fx0xx13mJ5L3kObf0nfK48qmgablhlaW1uNue0ljX8hKFrG\/8\/ne\/nll80f+yWjeA+r7Xm19bBfL0LPprxlqINNXiB02WWXEfds8n+7dOnS7q54KH3RokX09NNPm\/\/yPM1169bRvffea76XruEH2ARsRhUAtABaiuQlvCTI1Ctbe2tG2PSByKiNBUr+FOWIESM6BES8ZzOppzP+XLZzQi1Axq\/B9myzYsUK6tWrlxnKnzFjBs2bN4\/in9X0eeb72MgivVzW6mCT5bv77rvpiiuuoK985Sv0+c9\/nnjVOf94wRC\/DXEvJg+h88KgJ554wiwo4v03+RgNv9CDCg8\/WZSV7eGXdnfokZb5vVrWaG8yJcvW3gCbNxgHp\/Vs2iF3trFzOOMR4TOMHreJzwvlc0YBk9eHcEeWhc2sKPR55vvYyCK9XNYqYZN7L1evXm3eIN544412ivIXhPjtYsyYMWaj91mzZpmvLHAvp1213mgXhB5UePjJIqxsDz\/A5tsKwG\/JkdD6cleaurnFzJUfOHAgbd26lcaNG5d7fmF8IYnNL+fvm0Mtb54ga2wl9BtgMx02LWj+6U9\/yoS+NNi0x9thcg4mu6o9aVg+2nvJnVzjx48nOzyfNoRuIZXn2GatmA+dC1wNVSVs2krzNgb8bVj+ihD\/Bg8eTKeddpqZUMw\/hk3u7ezduzcdeOCBrnutW3noQQXYlIUSoAXQIouY9tZobzL1ytbeAJvJsOkLmhwdacPoDIp2HmZSz2a8RzVtqDw6dzMJOn2e+T42skgvl7Vq2CyXlH+vbehBhYefLDLL9vBDzyZ6NrMiHD2bfu3fN08CNjvCps\/QedQLcZCMz7dk2\/j2Sa5hdJ6nGf8xdD788MMdhvN9nvk+Nn6RVU4r1bDJi4Vef\/31DsryMPvmzZvNPE0tG7lHKxl6UPkmUUALoAXQUvzBgPYm07BsL3eAzY6wmQSCWVFg4fSMM84wC4i4zUybNo1WrVplFvNEV6xHh9GXL19OaQuEGCyjczbj15A+80PnAlcrVQmbzz33HM2ePZt4S6O035AhQ7wm7roEqEV56EGFh58sasr28MNLAl4S8JIga+NJ1r55ErDZHjaTtiyy+g4bNix1kZAd6rY269evNxvG299tt91GP\/7xj6lfv35mMVLS1kfxIfL4V4ziWyPZc\/s8831siked3jOog03utWQH8zZGY8eOJYZK3qz91FNPNW8ovFKdFwlxUHDQxDd91yB16EHlm0QBLYAWQEvxjIT2JtOwbC93WbA54Z\/K+wWhHS8QfXOt+3OVMu9Wzzprk3jpVXye+T420uuWyV4dbL700kvmy0EMl\/xf\/jFYHnDAAeZthDdgnTp1Kp177rmJ+21pED\/0oMLDTxZlZXv44SUBLwl4SZC18Wr3bPIOK9OnTzcLZMv8O\/XEAbRw6VoaMGCAutsAbNbXJepg086tmDJlCp1zzjlGjfikXJ5H8etf\/zq1O72+Ena8GmAz2wPYr7ExEYqXBJnueElI1gsLhPziyLe9JfVs8hUYOPlPmX8MmRpBkzUFbNY3stTBpu3ZPP\/88ys9l9xoFy5caIbTe\/bsaeZy8sbuPput1lfOt68G2ARsRhUAtABaiuQhX2hBj3Q5e6TTYLNIzODY+irg88z3salvret7NXWwaedsPvvss2b4nOGSJwxfeuml5u8nnngiLV26lB544IEKfNZXMvfVQg8qPPzcMQDYdGuEHjK3RtGX171nzqD9vY\/zOyhihZEEsWRVOcA3TwI2qyJ3Q0\/i88z3sWnoTdT44upgk++XtzWaNGmS2bx92bJl1LVrV\/OlIN7fijdwf+aZZ8xGrQyg8W+n11gvr9OHHlS+SRQ9LeXsaYHf4LesRIeXBK\/HQGWEy\/WSgGF0Pz01W\/k8831sNN9j0bqphE2+KZ6r8j\/\/8z9mIRB\/HWjXrl1mjgUPoV9wwQXEczq511PjL\/SgAmzKog7D6Ml6AVr84gjtzU8na1W29pYEm6EsEBo6dCjNnz9f7bxNWWSlW\/s8831sqlUfjedRCZv8HVTu1ezWrZtGzZx1Cj2o8PBzhkA7g7I9\/NCziZ5N9GzK2niStW+ezNr66OzXLqEebYcVr0wDzrCz8+P02IHrMr8XztXiBcBbtmwxu81Ef\/H9NrO+Tc7H8eJi\/sT1McccU\/e79Xnm+9jUveJ1vKA62LQLhD760Y\/SuHHj6ihF9S4VelD5JlFAC6AF0FI8r6C9yTQs28tdFmyev28Otbx5gkwAJdY+frAbscc3S48\/Q+OfmozfYvxzlfWWwOeZ72NT73rX83rqYJPfTCZMmEAXXXQR8Yr0Mv5CDyo8\/GRR6ZN0ZWfMZw2\/yXSD35L1wvQHvzjybW\/NCJv204+bNm0yo5inn356pWfTltkv\/Vi1s7YqAmz6xWQjrdTBZltbG\/3whz+ku+66i6644gp673vfq3IRUJbTAJvZIY3VsY1p8r4PP\/RIo0caPdLF26hve2tG2GRt7rjjDrNX9k033WTEtsPoUnCMfvecz2N7SZM+exntQU36XGW8h9X2vNpoSBvK93nm+9gUjzq9Z1AHm9yzedVVV9GDDz5ovhaU9sO30RsXVL5JFNACaAG0FG+naG8yDcvWI92MsBn1aLzHkiFxxowZle+Xt7a2GvOsOZtxQE0C1jjs2e+e2\/PaY0aOHGn2+GZ7trH7edt6zZs3z3w6O\/rzAUkfG1mkl8taHWzu3bvX9Gr+9a9\/zVTyXe96F1144YXUvXt3b8Wts11vKVkn5Dcd\/oJR1obyoQcVHn7eIWcMy\/bww0sCXhLwkiBr40nWvnkSsHmDkc\/2bNoeSR5ej4Le2LFjzS40DILxn09vaNyGQXLHjh3tvkQYBcz77rvP+ay39fB55vvYFI86vWdQB5u1koodPW3aNFq1apV5K7EBnRa8SfVIagRZSWbt2rVmLkpoP98kCmgBtABaird+tDeZhmV7uQNsdoRN7tmM9yBmdfSkwaad\/7lhw4ZKENnnctIc0Gjv5aGHHvBhPOsAACAASURBVErjx48nn55VH5D0sZFFermsVcMm93Ky8\/mrQvzloAMOOID27NlDhx12GHXu3NlbaRuI0UnIfLBPL6W9SDRoXUP4oQcVHn7eoYeezQypsNDEL47Q3vx0slaATZletbL29UPaMHocNuPD2tF6pw2jMyjaeZhJPZvRHlX+\/7Sh8ujczaThfJ9nvo9NrXyh4bwqYZPhkp17zTXX0BtvvEEW7vhrQZMnTyZepXbllVd6D6GnbZvg2k4h6iCuD3\/B6IMf\/CCtX78ew+ijRpHryxjo2UTPJno2i6d5wKZMQ1\/IkZ1Vbu3rN\/Rstu\/ZtB07n\/\/859uNDNpnMC8qiu\/BHQfJJDCNP+9dw+i9evXq4PS0OviApI+NPMrKc4RK2OSvBPGnKL\/+9a9TS0uL+QY6z9045JBDzEr1OXPm0NSpU2n06NFeSseH0O1BvkPp0eMfe+wx5zyO0IPKN4kCNgGbgE2vFJVphPYm0xCwKdOrVta+fkgazk5anJM1Z9MC6hlnnFFZ3BOdNhddsR4dRl++fHll4VF8gVB85DN+jahuPs98H5ta+ULDedXBpnUoD5XzhOFHH3203Yow3hpp6dKl5lOWixYt8urdTIPNeHAlOcQVgEnH2KBiYObPdUV\/ffv2Jf5T5h\/7hEG\/aM\/mypUrzZurPd\/CwTtpSI99YmnscOyyZcto4MCB5lOnvFfr8sltdPKx4tPRpqeJJi7tRPH65d1g2SZdez55japzBPwm0xF+S9YL7c0vjnzbG3o22\/dsWnXtc9T+3fUFITvUPWzYMLPoh0cgL7\/88oqzbrvtNvrxj39sRkaZLZK2Popfw65YtyeJb40Ur6vN8fx5bf4T\/W3fvp2mT5\/u\/KKSX3SVz0odbFq444U755xzToftB1hi7vlcsmRJ5lB2\/K0j+pZjy3xgM97V7jPPM95IonUZM2aMd4+s1nDieTCXXXZZYdjklwWeImHPVxQ2OVFwIuEVhpxkisJmvH5FYdOer1F+hd9kylvYhN\/a62ZhE+0tO55821sWbJ70+khqeaucXxB6udNz9MBBN6mFq6xN4mWZggynjBo1ynSA8TNtzZo1tHr16sTThLpw2KWZOti0XxC64IILiPe7Spp7wW8P999\/v+nh5KF11y\/vMHrScRLYnD9\/PvXv379d9ULo2WRd+FOiRXs2uUGecsopxsd8vqKwydMtBgwYQFu3bqVJkyYVhs14\/YrCpj2fK15rVQ6\/yZS1sAm\/JcMm2lt2PPm2tyTY5NEZ7gXbuHGjLGiVWfPIHj8HOS9r+9UCNm2uSOrZZF8uXrxYLXzX2j\/qYJMXB3EQ8DA5f1ngj3\/8Y7th9D\/84Q9mkRD3enJXOC8acv3yLhCKd6HHr1PkawKuOmsuxxwymXd85y7Jziq3ht9kmsFvyXphFwG\/OPJtb0mwyVdg4OQ\/Zf4xZGoETda0FrCZ1WuJOZs8CVLZ77nnnqNLLrnEbOx+7LHH0iOPPEKf+tSn6JlnnjFfFjryyCOJJ\/YeddRRXjVPGy736aWMX8DnmNCDyjeJpjkHn6v0CtuqG8FvMkkBm4BNWcS0t\/Ztb2mwWeTaOLa+Cvg8831s6lvr+l5NXc+mvf3du3cTL\/j4wQ9+UPls5Tve8Q769Kc\/bb4iEB+edslmJw\/bNw\/fleiAzY7K+iZRwObbCgBaAC2u\/JRVjvYmU69s7c3CZtKCUtmdw7pRCvgs\/gFsKuzZjAYMV4+\/kf7mm28Sf6JSspl7PPBcq9sYSHnhkf3KUFLgomfz75Ohi87ZtOBv\/VJ0zibP5eXV6Dxnk+eAFl0gFK9f0TmbjZ4YDmiRPYrKBi14uSvny13nV\/bQ8VvW0o7Nb38DHL9yKuCanwrYVAabvECI99L82Mc+ZsChCFw2KmRDDypAiyyyAC3o2ZRFTHtrtDeZemVsbwycnV95vt2NdtnzBHV9Yr1ZYMMjeXaByfQjd1PfA\/fLRCEinmu7ZmdP87GU7t27m2lpN998M\/3rqDbq13H\/cuf5eYu4W37aqUP9zn7tEurRdpjz+LjBzs6P02MHrqucT3yCBh\/gmp8aOhe45Fc3jG5Xo2\/atMnMyRwxYgQNHz6c3vOe91CnTp1c96OiPPSgwsNPFmZlfPgl3SHm2sr8Xi1rtDeZkmhveLmTRUx9rEPnApeK6mCTK8wr0nk1Og9Z\/+QnPzHD6KeeeqoBT+7x7Nmzp+u+GloeelDh4ScLLzz88PCTRQx6NnnPQkxbeZK6PzSvslUOphsVaUWNPzZ0LnAprBI2o5WOgyd\/Yeiss86iL3zhC+a\/PlsfuUSodnnoQQXYlEUMYBOwKYsYwCZgkwgjCUVajb5jQ+cCl+LqYTN6A3\/729\/M56Z426NevXp5f0HIJUK1y0MPKsCmLGIAm4BNWcQANgGbgM0ibUbjsaFzgUtz9bDJPZn8jVmGzA0bNpj7Offcc82nofi72ujZdLm4+uWATZmmgE3ApixiAJuATcBmkTaj8VjAprLV6BwkccB84403zJzN0aNH09lnn009evTQGEuVOoUeVIBNWfgBNgGbsogBbAI2AZtF2ozGY0PnApfm6no27dd+Wltb6bjjjqMxY8aYnsw+ffq47kVNeehBBdiUhRpgE7ApixjAJmATsFmkzWg8NnQucGmuDjZ55fm9995LZ555JvXr16802x1FhQ49qACbrmbVvhywCdiURQxgE7AJ2CzSZjQeGzoXuDRXB5uuCvMQO38f\/Wc\/+xnNnj3bLBTS9gs9qACbsogDbAI2ZRED2ARsAjaLtBmNx4bOBS7NSwGbPK302WefpTvuuMPsOca9n0OGDMFqdJd3a1QO2JQJC9gEbMoiBrAJ2ARsFmkzGo8FbCpcIGQDhffYfOSRR2jp0qXEjuJfS0sLfeUrXzFfFdK6UCj0oAJsylIZYBOwKYsYwCZgE7BZpM1oPDZ0LnBprrJnc\/fu3eb76KtXr6adO3dW7mHhwoU0bNgwldsdRYUOPagAm65m1b4csAnYlEUMYBOwCdgs0mY0Hhs6F7g0VwOb3IvJ+2muXLmS7r\/\/fvPJSrvd0fvf\/376+te\/TjNnzjR7a2r\/hR5UgE1ZBAI2AZuyiAFsAjYBm0XajMZjQ+cCl+YqYPPFF1+kSZMmmSFzBsvPfOYz9MlPfrKy3ZHdDgmw6XJnfcoBmzKdAZuATVnEADYBm4DNIm1G47GATQVzNi1Mvvbaa\/TZz36WzjvvPOrfvz917tzZxAxgU1fTAWzK\/AHYBGzKIgawCdgEbBZpMxqPBWwqgM233nqLnnrqKfrRj35UWW1+1FFHGfDkhUAHHXQQXXzxxRhGV9KCAJsyRwA2AZuyiAFsAjYBm0XajMZjAZsKYDMaGLyP5q9\/\/Wu6\/fbbzX6aPHfz8MMPp127dtFtt91GZ511lsY4alen0IMKsCkLQcAmYFMWMYBNwCZgs0ib0Xhs6Fzg0lzFnM20SsZXpXfp0oU++tGP0rhx4+iUU05Ruyo99KACbLqaVftywCZgUxYxgE3AJmCzSJvReGzoXODSXDVs2srzMPuTTz5ptkLiLZHeeOMNOu2002jZsmXUs2dP1z3WvTz0oAJsykIKsAnYlEUMYBOwCdgs0mY0Hhs6F7g0LwVsRm+Cvx70wAMP0C9\/+Ut8rtLl3RqVAzZlwgI2AZuyiAFsAjYBm0XajMZjAZvK5mxqDBJpnUIPKsCmLCIAm4BNWcQANgGbgM0ibUbjsaFzgUvz0vVsum5IQ3noQQXYlEUZYBOwKYsYwCZgE7BZpM1oPDZ0LnBpDth0KZSjPPSgAmzKggKwCdiURQxgE7AJ2CzSZjQeGzoXuDQHbLoUylEeelABNmVBAdgEbMoiBrAJ2ARsFmkzGo8NnQtcmgM2XQrlKA89qACbsqAAbAI2ZRED2ARsAjaLtBmNx4bOBS7NAZsuhXKUhx5UgE1ZUAA2AZuyiAFsAjYBm0XajMZjQ+cCl+aATZdCOcpDDyrApiwoAJuATVnEADYBm4DNIm1G47Ghc4FLc8CmS6Ec5aEHFWBTFhSATcCmLGIAm4BNwGaRNqPx2NC5wKU5YNOlUI7y0IMKsCkLCsAmYFMWMYBNwCZgs0ib0Xhs6Fzg0hyw6VIoR3noQQXYlAUFYBOwKYsYwCZgE7BZpM1oPDZ0LnBpDth0KZSjPPSgAmzKggKwCdiURQxgE7AJ2CzSZjQeGzoXuDQHbLoUylEeelABNmVBAdgEbMoiBrAJ2ARsFmkzGo8NnQtcmgM2XQrlKA89qACbsqAAbAI2ZRED2ARsAjaLtBmNx4bOBS7NAZsuhXKUhx5UgE1ZUAA2AZuyiAFsAjYBm0XajMZjQ+cCl+aATZdCOcpDDyrApiwoAJuATVnEADYBm4DNIm1G47Ghc4FLc8CmS6Ec5aEHFWBTFhSATcCmLGIAm4BNwGaRNqPx2NC5wKU5YNOlUI7y0IMKsCkLCsAmYFMWMYBNwCZgs0ib0Xhs6Fzg0hyw6VIoR3noQQXYlAUFYBOwKYsYwCZgE7BZpM1oPDZ0LnBpDth0KZSjPPSgAmzKggKwCdiURQxgE7AJ2CzSZjQeGzoXuDQHbLoUylEeelABNmVBAdgEbMoiBrAJ2ARsFmkzGo8NnQtcmgM2XQrlKA89qACbsqAAbAI2ZRED2ARsAjaLtBmNx4bOBS7NAZsuhXKUhx5UgE1ZUAA2AZuyiAFsAjYBm0XajMZjQ+cCl+aATZdCOcpDDyrApiwoAJuATVnEADYBm4DNIm1G47Ghc4FLc8CmS6Ec5aEHFWBTFhSATcCmLGIAm4BNwGaRNqPx2NC5wKU5YNOlUI7y0IMKsCkLCsAmYFMWMYBNwCZgs0ib0Xhs6Fzg0hyw6VIoR3noQQXYlAUFYBOwKYsYwCZgE7BZpM1oPDZ0LnBpDth0KZSjPPSgAmzKggKwCdiURQxgE7AJ2CzSZjQeGzoXuDQHbLoUylEeelABNmVBAdgEbMoiBrAJ2ARsFmkzGo8NnQtcmgM2XQrlKA89qACbsqAAbAI2ZRED2ARsAjaLtBmNx4bOBS7NAZsuhXKUhx5UgE1ZUAA2AZuyiAFsAjYBm0XajMZjQ+cCl+aATZdCOcpDDyrApiwoAJuATVnEADYBm4DNIm1G47Ghc4FLc8CmS6Ec5aEHFWBTFhSATcCmLGIAm4BNwGaRNqPx2NC5wKV5U8GmdbYVZe7cuTRixIhMjV599VWaNWsWbdiwoWLnOi70oAJsuppV+3LAJmBTFjGATcAmYLNIm9F4bOhc4NK8aWCTHT1t2jRatWoVDRo0iJ566ikaO3YsTZkyJRU4X3jhBRo\/fjwdccQRdP3111O3bt0qxw0bNoxmzpyZqG\/oQQXYdDUrwKaPQq0vd6Wpm1to5cqVNHDgQNq6dSuNGzeOlk9uo5OP9TlDe5tNTxNNXNqJ1q5dS6effjrZOD1\/3xxqefME8QnxkoCXBHHQRA5AnpSpp6W9yWrtbx06F7iUaArYtNDID6AoIN555520bt06WrFiBfXq1auDVnFAtQY+x\/GbuX3ouZxQtnIkUZnHtCRR+A1+i0L4wsE7aUiPfTJRiAgvCX6Sob356WSttORJWa39rQGbbW1t\/nKV09L2Yi5YsMD0eNhf2r+77pJhc8mSJZVe0rh96EGFJOqKEPRs+igEaPFRiSo9tHvPnEH7ex\/nd1DEqsueJ6n7Q\/M69PgCNsVSig5AnhTJRYBNmV5ls26Kns20HkqfofQkh95www3mAZDVI4qezfSmgIdfY9IEHn4y3bU8\/OA3+A090rIY0GgdeieUS\/Omhk07vD5y5EjnQiErpA2YrEVC1ubSSy+loUOHtvNB3759if+U+ffoo4\/S6NGjqWhPC8\/V4yRqz1e0p2XZsmVm7t+2bdtowoQJhef+xetXdO6fPV+jfA+\/yZS3sAm\/tdfN9kijvWXHE9pbOdubrNbJ1rt27SL+E\/1t376dpk+fHuz0OpdugM3x48kXNm1P6Mknn1xZMJQkcHzVe9RmzJgxBtTK\/GttbaXLLrusMGwuWrSIhgwZQvZ8RWGTXwD69etHO3bsoMsvv7wwbMbrVxQ27fka5Xv4Taa8hU34LRk20d6y4wntrZztTVbrZOs1a9bQ6tWrEwtDXcvh0q2pYVMyjO4Lmiy4hc358+dT\/\/79g+vZ5PvjVcNFeza5QZ5yyilGLz5fUdi85ZZbaMCAAWZV86RJkwrDZrx+RWHTns\/VKGtVDr\/JlLWwCb8lwybaW3Y8ob2Vs73Jau3fs7lx40ZavHgxejarIbDWcxRdIGThkbc7slsgZd1r6HMzMIdMFumY+5esFxYI+cUR2pufTtYK7Q3tTRYx9bEOnQtcKjZFz2ba3EzXFkbRXsqJEyem7qsZFzn0oMLDz9Ws2pfj4YeHnyxi2lujvcnUQ3tDe5NFTH2sQ+cCl4pNAZssAoMlz+Oz8yV8htCtTdYG7kkChx5UePi5mhVg00ch9Gz6qIStj\/xU+rsVYBOwKY2ZetiHzgUuDZsGNqO9lFaU+Iry+P6ZvMXR8uXLUzVMm+gbelABNl3NCrDpoxBg00clwKafSoBNl05oby6FalseOhe41Gsq2HSJUa3y0IMKsCmLFPS0oKdFFjHtrdHeZOqhvaG9ySKmPtahc4FLRcCmS6Ec5aEHFR5+sqDAww8PP1nEADb5oxhFd39o9BYzyJOyqNeSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbvkQji4Ei1mhvMvW05ElZrf2tQ+cClxKATZdCOcpDDyokUVlQaEmi8Bv8BtiUxUARa7Q3mXpa8qSs1v7WoXOBSwnApkuhHOWhBxWSqCwotCRR+A1+A2zKYqCINdqbTD0teVJWa3\/r0LnApQRg06VQjvLQgwpJVBYUWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUgKw6VIoR3noQYUkKgsKLUkUfoPfAJuyGChijfYmU09LnpTV2t86dC5wKQHYdCmUozz0oEISlQWFliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lABsuhTKUR56UCGJyoJCSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbYFMWA0Ws0d5k6mnJk7Ja+1uHzgUuJQCbLoVylIceVEiisqDQkkThN\/gNsCmLgSLWaG8y9bTkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/wG2JTFQBFrtDeZelrypKzW\/tahc4FLCcCmS6Ec5aEHFZKoLCi0JFH4DX4DbMpioIg12ptMPS15UlZrf+vQucClBGDTpVCO8tCDCklUFhRakij8Br8BNmUxUMQa7U2mnpY8Kau1v3XoXOBSArDpUihHeehBhSQqCwotSRR+g98Am7IYKGKN9iZTT0uelNXa3zp0LnApAdh0KZSjPPSgQhKVBYWWJAq\/wW+ATVkMFLFGe5OppyVPymrtbx06F7iUAGy6FMpRHnpQIYnKgkJLEoXf4DfApiwGilijvcnU05InZbX2tw6dC1xKADZdCuUoDz2okERlQaElicJv8BtgUxYDRazR3mTqacmTslr7W4fOBS4lAJsuhXKUhx5USKKyoNCSROE3+A2wKYuBItZobzL1tORJWa39rUPnApcSgE2XQjnKQw8qJFFZUGhJovAb\/AbYlMVAEWu0N5l6WvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfgNsymKgiDXam0w9LXlSVmt\/69C5wKUEYNOlUI7y0IMKSVQWFFqSKPwGvwE2ZTFQxBrtTaaeljwpq7W\/dehc4FICsOlSKEd56EGFJCoLCi1JFH6D3wCbshgoYo32JlNPS56U1drfOnQucCkB2HQplKM89KBCEpUFhZYkCr\/Bb4BNWQwUsUZ7k6mnJU\/Kau1vHToXuJQAbLoUylEeelAhicqCQksShd\/gN8CmLAaKWKO9ydTTkidltfa3Dp0LXEoANl0K5SgPPaiQRGVBoSWJwm\/wG2BTFgNFrNHeZOppyZOyWvtbh84FLiUAmy6FcpSHHlRIorKg0JJE4Tf4DbApi4Ei1mhvMvW05ElZrf2tQ+cClxKATZdCOcpDDyokUVlQaEmi8Bv8BtiUxUARa7Q3mXpa8qSs1v7WoXOBSwnApkuhHOWhBxWSqCwotCRR+A1+A2zKYqCINdqbTD0teVJWa3\/r0LnApQRg06VQjvLQgwpJVBYUWpIo\/Aa\/ATZlMVDEGu1Npp6WPCmrtb916FzgUgKw6VIoR3noQYUkKgsKLUkUfoPfAJuyGChijfYmU09LnpTV2t86dC5wKQHYdCmUozz0oEISlQWFliQKv8FvgE1ZDBSxRnuTqaclT8pq7W8dOhe4lABsuhTKUR56UCGJyoJCSxKF3+A3wKYsBopYo73J1NOSJ2W19rcOnQtcSgA2XQrlKA89qJBEZUGhJYnCb\/AbYFMWA0Ws0d5k6mnJk7Ja+1uHzgUuJQCbLoVylIceVEiisqDQkkThN\/gNsCmLgSLWaG8y9bTkSVmt\/a1D5wKXEoBNl0I5ykMPKiRRWVBoSaLwG\/wG2JTFQBFrtDeZelrypKzW\/tahc4FLCcCmS6Ec5aEHFZKoLCi0JFH4DX4DbMpioIg12ptMPS15UlZrf+vQucClBGDTpVCO8tCDCklUFhRakij8Br8BNmUxUMQa7U2mnpY8Kau1v3XoXOBSArDpUihHeehBhSQqCwotSRR+g98Am7IYKGKN9iZTT0uelNXa3zp0LnApAdh0KZSjPPSgQhKVBYWWJAq\/wW+ATVkMFLFGe5OppyVPymrtbx06F7iUAGy6FMpRHnpQIYnKgkJLEoXf4DfApiwGilijvcnU05InZbX2tw6dC1xKADZdCuUoDz2okERlQaElicJv8BtgUxYDRazR3mTqacmTslr7W4fOBS4lAJsuhXKUhx5USKKyoNCSROE3+A2wKYuBItZobzL1tORJWa39rUPnApcSgE2XQjnKQw8qJFFZUGhJovAb\/AbYlMVAEWu0N5l6WvKkrNb+1qFzgUsJwKZLoRzloQcVkqgsKLQkUfgNfgNsymKgiDXam0w9LXlSVmt\/69C5wKUEYNOlUI7y0IMKSVQWFFqSKPwGvwE2ZTFQxBrtTaaeljwpq7W\/dehc4FICsOlSKEd56EGFJCoLCi1JFH6D3wCbshgoYo32JlNPS56U1drfOnQucCkB2HQplKM89KBCEpUFhZYkCr\/Bb4BNWQwUsUZ7k6mnJU\/Kau1vHToXuJQAbLoUylEeelAhicqCQksShd\/gN8CmLAaKWKO9ydTTkidltfa3Dp0LXEoANl0K5SgPPaiQRGVBoSWJwm\/wG2BTFgNFrNHeZOppyZOyWvtbh84FLiUAmy6FcpSHHlRIorKg0JJE4Tf4DbApi4Ei1mhvMvW05ElZrf2tQ+cClxKATZdCOcpDDyokUVlQaEmi8Bv8BtiUxUARa7Q3mXpa8qSs1v7WoXOBSwnApkuhHOWhBxWSqCwotCRR+A1+A2zKYqCINdqbTD0teVJWa3\/r0LnApQRg06UQEdkgsaZz586lESNGpB4ZelBpTaLslyeeeIKOP\/54uvzyy2n55DY6+VgPB8dMNj1NNHFpJ1q7di1FH87n75tDLW+eID6hliQKv8lcB78l69X6cleaurmF0N6y4wntrZztTVZrf+vQucClBGDToRAHyLRp02jVqlU0aNAgeuqpp2js2LE0ZcqUVOAMPai0JtFLL72UFi9eTPa\/gM32wQ2\/udJh+3LAZjZsor2VEzbhN1keqJZ16Fzg0gmwmaHQCy+8QOPHjze9WzNnzqxY3nnnnbRu3TpasWIF9erVq8MZQg8qQIurWQFafBSyPWR4+AFaMJLQMQa67HmSuj80r8MIy8LBO2lIj30+TaydDdqbWLKqHhA6F7jEAmxmKGR7MRcsWGCA0\/7S\/t2Whx5UgE1XswJs+iiEh5+PSn+fxrP3zBm0v\/dxfgdFrAAtYsmqcgDypExGLSMJslr7W4fOBS4lAJsZCsWH0OOwmTaUboOKe2yGDh3q8kHpyrdv307Tp0+nfccPp\/29jxfXv\/Mre+jgx1aa4W7Wx55vdMuLud7Yd73eheZv6UMXXHAB\/eAHP6j8d8I\/5ZuzueMFom+u7dShfie9PpJa3pLP2Xy503P0wEE3Vc4nFqxKB8BvMiHht2S90N784gjtzU8na6Wlvclq7W9t48GuBfA\/MgxLwGYO2LTD6yNHjkyct7lt2zYDYxs3bgwjSnAXUAAKQAEoAAWgQCEFuHNl\/vz5NGDAgELnKePBgM0awCafkoGT\/+AHBaAAFIACUAAKQAGGzGYETfY8YDMHbPqsSEezggJQAApAASgABaAAFABsZsZA3gVCCCwoAAWgABSAAlAACkCBtxVAz2ZGJKTNzXRtfYTgggJQAApAASgABaAAFABsesUAgyV\/jcauIMMQupdsMIICUAAKQAEoAAWgAHo2fWNA+rlK3\/PCDgpAASgABaAAFIACoSuAYfTQPYz7gwJQAApAASgABaBAAxUAbDZQfFwaCkABKAAFoAAUgAKhKwDYDN3DuD8oAAWgABSAAlAACjRQAcBmA8XHpaEAFIACUAAKQAEoELoCgM3QPVzj+7Or83fs2NHhSkOGDKEVK1ZQr169TFnat+aLVDFtL9Qi52yWY+3Ct7Rv9brKNehUi5jScF+1rsMNN9xg2mO0fUav6Sqvdf3KeH7WbPny5alVnzt3buLnjYveK\/vxV7\/6Fc2cObPoqWp2PNppzaQtzYkBm6Vxlc6KZm0FFX9g1SLhADbzx4ULJl3l+a+MIxutgAsmXeWNrr\/G62dpFt9Cr1r1f\/XVV2nWrFnUr18\/wGa1RMV5aqIAYLMmsjbPSbNgM14G2NQVFy6YdJXruhvURqKACyZd5ZJrNYttlmYWClmL66+\/nrp161YVWQCbVZERJ6mDEbhEoQAAERpJREFUAoDNOogc8iV8YHPBggV0+umnJw6jJw3Dx4ebbELdsGGDkZLf4letWkWDBg2ipJ5N24swbNgwk9hvuukm4mH+D37wg3Tttde2O8fzzz9Po0aNqrgoPqRsz2UNotfmf7Nfk+IhrGnTppnr8K9WQ2bVjCUXTCaV269qtba2VqoyceLESq+K9cfs2bPp1ltvJWvHNpdcconphbF+tP7hB6\/V8eKLL6bJkye388ehhx5KY8eOTdQ2+gIzYMCASi\/PkUceaT7GEI8Xe+L4fXD9+Mf+qyYMVNNf1TyXCyaTyl1twfqC2zsfH\/W9HeKNfpVty5YtlWHn+JSbat5rvc7lA5t\/+tOfOkwtiuafaJuw9U7Lf9wuxo8fX9E5KS9GpzdFc5KPr+z1o9MD2E\/cRq+77rpKDk76op71M+d99n1SR4Mrnrj84Ycfph49epiPqsRzb738iutURwHAZnV0bNqzuIbRWRj7oIknnCRQjAOOTVpHHHFEBQI4CS1ZssQkO\/4xiFigtQksmlhtsrRQFE3eceCx52WQTUqi8QdKHGwtOEW\/OqU1OKSwmfT51rj\/7d\/5nu0LQfSjCBbm4w+jJB2t36IgEq9zEmwyzFr\/W19HH\/K2jux7G5v2WkkPe63+K1IvKWz6tAXrmyR\/WX9EIT8eC9E2XuTeGnVslqZpeSyaJ5J6P135L\/qCZWPZJ6\/6+Cqp7djjouCXBzZ94snmhOjLbKN8i+sWVwCwWVzDpj5D1gIhFiaaKOKwyck53pMUHxZKSkrW5owzzqCTTjqpAptbt241vVnxXsW0XpooWHJdo0l68ODBptdg5MiR7Sb1xxN5FHwZUPmXBGUagyT+Zay0OlooSPIFHxP147Zt24w\/pkyZUtEtDpbRHhPr\/\/Xr11deIKyOWT2r1i9JsMnnj\/ZOJsVdfHFM0kNdo8+qVSfXYha+joVG\/n+ftpDkr3h7TouFtNiq1v3W4zxZsGn1TnvZsvWLv7y58t\/w4cM7zNn0yas+vkp7GeXz8wudfZmUwqbtkc2TW+vhR1yjNgoANmuja9OcNatnM97LmDVnMw4+FlKTEmdUXHv9Y445hh588MF2cJsENXauVBIkZi02ij+cswCsbLCZZzV6\/CXD9gha2LQ9zVH4tkNqabC5bt26xCHGaP3i2qYNo0dX5vrY1GpOndZEIO3ZjN5HWltIat9psBkHjVBgM201enwIOA3k4nq58p9rzmZaXvXxVZpP4rlTCpv2ZTKaB6K6uV5utbYp1CtbAcAmIqSQAlmwGe8t4jlcPK8xPt8yOs8x+qZu5\/jFe6qSYJPPcdZZZxngjMNTUsL2gc3onCL7sOBrx4ft45CU1ntTSOgaHJx3GD0+Fy+pZ1MbbPK8OI4L22MdB994D221FnDUwG1VOaUUNn3aQhbA2DbMcJTUS5rUHqtyo3U8SdY81\/hoi2tUITrHOSv\/JcFm\/EWQrx3vAfXxFc91j48AcF3ix+aBTZ94CuEFpI7hp\/5SgE31LtJdwSzYjPcWRWHTzjWKJ9Jo8pTAJsMND\/vxApT4OfPApl2UEh0O5vMmDaM3C2ymAUqZYNPGSHyrGPRsts8zUV\/zIrr41IiktuADMM0Gm6xT0jzyrFEe6wmfmIzDZtoxcTsfX\/HUlnhus\/fjmtuetUDIvrDnya26n4aoXZYCgE3ERyEFfHo2k1Yk2nk78R6m+OIN1zzBL33pSzRp0qTKAqGk+uSBTRbF9oZxHe3PPjh8htGTes8KiV3lgyU9m2kgH++91jqMHvVlEjQ345zNpF4rG2JRjTZv3uzVFnwAphlh04Lepk2bKqM6aVNt4v\/uyn9XX301XXPNNZV9NtNGVeJ51cdX3DmQlAN95mxmXc\/uAJK28weG0auc6JWcDrCpxBFlrYZrNXp0InnS3LnoKuHoKnE7ZzMJAqKQZHsgo8O28WSYBzbteaMrlqPDU80GmwzOcV05ZuOruMsAm1mr0Ztl5atkGN32bLragg\/ANCNsRnuB+\/TpU5mXHH9xte0p+hLgyn+2XdqFdnwOHt1x5VUfXyWdy+be6I4D8WdAUh6PXs\/2bLriCcPoZaWC5HoDNsPyZ93vJms1uutzlfH947jyDHH86bVowpXus2kTNJ+PP8fH+z3GV737zNmM3xvfD0Mtzzu1vZY+85Xq7hTPC0p6Nm3vbnxxCM8H41\/aVlRc5rsavZYLhOI9NGn7bHJ9NX\/2z9O1TjMJbPLnZn3agg\/ANCtsskNse4turxWfuynZZzO+a4PNn3YUwu5nm5RXubc6On+ebdKG4KNt3u6zyTk1\/ini6H6ht912G\/34xz+u9LjGY8MnngCbzmZcKgPAZqnchcpCAShQCwWi22mNGDGiFpfAOaFAEAowBPJm683w8YMgHKbkJgCbShyBakABKFAfBXz3Xa1PbXAVKKBTgaSRj6xpUzrvArXSogBgU4snUA8oAAXqokDS9I0QPpdYF\/FwkaZSILpFkb3xtH15m0oY3KxYAcCmWDIcAAWgABSAAlAACkABKOCrAGDTVynYQQEoAAWgABSAAlAACogVAGyKJcMBUAAKQAEoAAWgABSAAr4KADZ9lYIdFIACUAAKQAEoAAWggFgBwKZYMhwABaAAFIACUAAKQAEo4KsAYNNXKdhBASgABaAAFIACUAAKiBUAbIolwwFQAApAASgABaAAFIACvgoANn2Vgh0UgAI1V8BuGj1u3DjiP9GfLXvttddo1apVdMIJJ7QrX7lyJfEfLuNv248fP77yWdGiFbd7c\/br16+hn7Pkb17\/9Kc\/pS9+8YvUrVu3oreF46EAFIACdVEAsFkXmXERKAAFfBTYu3cvXXbZZdSjRw+67rrr6KCDDqocds8995jvOb\/55pt07bXXUvSzkq+\/\/jpdffXV5jvsixYtIv57iLDJXz\/asWMHPhXoE0ywgQJQQI0CgE01rkBFoAAUYAWWLl1K9957L61YsYJaWlqMKPv37zeA+dJLL5m\/M4Rec801dOCBB5q\/7969m770pS\/ReeedR5MnT666kFp6NgGbVXctTggFoEAdFABs1kFkXAIKQAF\/BR566CGaNGkSrVmzhj70oQ+ZA\/fs2UMTJkyo9GauXr26HYz+5je\/oYsuuoiWL19uhs65hzPas2lhkeH1wx\/+MC1YsIB+\/\/vf01FHHUVf\/epXafjw4dSlSxdzrba2NnrggQcqNny+2bNn06233krxYXSGXK7nunXr6Pnnn6f3v\/\/9pp6f+MQnzPni9eLzMzBbIGawPuSQQ8x1+VvUl156qbmv+BSBpE9szp07l3hKwS233GKmDgwaNKgisu0hPvbYY2nKlCk0a9YsA+5879dffz0988wzRqepU6fSiSeeSJ06daoc+7e\/\/c3cE09JSLonf0\/CEgpAASjwtgKATUQCFIACqhR49tlnaezYsXTxxRdX4JKhjaHpu9\/9rqkrg+TixYsNMPGPv+EcBdA02Ny0aZOBu3\/5l38x8zr\/7d\/+jR588EEDkh\/5yEfMubhX9Wtf+xp99KMfpZEjR9L\/\/u\/\/GhsGuLPPPrsyZ\/O5556jSy65hLZt20Zf\/vKXDYjysevXrzewzPX9y1\/+YuDznHPOqQCmnXvKUMuQOHjwYHNdBs9HHnnE\/NcCqHUM9+wyHN92220GvFkbhst9+\/YZrfha0WkFjz\/+uKkTa\/S+973PwCbf+zve8Q4z35Pryt+4bm1trQA6X4vhmacq8NxQ7ikeMGAA\/frXvzb15PPxdSyUqwoaVAYKQAHVCgA2VbsHlYMCzaeA7cV797vfTVdddZWBG+5lu\/\/++w2I8Y97Bk877TTzXztfk\/\/dDq1nwSb3HB533HHmPLbHdOjQoQYiX3zxRdPTyb2A3\/zmN6l79+7G7le\/+pW51he+8AVjx6DIdWHIjZ7vrbfeou9973sG4Pi\/DIQ8\/M\/QyT2KvKiH557yv\/Hc02984xt0\/vnnG5Dluaof+MAHzHWiPY3RCIgPo3PPJmvE\/7XnZ3vW6z\/\/8z\/p5ptvpq5duxrYZP3472eeeaY55csvv2x6Njt37mx6cfleuUeT74nrz6Bpf7\/4xS\/MdRh23\/ve9zZfUOKOoQAUKKQAYLOQfDgYCkCBWigQhUvujWNY6t+\/P02fPt1cbv78+fT000+bxUDcu2eH2G3vXhpsMgwylL3zne8057Fgy\/9vh5e5p5Chjnsj7c\/C4DHHHGNg0w6F8zC8BWJru2XLFtPzyr2nvKKe4fLGG280vYMMcFx3rjPX5eCDDzb3xPfCPYl8Xdtbm6Rr0pzNDRs2mOPsULqt26mnnmrAla\/F+tl7jK5iZ7D8zne+Y8D4sMMOM3avvPIKjR49mg444IBKFbZv326AlHs9oz2otfA9zgkFoEB4CgA2w\/Mp7ggKlF6B\/\/7v\/zY9fdxryHDEAMjzJi0Ack8brz5nwOJePZ7ryEPGdq5jGmzGgSsOmzyszEPF8a2VbO9pz549DWza8w8bNqzDFk3xa9thc+515d5Yvq9PfepTpjfz5z\/\/uQFm7jnlXsfooihf2LRwa4Gbpxzw\/\/NcTp7zmrW4ieeJjho1ygyp83A+QzJrkPbjXl\/uDcUPCkABKCBRALApUQu2UAAK1EWBnTt3GvDhnjT+8TZIDICHH364+bud18kA+te\/\/tXAEsMVwyD\/qg2bdjU890RKYTO6WOeCCy4w98Q9kbwQh8\/Fw\/G33367geboCntf2LQgzD2lPDzP80+jcz9dsMm9mHx97rWt5nZRdQkUXAQKQIFSKADYLIWbUEko0FwKWIDq06eP2eaIewejcxItQB199NEGLPkXHc7OC5u8Srtaw+i8iIcXGPGPgfJ3v\/sd\/fM\/\/zPdfffd5u8Mlzx0ztfjuZIMejx\/M+uXtvUR9\/R+61vfMr2kCxcuNIub7Kb4VivuIY7DLA+jM5wyyDOoc68r\/\/g8dr5qc0Ue7hYKQIFaKADYrIWqOCcUgAKFFWAQ4vmI\/OPh8\/gXhewiGF7owiulo6CWFza5B5PnOfJqcN5ayAIXb8fEQ8g85OyzQIiH9BkghwwZYurPw9VXXHEFvec97zGLhhiMeYEQ98xu3rzZzJPklfbR7Yt8ezbZbteuXTRx4kQ66aSTzMr5JUuWVBbyWNjk1ejRxUy8GIqnH\/Tt29f0iDLUc515TinDsF2dz+fnYX6+b+5hjs5lLexknAAKQIGmUACw2RRuxk1CgfIpwHMPeYiXf9E9N+2d2LmJPLTNq6SjoJYXNrn3z4LlKaecQp\/73OfMF3uWLVtm9pxkoGPo4p\/P1kd2myA7LeCJJ54wPY+f\/vSnzTkYmOfMmUP\/+I\/\/6NWbyBD4\/e9\/3ywq4hX0dloBQzL3evJ+n\/\/wD\/9QWV3O14ju0Tlw4EDTm9qrVy+zKIhX4\/O92e2X7D1ZbVmDxx57zExT4G2foiv0yxdRqDEUgAKNUgCw2SjlcV0oAAUyFbDbErERz8fs3bt3O3v71SDuLYwP+xaBTd7WiIe8GQq5R5KBbsyYMfTwww+bTeAtbHJlkjZ15wVG3PsX3Y\/STgvgBUHRxUd8fgZq3q\/T58tHf\/zjH+nKK6809Yov1uHeRx6653IL6VHY5DmdvO0Ra8l7g3784x830HrEEUe005V7PBmC7Ub1vAsA9+jyOe0qfoQuFIACUECiAGBTohZsoQAUgAJKFWAAZRCO9\/LGV9xHtz5SeiuoFhSAAoEpANgMzKG4HSgABZpPAR5G5\/mUvPLdzr+0KgA2my8ecMdQQJsCgE1tHkF9oAAUgAKeCjBc3nXXXeYb7Pw99+gXggCbniLCDApAgZorANisucS4ABSAAlCgNgrYT05u3LjRDKFfeOGFHb5djp7N2miPs0IBKOCvwP8DhJH5s+4UHB4AAAAASUVORK5CYII=","height":0,"width":0}}
%---
