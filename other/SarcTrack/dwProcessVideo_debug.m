function debug = dwProcessVideo_debug(path, roi)

% Debug + relaxed version of dwProcessVideo.
%
% Usage:
%   debug = dwProcessVideo_debug('VasHO 1.avi');
%   debug = dwProcessVideo_debug('VasHO 1.avi', [40 230 80 430]);
%
% roi format:
%   [r1 r2 c1 c2]
%
% Changes from the original:
%   1) Keeps detailed debug outputs
%   2) Relaxes the position filter
%   3) Relaxes the intensity filter
%   4) Angle filter can be disabled for troubleshooting
%   5) Fixes the prms row indexing bug in DWStats output
%   6) Supports optional cropping to a better ROI
%
% Debug outputs written next to the video:
%   *_DWDetectCounts.csv
%   *_DWTrackMetrics.csv
%   *_DWFilterFlags.csv
%   *_DWDebugSummary.txt

if nargin < 2
    roi = [];
end

if nargout == 0
    clear debug
end

tic

%% normalize input path
path = char(path);
path = strrep(path, '\', '/');

if ismac && startsWith(path, 'Users/')
    path = ['/' path];
end

if ~startsWith(path, '/')
    path = fullfile(pwd, path);
end

if ~exist(path, 'file')
    error('Video file not found: %s', path);
end

%% parameters
ds = 22:1:30;
stretch = 2;
scale = 2;
nangs = 4;

trackWindow = 7;
rmseThr = 1.5;

positionVarianceThr = 5;   % relaxed from 4
intensityPercentile = 10;  % relaxed from 20
enableAngleFilter = false; % disabled first for debugging

ctrFramesRange = [3 7];
rlxFramesRange = [4 8];

showDetectionPreview = false;
saveDetectionOverlay = true;

%% read frames
disp('reading frames')

v = VideoReader(path);
fps = v.FrameRate;
frameMs = 1000 / fps;

nFramesEst = max(1, round(v.Duration * v.FrameRate));

if isempty(roi)
    useCrop = false;
    r1 = 1; r2 = v.Height; c1 = 1; c2 = v.Width;
else
    useCrop = true;
    roi = round(roi(:)');
    if numel(roi) ~= 4
        error('roi must be [r1 r2 c1 c2]');
    end
    r1 = max(1, roi(1));
    r2 = min(v.Height, roi(2));
    c1 = max(1, roi(3));
    c2 = min(v.Width, roi(4));
    if r2 <= r1 || c2 <= c1
        error('Invalid roi after clipping to image bounds.');
    end
end

hUse = r2 - r1 + 1;
wUse = c2 - c1 + 1;

S = zeros(hUse, wUse, nFramesEst);

count = 0;
progressStep = max(1, round(nFramesEst / 10));

while hasFrame(v)
    if mod(count + 1, progressStep) == 1
        fprintf('.')
    end

    count = count + 1;
    frame = readFrame(v);

    if ndims(frame) == 3
        I0 = double(rgb2gray(frame)) / 255;
    else
        I0 = double(frame) / 255;
    end

    if useCrop
        I = I0(r1:r2, c1:c2);
    else
        I = I0;
    end

    if count > size(S, 3)
        S(:, :, end + 100) = 0;
    end

    S(:, :, count) = I;
end
fprintf('\n')

S = S(:, :, 1:count);
nFrames = count;

if nFrames == 0
    error('No frames were read from the video.');
end

%% initialize debug containers
[rpath, fname, ~] = fileparts(path);
if isempty(rpath)
    rpath = pwd;
end

detectCounts = zeros(nFrames, 1);
detectMeanSp = NaN(nFrames, 1);
detectMeanAngle = NaN(nFrames, 1);

%% initialize tracks
disp('initialize tracks')

I = normalize(S(:, :, 1));
[rs, cs, as0, sp0, ~, K, imDA, W] = imFindSarcomeres(I, ds, nangs, stretch, scale);

detLines = cell(1, nFrames);
detLines{1} = struct('rs', rs, 'cs', cs, 'as', as0, 'sp', sp0);

detectCounts(1) = numel(rs);
if ~isempty(sp0)
    detectMeanSp(1) = mean(sp0, 'omitnan');
end
if ~isempty(as0)
    detectMeanAngle(1) = mean(as0, 'omitnan');
end

fprintf('initial sarcomeres detected: %d\n', length(rs));

%% compute ridge-evidence volume
disp('compute ridge-evidence volume')

V = zeros(size(S));
cimDA = cell(1, nFrames);

V(:, :, 1) = K;
cimDA{1} = imDA;
fprintf('.')

progressStep = max(1, round(nFrames / 10));
for iFrame = 2:nFrames
    if mod(iFrame, progressStep) == 1
        fprintf('.')
    end

    I = normalize(S(:, :, iFrame));
    [rsDet, csDet, asDet, spDet, ~, V(:, :, iFrame), cimDA{iFrame}] = ...
        imFindSarcomeres(I, ds, nangs, stretch, scale, W);

    detLines{iFrame} = struct('rs', rsDet, 'cs', csDet, 'as', asDet, 'sp', spDet);

    detectCounts(iFrame) = numel(rsDet);
    if ~isempty(spDet)
        detectMeanSp(iFrame) = mean(spDet, 'omitnan');
    end
    if ~isempty(asDet)
        detectMeanAngle(iFrame) = mean(asDet, 'omitnan');
    end
end
fprintf('\n')

fprintf('detections/frame: min=%d, median=%g, max=%d\n', ...
    min(detectCounts), median(detectCounts), max(detectCounts));

%% track
disp('tracking via dynamic programming')

d = trackWindow;
x = rs;
y = cs;
tracks = zeros(2, nFrames, length(x));

progressStep = max(1, round(max(1, length(x)) / 10));
for i = 1:length(x)
    if mod(i, progressStep) == 1
        fprintf('.')
    end

    row = x(i);
    col = y(i);

    if row - d >= 1 && row + d <= size(I, 1) && col - d >= 1 && col + d <= size(I, 2)
        CV = V(row-d:row+d, col-d:col+d, :);

        CV = CV - min(CV(:));
        if max(CV(:)) > 0
            CV = CV / max(CV(:));
        end
        C = 1 - CV;

        ijPath = dpV(C, d + 1, d + 1);
        tracks(:, :, i) = ijPath - (d + 1) * ones(size(ijPath)) + ...
            repmat([row; col], [1 size(ijPath, 2)]);
    else
        tracks(:, :, i) = repmat([row; col], [1 nFrames]);
    end
end
fprintf('\n')

%% gather data
disp('gather data')

nTracks = size(tracks, 3);
rcasm = zeros(5, nFrames, nTracks);

for iFrame = 1:nFrames
    imDA = cimDA{iFrame};
    M = V(:, :, iFrame);

    rs1 = squeeze(tracks(1, iFrame, :))';
    cs1 = squeeze(tracks(2, iFrame, :))';
    as1 = zeros(1, length(rs1));
    ds1 = zeros(1, length(rs1));
    ms1 = zeros(1, length(rs1));

    for j = 1:length(rs1)
        rj = round(rs1(j));
        cj = round(cs1(j));

        if rj >= 1 && rj <= size(imDA, 1) && cj >= 1 && cj <= size(imDA, 2)
            dsIdx = imDA(rj, cj, 1);
            angIdx = imDA(rj, cj, 2);

            if dsIdx >= 1 && dsIdx <= length(ds)
                ds1(j) = ds(dsIdx);
            else
                ds1(j) = NaN;
            end

            as1(j) = (angIdx - 1) / nangs * pi;
            ms1(j) = M(rj, cj);
        else
            ds1(j) = NaN;
            as1(j) = NaN;
            ms1(j) = NaN;
        end
    end

    rcasm(1, iFrame, :) = rs1;
    rcasm(2, iFrame, :) = cs1;
    rcasm(3, iFrame, :) = as1;
    rcasm(4, iFrame, :) = ds1; 
    rcasm(5, iFrame, :) = ms1;
end

fprintf('tracks before filtering: %d\n', nTracks);

%% per-track metrics
meanR = squeeze(mean(rcasm(1, :, :), 2, 'omitnan'));
meanC = squeeze(mean(rcasm(2, :, :), 2, 'omitnan'));
meanM = squeeze(mean(rcasm(5, :, :), 2, 'omitnan'));
meanDs = squeeze(mean(rcasm(4, :, :), 2, 'omitnan'));
stdDs = squeeze(std(rcasm(4, :, :), 0, 2, 'omitnan'));

meanR = meanR(:);
meanC = meanC(:);
meanM = meanM(:);
meanDs = meanDs(:);
stdDs = stdDs(:);

trackMaxDev = NaN(nTracks, 1);
trackNAngleClusters = NaN(nTracks, 1);
trackHasNaNAngle = false(nTracks, 1);

%% filter tracks based on proximity
disp('filter tracks based on proximity')

idx2remP = [];
validTracks = isfinite(meanR) & isfinite(meanC) & isfinite(meanM);

if nnz(validTracks) >= 2
    validIdx = find(validTracks);
    XY = [meanR(validTracks), meanC(validTracks)];
    dist = squareform(pdist(XY));

    dist(~isfinite(dist)) = Inf;
    dist(triu(true(size(dist)))) = Inf;

    while true
        [minDist, linearIdx] = min(dist(:));

        if ~isfinite(minDist)
            break
        end

        [ii, jj] = ind2sub(size(dist), linearIdx);

        iOrig = validIdx(ii);
        jOrig = validIdx(jj);

        if minDist < 0.5 * mean(ds)
            if meanM(iOrig) < meanM(jOrig)
                idx2remP(end + 1) = iOrig;
                dist(ii, :) = Inf;
                dist(:, ii) = Inf;
            else
                idx2remP(end + 1) = jOrig;
                dist(jj, :) = Inf;
                dist(:, jj) = Inf;
            end
        else
            break
        end
    end
end

%% filter tracks based on intensity
disp('filter tracks based on intensity')

finiteMeanM = meanM(isfinite(meanM));
if isempty(finiteMeanM)
    idx2remI = [];
    thrI = NaN;
else
    thrI = prctile(finiteMeanM, intensityPercentile);
    idx2remI = find(~isfinite(meanM) | meanM < thrI)';
end

fprintf('intensity threshold (%dth percentile of meanM): %.6f\n', ...
    intensityPercentile, thrI);

%% filter tracks based on position variance
disp('filter tracks based on position variance')

circs = zeros(nTracks, 3);
for iTrack = 1:nTracks
    rr = squeeze(rcasm(1, :, iTrack));
    cc = squeeze(rcasm(2, :, iTrack));

    dr = diff(rr);
    dc = diff(cc);

    validStep = isfinite(dr) & isfinite(dc);
    if any(validStep)
        devs = hypot(dr(validStep), dc(validStep));
        maxDev = max(devs);
    else
        maxDev = 0;
    end

    trackMaxDev(iTrack) = maxDev;
    circs(iTrack, :) = [meanC(iTrack), meanR(iTrack), maxDev];
end

rads = circs(:, 3);
idx2remV = find(rads > positionVarianceThr | ~isfinite(rads))';

fprintf('position variance threshold (max step): %.3f\n', positionVarianceThr);

%% filter tracks based on angle variance
disp('filter tracks based on angle variance')

idx2remA = [];
idx2remPIV = unique([idx2remP idx2remI idx2remV]);
idx2keepTmp = 1:nTracks;
idx2keepTmp(idx2remPIV) = [];

if enableAngleFilter
    progressStep = max(1, round(max(1, length(idx2keepTmp)) / 10));
    for iSelTrack = 1:length(idx2keepTmp)
        if mod(iSelTrack, progressStep) == 1
            fprintf('.')
        end

        iTrack = idx2keepTmp(iSelTrack);
        as = squeeze(rcasm(3, :, iTrack));

        if any(~isfinite(as))
            trackHasNaNAngle(iTrack) = true;
            trackNAngleClusters(iTrack) = NaN;
            idx2remA(end + 1) = iTrack;
            continue
        end

        xang = cos(as);
        yang = sin(as);
        xy = [xang(:), yang(:)];

        k = 2;
        clusterProximityThreshold = 0.9;
        ignoreAngleSign = true;
        c = directionClustering(xy, k, clusterProximityThreshold, ignoreAngleSign);

        trackNAngleClusters(iTrack) = size(c, 1);

        if size(c, 1) > 1
            idx2remA(end + 1) = iTrack;
        end
    end
    fprintf('\n')
else
    fprintf('angle filter disabled\n')
    for iSelTrack = 1:length(idx2keepTmp)
        iTrack = idx2keepTmp(iSelTrack);
        as = squeeze(rcasm(3, :, iTrack));
        if any(~isfinite(as))
            trackHasNaNAngle(iTrack) = true;
        else
            trackNAngleClusters(iTrack) = 1;
        end
    end
end

%% aggregate removal flags
removeProximity = false(nTracks, 1);
removeIntensity = false(nTracks, 1);
removePosition = false(nTracks, 1);
removeAngle = false(nTracks, 1);

removeProximity(unique(idx2remP)) = true;
removeIntensity(unique(idx2remI)) = true;
removePosition(unique(idx2remV)) = true;
removeAngle(unique(idx2remA)) = true;

idx2rem = find(removeProximity | removeIntensity | removePosition | removeAngle)';
idx2keep = setdiff(1:nTracks, idx2rem);

%% print filtering summary
disp('tracks flagged for removal:')
fprintf('proximity: %d, intensity: %d, position variance: %d, angle variance: %d\n', ...
    nnz(removeProximity), nnz(removeIntensity), nnz(removePosition), nnz(removeAngle));

fprintf('tracks kept after filtering: %d\n', length(idx2keep));

if isempty(idx2keep)
    fprintf('WARNING: all tracks were removed before fitting.\n');
end

%% estimate frequency
disp('estimating frequency')

hasFrequencyFit = false;
a1 = NaN;
b1 = NaN;
c1 = NaN;
cro = [];
lb = [0 0 -pi/4];
ub = [pi pi pi/4];

fpd = struct('x', [], 'y2Fit', [], 'ySin', [], 'ySaw', [], 'ySawFit', []);

if ~isempty(idx2keep) && nFrames >= 4
    avgDsts = mean(rcasm(4, :, idx2keep), 3, 'omitnan')';
    x = (0:nFrames-1)';

    mx = prctile(avgDsts, 90);
    mn = prctile(avgDsts, 10);
    sd = std(avgDsts, 'omitnan');

    fprintf('avgDsts min=%.3f max=%.3f std=%.3f\n', min(avgDsts), max(avgDsts), sd);

    if isfinite(mx) && isfinite(mn) && mx > mn && sd > 0
        yAvg = (avgDsts - mean(avgDsts, 'omitnan')) / sd;
        y2Fit = 2 * ((avgDsts - mn) / (mx - mn) - 0.5);

        span = min(15, nFrames);
        syAvg = smooth(yAvg, span);

        try
            f = fit(x, syAvg, 'sin1');
            a1 = f.a1;
            b1 = f.b1;
            c1 = f.c1;
        catch
            y0 = syAvg - mean(syAvg, 'omitnan');
            Y = fft(y0);
            P2 = abs(Y / nFrames);
            P1 = P2(1:floor(nFrames/2) + 1);
            if numel(P1) > 2
                P1(2:end-1) = 2 * P1(2:end-1);
            end
            freqCpf = (0:floor(nFrames/2))' / nFrames;

            if numel(P1) >= 2
                [~, kMax] = max(P1(2:end));
                kMax = kMax + 1;
                f0 = freqCpf(kMax);

                if isfinite(f0) && f0 > 0
                    a1 = max(abs(syAvg));
                    b1 = 2 * pi * f0;
                    c1 = 0;
                end
            end
        end

        if isfinite(a1) && isfinite(b1) && isfinite(c1) && b1 > 0
            ySin = a1 * sin(b1 * x + c1);

            c0 = pi/2;
            r0 = pi/2;
            o0 = 0;
            ySaw = a1 * sawtooth(b1 * x + pi/2 + c1, c0, r0);

            f2m = @(cro_) -corr( ...
                a1 * sawtooth(b1 * x + cro_(3) + pi/2 + c1, cro_(1), cro_(2)), ...
                y2Fit, ...
                'rows', 'complete');

            cro0 = [c0; r0; o0];

            try
                cro = fmincon(f2m, cro0, [], [], [], [], lb, ub, [], ...
                    optimoptions('fmincon', 'Display', 'off'));
            catch
                cro = cro0;
            end

            ySawFit = a1 * sawtooth(b1 * x + cro(3) + pi/2 + c1, cro(1), cro(2));

            fpd.x = x;
            fpd.y2Fit = y2Fit;
            fpd.ySin = ySin;
            fpd.ySaw = ySaw;
            fpd.ySawFit = ySawFit;

            hasFrequencyFit = true;
        end
    end
end

fprintf('hasFrequencyFit = %d\n', hasFrequencyFit);

[fpdPath, fpdName, ~] = fileparts(path);
if isempty(fpdPath)
    fpdPath = pwd;
end

[mkOK, mkMsg] = mkdir(fpdPath);
if ~mkOK && ~exist(fpdPath, 'dir')
    error('Could not create output folder: %s\n%s', fpdPath, mkMsg);
end

save(fullfile(fpdPath, [fpdName '_fpd.mat']), 'fpd');

%% fit sawtooth curves
disp('fitting sawtooth curves')

prms = [];
idcs = [];
dsls = [];

if hasFrequencyFit
    progressStep = max(1, round(max(1, length(idx2keep)) / 10));

    for index = 1:length(idx2keep)
        if mod(index, progressStep) == 1
            fprintf('.')
        end

        x = (0:nFrames-1)';
        dsl = squeeze(rcasm(4, :, idx2keep(index)))';

        if any(~isfinite(dsl))
            continue
        end

        mx = prctile(dsl, 90);
        mn = prctile(dsl, 10);

        if mx > mn
            y = 2 * ((dsl - mn) / (mx - mn) - 0.5);

            prdFrames = 2 * pi / b1;

            cBounds = 2 * pi * ctrFramesRange / prdFrames;
            rBounds = 2 * pi * rlxFramesRange / prdFrames;

            lbTrack = [max(0, cBounds(1)), max(0, rBounds(1)), -pi/4];
            ubTrack = [min(pi, cBounds(2)), min(pi, rBounds(2)), pi/4];

            if lbTrack(1) >= ubTrack(1) || lbTrack(2) >= ubTrack(2)
                lbTrack = lb;
                ubTrack = ub;
            end

            f2m = @(prm) -corr( ...
                a1 * sawtooth(b1 * x + prm(3) + pi/2 + c1, prm(1), prm(2)), ...
                y, ...
                'rows', 'complete');

            options = optimoptions('fmincon', 'Display', 'off');

            prm0 = cro;
            if isempty(prm0)
                prm0 = [(lbTrack(1) + ubTrack(1))/2; (lbTrack(2) + ubTrack(2))/2; 0];
            end

            prm0 = prm0(:);
            prm0(1) = min(max(prm0(1), lbTrack(1)), ubTrack(1));
            prm0(2) = min(max(prm0(2), lbTrack(2)), ubTrack(2));
            prm0(3) = min(max(prm0(3), lbTrack(3)), ubTrack(3));

            try
                prm = fmincon(f2m, prm0, [], [], [], [], lbTrack, ubTrack, [], options);
            catch
                prm = prm0;
            end

            ySawFit = a1 * sawtooth(b1 * x + prm(3) + pi/2 + c1, prm(1), prm(2));
            z = ((ySawFit / 2) + 0.5) * (mx - mn) + mn;

            rmse = sqrt(sum((y - ySawFit).^2) / nFrames);

            cFrames = prm(1) / (2 * pi) * prdFrames;
            rFrames = prm(2) / (2 * pi) * prdFrames;
            cMs = cFrames * frameMs;
            rMs = rFrames * frameMs;

            if rmse < rmseThr && ...
               max(dsl) <= max(ds) && min(dsl) >= min(ds) && ...
               cFrames >= ctrFramesRange(1) && cFrames <= ctrFramesRange(2) && ...
               rFrames >= rlxFramesRange(1) && rFrames <= rlxFramesRange(2)

                prms = [prms [prm; min(dsl); max(dsl); min(z); max(z); cFrames; rFrames; cMs; rMs]];
                idcs = [idcs idx2keep(index)];
                dsls = [dsls dsl]; %#ok<AGROW>
            end
        end
    end
    fprintf('\n')
else
    fprintf('frequency fit skipped\n')
end

fprintf('accepted fitted tracks: %d\n', length(idcs));

%% write detection overlays
if saveDetectionOverlay
    disp('writing per-frame detection overlays')

    outDet = fullfile(rpath, [fname '_DWDetect']);
    [mkOK, mkMsg] = mkdir(outDet);
    if ~mkOK && ~exist(outDet, 'dir')
        error('Could not create detection output folder: %s\n%s', outDet, mkMsg);
    end

    progressStep = max(1, round(nFrames / 10));
    for iFrame = 1:nFrames
        if mod(iFrame, progressStep) == 1
            fprintf('.')
        end

        I = normalize(S(:, :, iFrame));
        D = detLines{iFrame};

        if isempty(D) || isempty(D.rs)
            Jdet = repmat(I, [1 1 3]);
        else
            Jdet = imDrawSarcomeresCB(repmat(I, [1 1 3]), D.rs, D.cs, D.as, D.sp, ds);
        end

        imwrite(Jdet, fullfile(outDet, sprintf('Frame%03d.png', iFrame)));

        if showDetectionPreview
            figure(200)
            imshow(Jdet)
            title(sprintf('Detected lines, frame %d / %d', iFrame, nFrames))
            drawnow
        end
    end
    fprintf('\n')
end

%% draw fitted outputs
disp('writing images')

outFit = fullfile(rpath, [fname '_DWFit']);
[mkOK, mkMsg] = mkdir(outFit);
if ~mkOK && ~exist(outFit, 'dir')
    error('Could not create image output folder: %s\n%s', outFit, mkMsg);
end

if isempty(idcs)
    stracks = zeros(2, nFrames, 0);
else
    stracks = tracks(:, :, idcs);
end

progressStep = max(1, round(nFrames / 10));
for iFrame = 1:nFrames
    if mod(iFrame, progressStep) == 1
        fprintf('.')
    end

    I = normalize(S(:, :, iFrame));
    imDA = cimDA{iFrame};

    rs1 = squeeze(stracks(1, iFrame, :))';
    cs1 = squeeze(stracks(2, iFrame, :))';
    as1 = zeros(1, length(rs1));
    ds1 = zeros(1, length(rs1));

    for j = 1:length(rs1)
        rj = round(rs1(j));
        cj = round(cs1(j));

        if rj >= 1 && rj <= size(imDA, 1) && cj >= 1 && cj <= size(imDA, 2)
            dsIdx = imDA(rj, cj, 1);
            angIdx = imDA(rj, cj, 2);

            if dsIdx >= 1 && dsIdx <= length(ds)
                ds1(j) = ds(dsIdx);
            else
                ds1(j) = NaN;
            end

            as1(j) = (angIdx - 1) / nangs * pi;
        else
            ds1(j) = NaN;
            as1(j) = NaN;
        end
    end

    if isempty(rs1)
        J = repmat(0.5 * I, [1 1 3]);
    else
        validDraw = isfinite(rs1) & isfinite(cs1) & isfinite(as1) & isfinite(ds1);
        J = imDrawSarcomeresCB( ...
            repmat(0.5 * I, [1 1 3]), ...
            rs1(validDraw), cs1(validDraw), as1(validDraw), ds1(validDraw), ds);
    end

    imwrite(J, fullfile(outFit, sprintf('Frame%03d.png', iFrame)));
end
fprintf('\n')

%% write tables
disp('writing tables')

outPathS = fullfile(rpath, [fname '_DWStats.csv']);
outPathD = fullfile(rpath, [fname '_DWDists.csv']);
outPathPF = fullfile(rpath, [fname '_DWPrdFrq.csv']);
outPathDetect = fullfile(rpath, [fname '_DWDetectCounts.csv']);
outPathTrackMetrics = fullfile(rpath, [fname '_DWTrackMetrics.csv']);
outPathFilterFlags = fullfile(rpath, [fname '_DWFilterFlags.csv']);
outPathSummary = fullfile(rpath, [fname '_DWDebugSummary.txt']);

if ~isempty(idx2keep)
    rawDsls = permute(rcasm(4, :, idx2keep), [2 3 1]);
    rawDsls = reshape(rawDsls, nFrames, length(idx2keep));

    frameCol = (1:nFrames)';
    timeMsCol = (0:nFrames-1)' * frameMs;

    T = array2table([frameCol timeMsCol rawDsls], ...
        'VariableNames', [{'frame', 'time_ms'}, ...
        arrayfun(@(k) sprintf('track%05d', k), idx2keep, 'UniformOutput', false)]);
    writetable(T, outPathD);
else
    writetable(array2table([]), outPathD);
end

if ~isempty(prms) && hasFrequencyFit
    prdFrames = 2 * pi / b1;
    freqPerFrame = 1 / prdFrames;
    prdSeconds = prdFrames / fps;
    freqHz = 1 / prdSeconds;

    % prms rows:
    % 1-3: prm
    % 4: min(dsl)
    % 5: max(dsl)
    % 6: min(z)
    % 7: max(z)
    % 8: cFrames
    % 9: rFrames
    % 10: cMs
    % 11: rMs
    cFrames = prms(8, :);
    rFrames = prms(9, :);
    cMs = prms(10, :);
    rMs = prms(11, :);
    oFrames = prms(3, :) / (2 * pi) * prdFrames;
    oMs = oFrames * frameMs;

    T = array2table([ ...
        cFrames' rFrames' oFrames' ...
        cMs' rMs' oMs' ...
        prms(4, :)' prms(5, :)' prms(6, :)' prms(7, :)' ], ...
        'VariableNames', { ...
        'contraction_time_frames', 'relaxation_time_frames', 'offset_frames', ...
        'contraction_time_ms', 'relaxation_time_ms', 'offset_ms', ...
        'min_ds', 'max_ds', 'min_ds_fit', 'max_ds_fit'});
    writetable(T, outPathS);

    T = array2table([prdFrames freqPerFrame prdSeconds freqHz], ...
        'VariableNames', {'period_frames', 'frequency_per_frame', ...
        'period_seconds', 'frequency_hz'});
    writetable(T, outPathPF);
else
    writetable(array2table([]), outPathS);
    writetable(array2table([]), outPathPF);
end

Tdetect = table((1:nFrames)', (0:nFrames-1)' * frameMs, detectCounts, detectMeanSp, detectMeanAngle, ...
    'VariableNames', {'frame', 'time_ms', 'n_detected', 'mean_sp', 'mean_angle_rad'});
writetable(Tdetect, outPathDetect);

Tmetrics = table((1:nTracks)', meanR, meanC, meanM, meanDs, stdDs, trackMaxDev, trackNAngleClusters, trackHasNaNAngle, ...
    'VariableNames', {'track_id', 'mean_row', 'mean_col', 'mean_magnitude', 'mean_ds', 'std_ds', 'max_step_dev', 'n_angle_clusters', 'has_nan_angle'});
writetable(Tmetrics, outPathTrackMetrics);

Tflags = table((1:nTracks)', removeProximity, removeIntensity, removePosition, removeAngle, ...
    (removeProximity | removeIntensity | removePosition | removeAngle), ismember((1:nTracks)', idx2keep(:)), ...
    'VariableNames', {'track_id', 'remove_proximity', 'remove_intensity', 'remove_position', 'remove_angle', 'removed_any', 'kept'});
writetable(Tflags, outPathFilterFlags);

fid = fopen(outPathSummary, 'w');
if fid ~= -1
    fprintf(fid, 'Video: %s\n', path);
    fprintf(fid, 'fps: %.6f\n', fps);
    fprintf(fid, 'nFrames: %d\n', nFrames);
    fprintf(fid, 'roi_used: %d\n', useCrop);
    fprintf(fid, 'roi_r1_r2_c1_c2: [%d %d %d %d]\n', r1, r2, c1, c2);
    fprintf(fid, 'initial_sarcomeres_detected: %d\n', numel(rs));
    fprintf(fid, 'detections_per_frame_min: %d\n', min(detectCounts));
    fprintf(fid, 'detections_per_frame_median: %.6f\n', median(detectCounts));
    fprintf(fid, 'detections_per_frame_max: %d\n', max(detectCounts));
    fprintf(fid, 'tracks_before_filtering: %d\n', nTracks);
    fprintf(fid, 'intensity_percentile: %d\n', intensityPercentile);
    fprintf(fid, 'intensity_threshold: %.12f\n', thrI);
    fprintf(fid, 'position_variance_threshold: %.12f\n', positionVarianceThr);
    fprintf(fid, 'angle_filter_enabled: %d\n', enableAngleFilter);
    fprintf(fid, 'removed_proximity: %d\n', nnz(removeProximity));
    fprintf(fid, 'removed_intensity: %d\n', nnz(removeIntensity));
    fprintf(fid, 'removed_position: %d\n', nnz(removePosition));
    fprintf(fid, 'removed_angle: %d\n', nnz(removeAngle));
    fprintf(fid, 'tracks_kept: %d\n', numel(idx2keep));
    fprintf(fid, 'hasFrequencyFit: %d\n', hasFrequencyFit);
    fprintf(fid, 'accepted_fitted_tracks: %d\n', numel(idcs));
    fclose(fid);
end

%% return debug struct
debug = struct();
debug.path = path;
debug.fps = fps;
debug.nFrames = nFrames;
debug.roi = [r1 r2 c1 c2];
debug.useCrop = useCrop;
debug.ds = ds;
debug.positionVarianceThr = positionVarianceThr;
debug.intensityPercentile = intensityPercentile;
debug.enableAngleFilter = enableAngleFilter;
debug.detectCounts = detectCounts;
debug.initialDetections = numel(rs);
debug.nTracksBeforeFiltering = nTracks;
debug.intensityThreshold = thrI;
debug.removeProximity = removeProximity;
debug.removeIntensity = removeIntensity;
debug.removePosition = removePosition;
debug.removeAngle = removeAngle;
debug.idx2keep = idx2keep;
debug.idx2rem = idx2rem;
debug.hasFrequencyFit = hasFrequencyFit;
debug.nAcceptedFits = numel(idcs);
debug.outDetectCounts = outPathDetect;
debug.outTrackMetrics = outPathTrackMetrics;
debug.outFilterFlags = outPathFilterFlags;
debug.outSummary = outPathSummary;

toc

end
