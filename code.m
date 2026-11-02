%% ========================================================================
%  AUDIO TAMPERING DETECTION - MACHINE LEARNING (NO REFERENCE FILE NEEDED)
%
%  IDEA:
%  Instead of comparing original vs tampered, we TRAIN a classifier on
%  forensic features extracted from short 1-second windows of audio.
%  A window label is 1 if it contains an edit point (cut / splice / insert /
%  replace / silence), else 0. At test time, ONE suspect file is split into
%  windows, each window gets a tampering probability, and the windows with
%  high probability localize the edit.
%
%  TRAINING DATA:
%  Put 30+ clean speech .wav files in the folder CLEAN_DIR (each >= 4 s).
%  The script CREATES the tampered versions automatically (see makeTamper),
%  so you get a labeled dataset without needing one from outside.
%
%  REQUIRES: Signal Processing Toolbox, Statistics and Machine Learning Toolbox
%
%  USAGE:
%   1) First run : RUN_TRAINING = true   -> trains, evaluates, saves model
%   2) Later runs: RUN_TRAINING = false  -> loads model, tests TEST_FILE
% ========================================================================
clear; clc; close all; rng(1);

%% ----------------------------- SETTINGS --------------------------------
RUN_TRAINING      = true;
CLEAN_DIR         = 'clean_audio';     % folder with clean .wav speech files
MODEL_FILE        = 'tamper_model.mat';
TEST_FILE         = 'suspect.wav';     % single file to analyse (no reference)

FS                = 16000;   % common sampling rate
WIN_SEC           = 1.0;     % window length (s)
HOP_SEC           = 0.5;     % window hop (s)
MAX_SEC           = 10;      % crop each training clip to this length
TAMPERED_PER_FILE = 3;       % tampered versions generated per clean file
TEST_RATIO        = 0.2;     % fraction of SOURCE FILES held out for testing
ADD_NOISE         = true;    % add random background noise (robustness)
DECISION_THRESH   = 0.5;     % probability threshold for "tampered"

%% ----------------------------- TRAINING --------------------------------
if RUN_TRAINING
    files = dir(fullfile(CLEAN_DIR, '*.wav'));
    assert(numel(files) >= 20, 'Put at least 20 clean .wav files in "%s".', CLEAN_DIR);

    sigs = cell(numel(files), 1);
    for k = 1:numel(files)
        sigs{k} = loadClean(fullfile(files(k).folder, files(k).name), FS, MAX_SEC);
    end
    sigs = sigs(cellfun(@(s) numel(s) >= 4*FS, sigs));   % keep clips >= 4 s
    nFiles = numel(sigs);
    fprintf('Using %d clean clips.\n', nFiles);

    X = []; y = []; gid = [];
    for k = 1:nFiles
        x = sigs{k};
        for v = 0:TAMPERED_PER_FILE          % v = 0 -> genuine version
            if v == 0
                s = x; edits = [];
            else
                j = randi(nFiles); while j == k, j = randi(nFiles); end
                [s, edits] = makeTamper(x, sigs{j}, FS);
            end
            if ADD_NOISE, s = addNoise(s); end
            [F, lab] = extractWindows(s, FS, WIN_SEC, HOP_SEC, edits);
            X   = [X; F];                    %#ok<AGROW>
            y   = [y; lab];                  %#ok<AGROW>
            gid = [gid; k*ones(size(lab))];  %#ok<AGROW>
        end
        if mod(k,10)==0, fprintf('  processed %d / %d clips\n', k, nFiles); end
    end
    fprintf('Total windows: %d  (tampered: %d, genuine: %d)\n', ...
            numel(y), sum(y==1), sum(y==0));

    % ---- split BY SOURCE FILE so no clip leaks between train and test ----
    perm      = randperm(nFiles);
    testFiles = perm(1:round(TEST_RATIO*nFiles));
    isTest    = ismember(gid, testFiles);
    Xtr = X(~isTest,:); ytr = y(~isTest);
    Xte = X(isTest,:);  yte = y(isTest);

    % ---- train Random Forest (bagged trees), balanced priors ----
    model = fitcensemble(Xtr, ytr, 'Method', 'Bag', 'NumLearningCycles', 200, ...
                         'Learners', templateTree('MaxNumSplits', 60), ...
                         'Prior', 'uniform');

    % ---- evaluate on held-out files ----
    [~, score] = predict(model, Xte);
    pTamp = score(:, model.ClassNames == 1);
    pred  = double(pTamp >= DECISION_THRESH);

    TP = sum(pred==1 & yte==1); TN = sum(pred==0 & yte==0);
    FP = sum(pred==1 & yte==0); FN = sum(pred==0 & yte==1);
    acc  = (TP+TN)/numel(yte);
    prec = TP/(TP+FP+eps);  rec = TP/(TP+FN+eps);
    f1   = 2*prec*rec/(prec+rec+eps);
    [fpr, tpr, ~, auc] = perfcurve(yte, pTamp, 1);

    fprintf('\n===== TEST RESULTS (held-out source files) =====\n');
    fprintf('Accuracy : %.2f %%\n', 100*acc);
    fprintf('Precision: %.2f %%\n', 100*prec);
    fprintf('Recall   : %.2f %%\n', 100*rec);
    fprintf('F1-score : %.3f\n', f1);
    fprintf('AUC      : %.3f\n', auc);

    figure('Name','Confusion Matrix','NumberTitle','off');
    confusionchart(yte, pred); title('Window-level Confusion Matrix (0=genuine, 1=tampered)');
    figure('Name','ROC Curve','NumberTitle','off');
    plot(fpr, tpr, 'LineWidth', 2); hold on; plot([0 1],[0 1],'k--'); grid on;
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve (AUC = %.3f)', auc));
    try
        imp = predictorImportance(model);
        figure('Name','Feature Importance','NumberTitle','off');
        bar(imp); grid on; xlabel('Feature index'); ylabel('Importance');
        title('Which features matter most');
    catch
    end

    save(MODEL_FILE, 'model');
    fprintf('\nModel saved to %s\n', MODEL_FILE);
else
    S = load(MODEL_FILE); model = S.model;
end

%% ------------------- TEST ONE FILE (NO REFERENCE) ----------------------
if exist(TEST_FILE, 'file')
    detectSingleFile(TEST_FILE, model, FS, WIN_SEC, HOP_SEC, DECISION_THRESH);
else
    fprintf('\n"%s" not found - put a suspect file there to test.\n', TEST_FILE);
end


%% ======================= LOCAL FUNCTIONS ================================

function detectSingleFile(fname, model, FS, W, H, thresh)
    x = loadClean(fname, FS, inf);
    [F, ~, t0] = extractWindows(x, FS, W, H, []);
    [~, score] = predict(model, F);
    p = score(:, model.ClassNames == 1);
    flagged = p >= thresh;
    clipScore = max(p);

    fprintf('\n================ SINGLE-FILE ANALYSIS ================\n');
    fprintf('File            : %s\n', fname);
    fprintf('Max tamper prob : %.3f\n', clipScore);
    if any(flagged)
        fprintf('VERDICT         : TAMPERED (suspicious windows below)\n');
        idx = find(flagged);
        for i = idx'
            fprintf('   %.2f s - %.2f s   (p = %.2f)\n', t0(i), t0(i)+W, p(i));
        end
    else
        fprintf('VERDICT         : No tampering detected\n');
    end

    figure('Name','Forensic Report','NumberTitle','off');
    ax1 = subplot(3,1,1);
    t = (0:numel(x)-1)/FS;
    plot(t, x); title('Waveform'); ylabel('Amplitude'); grid on;
    ax2 = subplot(3,1,2);
    spectrogram(x, hamming(400), 240, 512, FS, 'yaxis'); title('Spectrogram');
    ax3 = subplot(3,1,3);
    plot(t0 + W/2, p, 'k', 'LineWidth', 1.5); hold on;
    yline(thresh, '--r', 'Threshold'); ylim([0 1]); grid on;
    xlabel('Time (s)'); ylabel('P(tampered)'); title('Confidence Timeline');
    for i = find(flagged)'
        patch(ax3, [t0(i) t0(i)+W t0(i)+W t0(i)], [0 0 1 1], 'r', ...
              'FaceAlpha', 0.25, 'EdgeColor', 'none');
        xline(ax1, t0(i)+W/2, 'r', 'Alpha', 0.4);
    end
    linkaxes([ax1 ax3], 'x');
end

function x = loadClean(fname, FS, maxSec)
    [x, fs0] = audioread(fname);
    if size(x,2) > 1, x = mean(x,2); end
    if fs0 ~= FS, x = resample(x, FS, fs0); end
    if isfinite(maxSec), x = x(1:min(numel(x), round(maxSec*FS))); end
    x = x / (sqrt(mean(x.^2)) + eps) * 0.1;     % RMS-normalise loudness
end

function s = addNoise(s)
    snr = 15 + 15*rand;                          % 15-30 dB SNR
    s = s + randn(size(s)) * sqrt(mean(s.^2)) * 10^(-snr/20);
end

function [y, edits] = makeTamper(x, other, fs)
    % Random edit: 1=delete, 2=insert, 3=replace, 4=silence
    op = randi(4); L = numel(x);
    segLen = round((0.3 + 0.7*rand) * fs);
    s = randi([round(0.5*fs), L - segLen - round(0.5*fs)]);
    if numel(other) < segLen + 1, other = [other; zeros(segLen+1, 1)]; end
    st  = randi(numel(other) - segLen);
    ins = other(st : st+segLen-1);
    switch op
        case 1   % deletion
            y = [x(1:s); x(s+segLen+1:end)];
            edits = s/fs;
        case 2   % insertion
            y = [x(1:s); ins; x(s+1:end)];
            edits = [s, s+segLen]/fs;
        case 3   % replacement
            y = x; y(s+1:s+segLen) = ins;
            edits = [s, s+segLen]/fs;
        case 4   % silence / muting
            y = x; y(s+1:s+segLen) = 1e-4*randn(segLen,1);
            edits = [s, s+segLen]/fs;
    end
end

function [F, lab, t0] = extractWindows(x, fs, W, H, edits)
    wl = round(W*fs); hl = round(H*fs);
    n  = floor((numel(x) - wl)/hl) + 1;
    t0 = ((0:n-1)' * hl) / fs;
    lab = zeros(n,1);
    for e = edits(:)'
        lab(t0 + 0.05 <= e & e <= t0 + W - 0.05) = 1;
    end
    for i = 1:n
        f = windowFeatures(x((i-1)*hl + (1:wl)), fs);
        if i == 1, F = zeros(n, numel(f)); end
        F(i,:) = f;
    end
end

function f = windowFeatures(x, fs)
    % Forensic feature vector for ONE window (no reference needed)
    frameLen = round(0.025*fs); hop = round(0.010*fs); nfft = 512;
    nF  = floor((numel(x) - frameLen)/hop) + 1;
    win = hamming(frameLen);
    mag = zeros(nF, nfft/2+1); e = zeros(nF,1); zcr = zeros(nF,1);
    for i = 1:nF
        fr = x((i-1)*hop + (1:frameLen));
        e(i)   = 10*log10(mean(fr.^2) + 1e-10);
        zcr(i) = sum(abs(diff(sign(fr)))) / (2*frameLen);
        X = abs(fft(fr .* win, nfft));
        mag(i,:) = X(1:nfft/2+1)';
    end
    freqs = linspace(0, fs/2, nfft/2+1);

    % MFCCs
    melFB  = melFilterBank(26, nfft, fs);
    logMel = log(max(((mag.^2)/nfft) * melFB', eps));
    c = dct(logMel')';  c = c(:, 1:13);
    mj = sqrt(sum(diff(c).^2, 2));            % frame-to-frame MFCC jump

    % Energy discontinuity
    de = diff(e);

    % Spectral descriptors
    cent = (mag * freqs') ./ (sum(mag,2) + eps);
    flux = sum(diff(mag,1,1).^2, 2) ./ (sum(mag(2:end,:).^2, 2) + eps);
    hf   = sum(mag(:, freqs > 4000).^2, 2) ./ (sum(mag.^2, 2) + eps);

    % Background-noise consistency: compare noise floor of 1st vs 2nd half
    logmag = log(mag + 1e-8); h = floor(nF/2);
    n1 = prctile(logmag(1:h,:), 10, 1);
    n2 = prctile(logmag(h+1:end,:), 10, 1);
    lsd = mean(abs(n1 - n2));

    f = [mean(c), std(c), std(diff(c)), ...
         max(mj), mean(mj), max(mj)/(median(mj)+eps), ...
         max(abs(de)), std(de), min(e), prctile(e,10), max(e), ...
         mean(cent), std(cent), ...
         max(flux), mean(flux), max(flux)/(mean(flux)+eps), ...
         mean(zcr), std(zcr), mean(hf), std(hf), lsd];
end

function fb = melFilterBank(numFilters, nfft, fs)
    melPts = linspace(0, 2595*log10(1 + (fs/2)/700), numFilters+2);
    hzPts  = 700*(10.^(melPts/2595) - 1);
    bin    = floor((nfft+1) * hzPts / fs);
    fb = zeros(numFilters, nfft/2+1);
    for m = 2:numFilters+1
        l = bin(m-1); c = bin(m); r = bin(m+1);
        for k = l:c, if c > l, fb(m-1, k+1) = (k-l)/(c-l); end, end
        for k = c:r, if r > c, fb(m-1, k+1) = (r-k)/(r-c); end, end
    end
end
