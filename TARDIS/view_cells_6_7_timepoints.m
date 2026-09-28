function view_cells_6_7_timepoints(matFile)
%VIEW_CELLS_6_7_TIMEPOINTS Browse localization scatter plots for data{6} and data{7}.
%
% Usage:
%   view_cells_6_7_timepoints
%   view_cells_6_7_timepoints('20251016_mdk_Miro1_V7.mat')
%
% The left and right panels show the two cells at the same time point.
% Use the slider, left/right arrow keys, or spacebar to move through time.
% A second figure is also created with all time points overlaid; point color
% represents time so the full spatial footprint and time progression are visible.

if nargin < 1 || isempty(matFile)
    [fileName, folder] = uigetfile('*.mat', 'Choose a MATLAB data file');
    if isequal(fileName, 0)
        return
    end
    matFile = fullfile(folder, fileName);
end

loaded = load(matFile, 'data');
if ~isfield(loaded, 'data') || ~iscell(loaded.data) || numel(loaded.data) < 7
    error('The MAT file must contain a cell array named data with at least 7 entries.');
end

cellIndices = [6, 7];
records = loaded.data(cellIndices);
requiredFields = {'time', 'x_data', 'y_data'};
for recordIndex = 1:numel(records)
    if ~isstruct(records{recordIndex}) || ...
            ~all(isfield(records{recordIndex}, requiredFields))
        error('data{%d} must be a struct with time, x_data, and y_data fields.', ...
            cellIndices(recordIndex));
    end
end

allTimes = unique([records{1}.time(:); records{2}.time(:)]);
if isempty(allTimes)
    error('data{6} and data{7} contain no time points.');
end

allX = [records{1}.x_data(:); records{2}.x_data(:)];
allY = [records{1}.y_data(:); records{2}.y_data(:)];
validCoordinates = isfinite(allX) & isfinite(allY);
allX = allX(validCoordinates);
allY = allY(validCoordinates);
if isempty(allX)
    error('No finite x/y localizations were found in data{6} and data{7}.');
end

[xLimits, yLimits] = sharedLimits(allX, allY);
figureHandle = figure( ...
    'Name', sprintf('data{6} and data{7}: %s', string(matFile)), ...
    'NumberTitle', 'off', ...
    'Color', 'w', ...
    'Position', [100, 100, 1200, 650], ...
    'WindowKeyPressFcn', @keyPressed);

layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
axesHandles = [nexttile(layout), nexttile(layout)];
for axisHandle = axesHandles
    axis(axisHandle, 'equal');
    xlim(axisHandle, xLimits);
    ylim(axisHandle, yLimits);
    grid(axisHandle, 'on');
    xlabel(axisHandle, 'x (pixels)');
    ylabel(axisHandle, 'y (pixels)');
end

timeText = uicontrol(figureHandle, ...
    'Style', 'text', ...
    'Units', 'normalized', ...
    'Position', [0.34, 0.01, 0.32, 0.04], ...
    'BackgroundColor', 'w', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');

if numel(allTimes) == 1
    sliderStep = [1, 1];
else
    sliderStep = [1 / (numel(allTimes) - 1), min(10 / (numel(allTimes) - 1), 1)];
end
sliderHandle = uicontrol(figureHandle, ...
    'Style', 'slider', ...
    'Units', 'normalized', ...
    'Position', [0.18, 0.06, 0.64, 0.035], ...
    'Min', 1, ...
    'Max', numel(allTimes), ...
    'Value', 1, ...
    'SliderStep', sliderStep, ...
    'Callback', @sliderMoved);
uicontrol(figureHandle, ...
    'Style', 'pushbutton', ...
    'String', '< Previous', ...
    'Units', 'normalized', ...
    'Position', [0.02, 0.055, 0.12, 0.045], ...
    'Callback', @(~, ~) showTimepoint(currentIndex() - 1));
uicontrol(figureHandle, ...
    'Style', 'pushbutton', ...
    'String', 'Next >', ...
    'Units', 'normalized', ...
    'Position', [0.86, 0.055, 0.12, 0.045], ...
    'Callback', @(~, ~) showTimepoint(currentIndex() + 1));

showTimepoint(1);
plotAllTimepoints();

    function sliderMoved(~, ~)
        showTimepoint(round(sliderHandle.Value));
    end

    function keyPressed(~, event)
        switch event.Key
            case {'rightarrow', 'space'}
                showTimepoint(currentIndex() + 1);
            case 'leftarrow'
                showTimepoint(currentIndex() - 1);
            case 'home'
                showTimepoint(1);
            case 'end'
                showTimepoint(numel(allTimes));
        end
    end

    function index = currentIndex()
        index = round(sliderHandle.Value);
    end

    function showTimepoint(index)
        index = max(1, min(numel(allTimes), index));
        sliderHandle.Value = index;
        currentTime = allTimes(index);
        set(timeText, 'String', sprintf('Time point %d / %d   |   t = %.6g s', ...
            index, numel(allTimes), currentTime));
        for recordIndex = 1:numel(records)
            record = records{recordIndex};
            inTimepoint = abs(record.time(:) - currentTime) < 1e-10;
            x = record.x_data(inTimepoint);
            y = record.y_data(inTimepoint);
            valid = isfinite(x) & isfinite(y);
            cla(axesHandles(recordIndex));
            scatter(axesHandles(recordIndex), x(valid), y(valid), 14, ...
                'filled', 'MarkerFaceAlpha', 0.65, 'MarkerEdgeAlpha', 0.65);
            axis(axesHandles(recordIndex), 'equal');
            xlim(axesHandles(recordIndex), xLimits);
            ylim(axesHandles(recordIndex), yLimits);
            grid(axesHandles(recordIndex), 'on');
            xlabel(axesHandles(recordIndex), 'x (pixels)');
            ylabel(axesHandles(recordIndex), 'y (pixels)');
            title(axesHandles(recordIndex), sprintf('%s   |   n = %d', ...
                recordName(record, cellIndices(recordIndex)), nnz(valid)));
        end
        drawnow;
    end

    function plotAllTimepoints()
        allPointsFigure = figure( ...
            'Name', sprintf('All time points: %s', char(matFile)), ...
            'NumberTitle', 'off', ...
            'Color', 'w', ...
            'Position', [150, 150, 1200, 600]);
        allPointsLayout = tiledlayout(allPointsFigure, 1, 2, ...
            'TileSpacing', 'compact', 'Padding', 'compact');
        timeLimits = [min(allTimes), max(allTimes)];
        if timeLimits(1) == timeLimits(2)
            timeLimits = timeLimits + [-0.5, 0.5];
        end
        for recordIndex = 1:numel(records)
            record = records{recordIndex};
            axisHandle = nexttile(allPointsLayout);
            x = record.x_data(:);
            y = record.y_data(:);
            time = record.time(:);
            valid = isfinite(x) & isfinite(y) & isfinite(time);
            scatter(axisHandle, x(valid), y(valid), 5, time(valid), 'filled', ...
                'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
            axis(axisHandle, 'equal');
            xlim(axisHandle, xLimits);
            ylim(axisHandle, yLimits);
            clim(axisHandle, timeLimits);
            grid(axisHandle, 'on');
            xlabel(axisHandle, 'x (pixels)');
            ylabel(axisHandle, 'y (pixels)');
            title(axisHandle, sprintf('%s   |   all time points   |   n = %d', ...
                recordName(record, cellIndices(recordIndex)), nnz(valid)));
        end
        colormap(allPointsFigure, parula(256));
        colorbarHandle = colorbar(allPointsLayout, 'eastoutside');
        colorbarHandle.Label.String = 'Time (s)';
        title(allPointsLayout, 'All localizations overlaid; color encodes time');
    end
end

function [xLimits, yLimits] = sharedLimits(x, y)
% Add a small margin while preserving one fixed coordinate system over time.
xRange = [min(x), max(x)];
yRange = [min(y), max(y)];
range = max([diff(xRange), diff(yRange), eps]);
padding = 0.03 * range;
xLimits = xRange + [-padding, padding];
yLimits = yRange + [-padding, padding];
end

function name = recordName(record, fallbackIndex)
if isfield(record, 'name') && ~isempty(record.name)
    name = string(record.name);
else
    name = sprintf('data{%d}', fallbackIndex);
end
end
