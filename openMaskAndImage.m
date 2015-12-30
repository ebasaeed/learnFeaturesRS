function [inputImg, mask] = openMaskAndImage(homeDir, setNo, maskNo, mosaicNo, noBands)

load normalizationValues.mat %this will load 'maxi' to the workspace which has the maximum value per band

%open mask image
mask = imread([homeDir, 'gt', num2str(setNo), '_', num2str(maskNo),'.png']);
mask = mask+1; %one of the regions is marked as zero, hence the addition

%open bands and construct a single image of all bands
inputImg = double(zeros(size(mask,1), size(mask,2), noBands));
for bandNo=1:noBands
    band = imread([homeDir,'tm', num2str(setNo), '_', num2str(maskNo),'_', num2str(mosaicNo), '_', '000', num2str(bandNo-1), '.png']);
    band = double(band)/maxi(bandNo);
    inputImg(:,:,bandNo) = band;
end;

end