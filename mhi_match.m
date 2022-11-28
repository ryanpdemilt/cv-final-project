%load in values

Im = zeros(256, 256, 100);

for i=0:99
    filename = sprintf('Z/Z_%d.png', i); 
    temp = imread(filename);
    temp = double(rgb2gray(temp));
    Im(:,:,i+1) = temp;
end

diff_im = zeros(256, 256, 99);

for i=2:99
    temp_im = abs(Im(:,:,i) - Im(:,:,i-1));
    temp_im = temp_im > 15;
    temp_im = bwmorph(temp_im, 'dilate');
    temp_im = bwlabel(temp_im, 8) == 1;

    diff_im(:,:,i-1) = temp_im;
end

% to test params for mhi
% temp_im = abs(Im(:,:,2) - Im(:,:,1));
% temp_im = temp_im > 20;
% temp_im = bwmorph(temp_im, 'dilate');
% temp_im = bwlabel(temp_im, 8) == 1;

% 
% figure;
% imagesc(temp_im);
% axis('image');
% colormap('gray');

%create MHI/MEI

MHI = zeros(256, 256);

for t=2:99
    idx = find(diff_im(:,:,t-1)>0);
    MHI(idx) = t;
end

MEI = MHI > 0;

figure;
imagesc(MHI);
axis('image');
colormap('gray');


% figure;
% imagesc(MEI);
% axis('image');
% colormap('gray');


MHI = max(0, (MHI-1.0)/99.0);

simMEI = similitudeMoments(MEI);
simMHI = similitudeMoments(MHI);
vector = [simMEI simMHI];

J_vec = [0.0692,0.0232,0.0208,-0.0131,0.2174,0.0132,0.0390,...
    0.1304,0.0325,0.0599,-0.0167,0.5142,0.0410,0.1422];

Z_vec = [0.0443, 0.0021, -0.0127, -0.0020, 0.1670, -0.0006, 0.0058, 0.1283,...    
    0.0216, -0.0219, -0.0159, 0.4259, 0.0097, -0.0558];

dist_j = sqrt(sum((vector - J_vec).^2));
dist_z = sqrt(sum((vector - Z_vec).^2));

if dist_z < dist_j
    disp('ZZZZZZZ')
else
    disp('JJJJJJJ')
end




%% functions

function [val]=spatialMoment(im, p, q)
    % row is y value and column is x value
    s = size(im);
    val = 0;
    for row=1:s(1)
        for col=1:s(2)
            val = val + (col^p * row^q * im(row,col));
        end
    end
end

function [val]=centralMoment(im, p, q)
    % row is y value and column is x value
    s = size(im);
    val = 0;
    rowBar = spatialMoment(im, 0, 1)/spatialMoment(im, 0, 0);
    colBar = spatialMoment(im, 1, 0)/spatialMoment(im, 0, 0);
    for row=1:s(1)
        for col=1:s(2)
            val = val + ((col-colBar)^q * (row-rowBar)^p * im(row, col));
        end
    end
end

function [val]=similitudeMoment(im, i, j)
    val = centralMoment(im, i, j)/(spatialMoment(im, 0, 0)^(1+(i+j)/2));
end

function [tempIm]=sim_pixel(im, i, j)
    
    tempIm = zeros(size(im));
    [r,c] = size(im);

    rowBar = spatialMoment(im, 0, 1)/spatialMoment(im, 0, 0);
    colBar = spatialMoment(im, 1, 0)/spatialMoment(im, 0, 0);

    for x=1:r
        for y=1:c
            tempIm(x,y) = (y-colBar)^i * (x-rowBar)^j * im(x,y);
        end
    end
    
    tempIm = mat2gray(tempIm);
    
    figure;
    imagesc(tempIm);
    axis('image');
    colormap('gray');
end

function [Nvals]=similitudeMoments(im)
    im = mat2gray(im);
    
    i = [0; 0; 1; 1; 2; 2; 3];
    j = [2; 3; 1; 2; 0; 1; 0];

    ij = [i'; j'];

    Nvals = [0,0,0,0,0,0,0];
    for index=1:7
        Nvals(index) = similitudeMoment(im, ij(1,index), ij(2, index));
    end

end

    