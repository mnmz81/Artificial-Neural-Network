function [] = ShowImage(img)
% Show image from given vector
size=sqrt(length(img));
colormap gray;
imagesc(reshape(img,size,size));
end