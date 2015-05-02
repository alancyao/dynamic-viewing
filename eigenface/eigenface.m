CROP_X = 50;
CROP_Y = 50;
NUM_EIGS = 1000;
fid = fopen('./faces.txt');
face_file = fgetl(fid);
faces_mat = [];
while ischar(face_file)
  face = rgb2gray(im2double(imread(face_file)));
  [height, width] = size(face);
  x=width/2;
  y=height/2;
  cropped_face = face(y-CROP_Y:y+CROP_Y,x-CROP_X:x+CROP_X);
  [f_height, f_width] = size(cropped_face);
  features = reshape(cropped_face,[f_height*f_width,1]);
  features = features / norm(features);
  faces_mat = [faces_mat features];
  face_file = fgetl(fid);
end
[d,n] = size(faces_mat);
average = sum(faces_mat,2)/n;
faces_mat = faces_mat - repmat(average,[1,n]);
[eigenfaces,S,V] = svds(faces_mat,NUM_EIGS);
save('eigenfaces.mat', 'eigenfaces', 'f_width', 'f_height', 'average');
