import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.filter as filt
from facial_recognition import WINDOW_AMT
from scipy.io import loadmat
from scipy.misc import imresize, imread


mat = loadmat("./eigenface/eigenfaces.mat");
f_width = mat['f_width'];
f_height = mat['f_height'];
eigenfaces = mat['eigenfaces'];
d,n = np.shape(eigenfaces);
avg_face = np.reshape(mat['average'], [f_height, f_width]).T;

faces = []

NUM_EIGENFACES_USED=100
FACE_THRESHOLD=15.0

for i in range(NUM_EIGENFACES_USED):
    faces.append(np.reshape(eigenfaces[:,i], [f_height, f_width]).T)

def normalize(x):
    return x/np.max(x)

def project_eigenfaces(face, faces):
    face = imresize(face,[f_height,f_width])
    grey = np.average(face, axis=2)/255.0
    normalized = grey - avg_face
    norm = np.linalg.norm(normalized)
    normalized  = normalized / norm
    projection = np.zeros_like(normalized)
    for i in range(len(faces)):
        projection += np.sum(faces[i] * normalized) * faces[i]
    return np.linalg.norm(normalize(grey) - normalize(projection * norm + avg_face)), projection * norm + avg_face

def check_face(frame, shape, interp_rot, i, j):
    height, width,_ = shape
    y, x = i*WINDOW_AMT, j*WINDOW_AMT
    face = frame[y:y+height,x:x+width,:]
    score, projection = project_eigenfaces(face,faces)
    #print score
    #cv2.imshow("face", face)
    #cv2.imshow("projection", projection)
    print score
    return score <= FACE_THRESHOLD


def main():
    for n in [1,5,10,50,100,500,1000]:
        faces = []
        for i in range(n):
            faces.append(np.reshape(eigenfaces[:,i], [f_height, f_width]).T)
        print "{0} Eigenfaces.".format(n)
        n_scores = []
        f_scores = []
        for i in range(5):
            if i == 2:
                continue
            img = imread("./face/nonface{0}.jpg".format(i))
            score = project_eigenfaces(img,faces)
            print "NonFace{0}".format(i), score
            n_scores.append(score)
        for i in range(6):
            if i == 2:
                continue
            img = imread("./face/face{0}.jpg".format(i))
            score = project_eigenfaces(img,faces)
            print "Face{0}".format(i), score
            f_scores.append(score)
        print
        print max(f_scores), min(n_scores), min(n_scores) - max(f_scores), (min(n_scores) + max(f_scores)) / 2.
        print

if __name__ == '__main__':
    main()
