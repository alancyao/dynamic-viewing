import sys
import numpy as np
from skimage import transform as tf
import cv2

NUM_PYR = 2
WINDOW_AMT = 8
assert (float(WINDOW_AMT) / (2 ** NUM_PYR)).is_integer(), "WINDOW_AMT must remain an integer at all pyramid levels."
DISP_SCALE = 0.7
AVERAGE_FACE_WIDTH = 250
START_FACE_DIST = 310

class FacialRecognition:
  def run(self):
    self.cap = cv2.VideoCapture(0)
    print "When face is visible, press Enter to continue."
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        print "Something is wrong..."
        return
      frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
      for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
      cv2.imshow("calibration", frame)
      if cv2.waitKey(1) & 0xFF == 10:
        cv2.destroyWindow("calibration")
        if len(faces) > 0:
          break
        else:
          print "No face detected."

    x, y, w, h = faces[0]
    face_roi = frame[y:y+h, x:x+w]
    face_pyramid = list(tf.pyramid_gaussian(face_roi, max_layer=NUM_PYR, downscale=2))
    print "Tracking face..."

    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        print "Something is wrong..."
        return
      frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
      frame_pyramid = list(tf.pyramid_gaussian(frame, max_layer=NUM_PYR, downscale=2))

      best_i, best_j = self.determine_best_shift(face_pyramid, frame_pyramid)
      cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + w, WINDOW_AMT*best_i + h), color=(255, 0, 0), thickness=2)
      cv2.imshow("frame", frame)
      cv2.waitKey(5)

  def calibrate(self):
    self.cap = cv2.VideoCapture(0)
    print "Place face 1 ft from camera. When face is visible, press Enter to continue."
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        print "Something is wrong..."
        return
      frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
      for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
      cv2.imshow("calibration", frame)
      if cv2.waitKey(1) & 0xFF == 10:
        cv2.destroyWindow("calibration")
        if len(faces) > 0:
          break
        else:
          print "No face detected."

    x, y, w, h = faces[0]
    self.face_roi = frame[y:y+h, x:x+w]
    self.face_pyramid = list(tf.pyramid_gaussian(self.face_roi, max_layer=NUM_PYR, downscale=2))
    # w = f*Y/Z  -->  f = wZ/Y
    self.camera_f = w * START_FACE_DIST/AVERAGE_FACE_WIDTH
    self.start_center = np.array((x + w/2.0, y+h/2.0))
    self.w = w; self.h = h
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    self.wtf()

  def wtf(self):
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)

  def get_face(self):
    ret, frame = self.cap.read()
    if not ret:
      print "Something is wrong..."
      return
    frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
    frame_pyramid = list(tf.pyramid_gaussian(frame, max_layer=NUM_PYR, downscale=2))

    best_i, best_j = self.determine_best_shift(self.face_pyramid, frame_pyramid)
    return (np.array((WINDOW_AMT*best_j, WINDOW_AMT*best_i)) + np.array((WINDOW_AMT*best_j + self.w, WINDOW_AMT*best_i + self.h))) / 2.0

  def get_rotation(self):
    center = self.get_face()
    disp = center - self.start_center
    rot = np.arctan(disp/START_FACE_DIST) * (180 / np.pi) # change to actual face dist
    return rot

  def determine_best_shift(self, face_pyramid, frame_pyramid):
    wa = int(WINDOW_AMT / (2 ** NUM_PYR))
    region_indices = [0, (frame_pyramid[-1].shape[0] - face_pyramid[-1].shape[0]) / wa,
                      0, (frame_pyramid[-1].shape[1] - face_pyramid[-1].shape[1]) / wa]
    for pyr_index in reversed(range(NUM_PYR+1)):
      res = self.compute_ssd(frame_pyramid[pyr_index],
                             face_pyramid[pyr_index],
                             2 ** pyr_index,
                             region_indices)
      if res is None:
        break
      best_i, best_j = res
      region_indices = [best_i - 1, best_i + 1,
                        best_j - 1, best_j + 1]
    return best_i, best_j

  def compute_ssd(self, frame, face, scaleAmt, region_indices):
    wa = int(WINDOW_AMT / scaleAmt)
    h = face.shape[0]
    w = face.shape[1]
    ssds = {}
    for i in range(region_indices[0], region_indices[1]):
      for j in range(region_indices[2], region_indices[3]):
        cand_roi = frame[wa*i:wa*i+h, wa*j:wa*j+w]
        if cand_roi.shape == face.shape:
          ssds[(i, j)] = ((face - cand_roi) ** 2).sum()
    if not ssds:
      return None
    best_i, best_j = min(ssds, key=lambda k: ssds[k])
    return best_i, best_j

def main():
  FacialRecognition().run()

if __name__ == "__main__":
  main()
