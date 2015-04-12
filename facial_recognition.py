import sys
import numpy as np
from skimage import transform as tf
import cv2
from threading import Thread

NUM_PYR = 2
WINDOW_AMT = 8
assert (float(WINDOW_AMT) / (2 ** NUM_PYR)).is_integer(), "WINDOW_AMT must remain an integer at all pyramid levels."
DISP_SCALE = 0.7
AVERAGE_FACE_WIDTH = 250
START_FACE_DIST = 310
RESCALING_FACTORS = [0.7, 1, 1.3]


class WebcamImageGetter:
  def __init__(self):
    self.currentFrame = None
    self.capture = cv2.VideoCapture(0)
    self.keep_going = True

  def start(self):
    Thread(target=self.updateFrame, args=()).start()

  def updateFrame(self):
    while self.keep_going:
      ret, frame = self.capture.read()
      self.currentFrame= cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)

  def getFrame(self):
    return self.currentFrame

class FacialRecognition:
  def end(self):
    cv2.destroyWindow("frame")
    cv2.waitKey(1)
    self.ig.keep_going = False

  def run(self):
    self.calibrate()
    print "Tracking face..."

    while True:
      best_i, best_j, frame, interp_shape = self.get_face()
      # Display bounding box
      cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + interp_shape[1], WINDOW_AMT*best_i + interp_shape[0]), color=(255, 0, 0), thickness=2)
      cv2.imshow("frame", frame)
      if cv2.waitKey(1) & 0xFF == 10:
        cv2.destroyWindow("frame")
        self.ig.keep_going = False
        break

  def calibrate(self):
    self.ig = WebcamImageGetter()
    self.ig.start()
    print "Place face 1 ft from camera. When face is visible, press Enter to continue."
    while True:
      frame = self.ig.getFrame()
      if frame is None:
        continue
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
    num_pix = float(w*h)
    face_roi = frame[y:y+h, x:x+w]
    scaled_faces = [tf.rescale(face_roi, scale=sc) for sc in RESCALING_FACTORS]
    self.face_pyramids = [list(tf.pyramid_gaussian(face, max_layer=NUM_PYR, downscale=2))
                          for face in scaled_faces]
    self.scaled_weights = [num_pix / (sf.shape[0]*sf.shape[1]) for sf in scaled_faces]
    # w = f*Y/Z  -->  f = wZ/Y
    self.camera_f = w * START_FACE_DIST/AVERAGE_FACE_WIDTH
    self.start_center = np.array((x + w/2.0, y+h/2.0))
    self.w = w; self.h = h
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)

  def get_face(self):
    frame = self.ig.getFrame()
    frame_pyramid = list(tf.pyramid_gaussian(frame, max_layer=NUM_PYR, downscale=2))

    ssds = {}
    for i, face_pyramid in enumerate(self.face_pyramids):
      res = self.determine_best_shift(face_pyramid, frame_pyramid)
      if res is None:
          print "No face detected."
          return self.best_i, self.best_j, frame, self.interp_shape
      best_i, best_j, best_ssd = res
      ssds[i] = (best_ssd * self.scaled_weights[i], best_i, best_j, np.array(face_pyramid[0].shape))
    sorted_ssds = sorted(ssds.items(), key=lambda x : x[1][0])
    best_ssd, best_i, best_j, best_shape = sorted_ssds[0][1]
    sbest_ssd, best_i, sbest_j, sbest_shape = sorted_ssds[1][1]
    interp_shape = map(int, best_ssd / (best_ssd + sbest_ssd) * sbest_shape + \
                            sbest_ssd / (best_ssd + sbest_ssd) * best_shape)
    print interp_shape
    self.best_i, self.best_j, self.interp_shape = best_i, best_j, interp_shape
    return best_i, best_j, frame, interp_shape

  def get_rotation(self):
    best_i, best_j, frame, interp_shape = self.get_face()
    center = (np.array((WINDOW_AMT*best_j, WINDOW_AMT*best_i)) + np.array((WINDOW_AMT*best_j + self.w, WINDOW_AMT*best_i + self.h))) / 2.0
    disp = center - self.start_center
    rot = np.arctan(disp/START_FACE_DIST) * (180 / np.pi) # change to actual face dist

    # Display image (for fun)
    cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + interp_shape[1], WINDOW_AMT*best_i + interp_shape[0]), color=(255, 0, 0), thickness=2)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    return np.array(rot)

  def determine_best_shift(self, face_pyramid, frame_pyramid):
    wa = int(WINDOW_AMT / (2 ** NUM_PYR))
    region_indices = [0, (frame_pyramid[-1].shape[0] - face_pyramid[-1].shape[0]) / wa,
                      0, (frame_pyramid[-1].shape[1] - face_pyramid[-1].shape[1]) / wa]
    for pyr_index in reversed(range(NUM_PYR+1)):
      res = self.compute_ssd(frame_pyramid[pyr_index],
                             face_pyramid[pyr_index],
                             2 ** pyr_index,
                             region_indices)
      if res == "no face":
        return None
      if res is None:
        break
      best_i, best_j, tmp_ssd = res
      if pyr_index == NUM_PYR:
          best_ssd = tmp_ssd
      region_indices = [best_i - 1, best_i + 1,
                        best_j - 1, best_j + 1]
    return best_i, best_j, best_ssd

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
    z_score = (ssds[(best_i, best_j)] - np.mean(ssds.values())) / np.std(ssds.values()) if np.std(ssds.values()) != 0 else 0
    if z_score == float("nan") or z_score >= -1:
      return "no face"
    return best_i, best_j, ssds[(best_i, best_j)]

def main():
  FacialRecognition().run()

if __name__ == "__main__":
  main()
