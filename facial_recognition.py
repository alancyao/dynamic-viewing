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
RESCALING_FACTORS = [0.5, 1, 1.5]
ROT_AMTS = [0] #np.linspace(-45, 45, num=3)


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
    while True:
      best_i, best_j, frame, interp_shape, interp_rot = self.get_face()
      if self.init_interp_shape is None:
        self.init_interp_shape = interp_shape
      # Display bounding box
      color = self.compute_interp_color(interp_shape)
      cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + self.w, WINDOW_AMT*best_i + self.h), color=color, thickness=2)
      cv2.imshow("frame", frame)
      if cv2.waitKey(1) & 0xFF == 10:
        cv2.destroyWindow("frame")
        self.ig.keep_going = False
        break

  def compute_interp_color(self, interp_shape):
    amt = np.linalg.norm(interp_shape - self.init_interp_shape)
    if interp_shape[0] < self.init_interp_shape[0]:
      val = min(1.0, amt / 15.0)
      return (255 * (1 - val), 255 * val, 0)
    else:
      val = min(1.0, amt / 15.0)
      return (255 * (1 - val), 0, 255 * val)

  def calibrate(self):
    self.ig = WebcamImageGetter()
    self.ig.start()
    self.init_interp_shape = None
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
    rotated_faces = [tf.rotate(face_roi, angle=rot_ang) for rot_ang in ROT_AMTS]
    self.rotated_face_pyramids = [list(tf.pyramid_gaussian(face, max_layer=NUM_PYR, downscale=2))
                                  for face in rotated_faces]
    scaled_faces = [tf.rescale(face_roi, scale=sc) for sc in RESCALING_FACTORS]
    self.scaled_face_pyramids = [list(tf.pyramid_gaussian(face, max_layer=NUM_PYR, downscale=2))
                                 for face in scaled_faces]
    # scaled_weights are used for scaled_faces
    self.scaled_weights = [num_pix / (sf.shape[0]*sf.shape[1]) for sf in scaled_faces]
    # we observed that the small detector is too strong, so we penalize it more
    self.scaled_weights[0] *= 1.5
    # w = f*Y/Z  -->  f = wZ/Y
    self.camera_f = w * START_FACE_DIST/AVERAGE_FACE_WIDTH
    self.start_center = np.array((x + w/2.0, y+h/2.0))
    self.w = w; self.h = h
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    cv2.destroyWindow("calibration")
    cv2.waitKey(1)
    print "Tracking face...press Enter to quit."
    print "Red: close, green: far, blue: in between."

  def get_face(self):
    frame = self.ig.getFrame()
    frame_pyramid = list(tf.pyramid_gaussian(frame, max_layer=NUM_PYR, downscale=2))

    scale_ssds = {}
    for i, face_pyramid in enumerate(self.scaled_face_pyramids):
      res = self.determine_best_shift(face_pyramid, frame_pyramid)
      best_i, best_j, best_ssd = res
      scale_ssds[i] = (1.0 / (best_ssd * self.scaled_weights[i]), best_i, best_j, np.array(face_pyramid[0].shape))
    if len(scale_ssds) == 3:
      best_i, best_j = scale_ssds[1][1], scale_ssds[1][2]
    else:
      best_i, best_j = scale_ssds[0][1], scale_ssds[0][2]
    total = sum([v[0] for v in scale_ssds.values()])
    interp_shape = sum([v[0] / total * v[3] for v in scale_ssds.values()])

    rot_ssds = {}
    for i, face_pyramid in enumerate(self.rotated_face_pyramids):
      res = self.determine_best_shift(face_pyramid, frame_pyramid)
      rot_best_i, rot_best_j, best_ssd = res
      rot_ssds[i] = (1.0 / best_ssd, rot_best_i, rot_best_j, np.array(face_pyramid[0].shape))
    total = sum([v[0] for v in rot_ssds.values()])
    interp_rot = sum([v[0] / total * ROT_AMTS[k] for k, v in rot_ssds.items()])
    print "Interpolated rot: ", interp_rot

    return best_i, best_j, frame, interp_shape, interp_rot

  def get_transforms(self):
    best_i, best_j, frame, interp_shape, interp_rot = self.get_face()
    # Rotation amount
    if self.init_interp_shape is None:
      self.init_interp_shape = interp_shape
    center = (np.array((WINDOW_AMT*best_j, WINDOW_AMT*best_i)) + np.array((WINDOW_AMT*best_j + self.w, WINDOW_AMT*best_i + self.h))) / 2.0
    disp = center - self.start_center
    rot = np.arctan(disp/START_FACE_DIST) * (180 / np.pi) # change to actual face dist

    # Z-Axis translation amt:  w = fX/Z -> Z = fX/w
    ztrans = self.camera_f - self.camera_f * interp_shape[0] / self.w

    # Display image (for fun)
    color = self.compute_interp_color(interp_shape)
    cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + self.w, WINDOW_AMT*best_i + self.h), color=color, thickness=2)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    return np.array(rot), ztrans, interp_rot

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
    return best_i, best_j, ssds[(best_i, best_j)]

def main():
  FacialRecognition().run()

if __name__ == "__main__":
  main()
