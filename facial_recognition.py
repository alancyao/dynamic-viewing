import sys
import numpy as np
from skimage import transform as tf
import cv2

WINDOW_AMT = 10
DISP_SCALE = 0.7
NUM_PYR = 2
ROT_AMTS = np.linspace(-15, 15, num=5)
ROT_AMTS = [0]

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
          x, y, w, h = faces[0]
          # x, y, w, h = x-10,y-10,w+20,h+20
          ret, frame = self.cap.read()
          frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
          face_roi = frame[y:y+h, x:x+w]
          break
        else:
          print "No face detected."

    circle = np.zeros_like(face_roi)
    cv2.circle(circle, (h / 2, w / 2), int(min(h / 2, w / 2) * 1.0), color=(255, 255, 255), thickness=-1)
    self.circle_pyramid = [tf.rescale(circle, scale=0.5**i) for i in range(NUM_PYR+1)]
    face_pyramids = [list(tf.pyramid_gaussian(tf.rotate(face_roi, angle=rot_ang),
                                              max_layer=NUM_PYR,
                                              downscale=2)) for rot_ang in ROT_AMTS]
    for pyr in face_pyramids:
      for i, img in enumerate(pyr):
        img[self.circle_pyramid[i] == 0] = 0
        cv2.imshow("ew", img); cv2.waitKey(0)
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        print "Something is wrong..."
        return
      frame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)
      frame_pyramid = list(tf.pyramid_gaussian(frame, max_layer=NUM_PYR, downscale=2))

      best = [None, None]
      for i, face_pyramid in enumerate(face_pyramids):
        best_i, best_j, best_ssd = self.determine_best_shift(face_pyramid, frame_pyramid)
        if best[0] is None or best_ssd < best[1]:
          best[0] = (ROT_AMTS[i], best_i, best_j)
          best[1] = best_ssd
      (best_rot_amt, best_i, best_j), best_ssd = best
      print best
      cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + w, WINDOW_AMT*best_i + h), color=(255, 0, 0), thickness=2)
      cv2.imshow("frame", frame)
      cv2.waitKey(5)

  def determine_best_shift(self, face_pyramid, frame_pyramid):
    wa = int(WINDOW_AMT / (2 ** NUM_PYR))
    region_indices = [0, (frame_pyramid[-1].shape[0] - face_pyramid[-1].shape[0]) / wa,
                      0, (frame_pyramid[-1].shape[1] - face_pyramid[-1].shape[1]) / wa]
    for pyr_index in reversed(range(NUM_PYR+1)):
      res = self.compute_ssd(frame_pyramid[pyr_index],
                             face_pyramid[pyr_index],
                             self.circle_pyramid[pyr_index],
                             2 ** pyr_index,
                             region_indices)
      if res is None:
        break
      best_i, best_j, temp_ssd = res
      region_indices = [best_i - 1, best_i + 1,
                        best_j - 1, best_j + 1]
      if pyr_index == NUM_PYR:
        best_ssd = temp_ssd
    return best_i, best_j, best_ssd

  def compute_ssd(self, frame, face, circle, scaleAmt, region_indices):
    wa = int(WINDOW_AMT / scaleAmt)
    h = face.shape[0]
    w = face.shape[1]
    ssds = {}
    for i in range(region_indices[0], region_indices[1]):
      for j in range(region_indices[2], region_indices[3]):
        cand_roi = frame[wa*i:wa*i+h, wa*j:wa*j+w]
        if cand_roi.shape == face.shape:
          cand_roi[circle == 0] = 0
          ssds[(i, j)] = ((face - cand_roi) ** 2).sum()
    if not ssds:
      return None
    best_i, best_j = min(ssds, key=lambda k: ssds[k])
    return best_i, best_j, ssds[(best_i, best_j)]

def main():
  FacialRecognition().run()
        
if __name__ == "__main__":
  main()
