import sys
import numpy as np
import cv2

class FacialRecognition:
  def __init__(self):
    self.face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    self.nose_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_nose.xml")
    self.eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

  def run(self):
    self.cap = cv2.VideoCapture(0)
    print "When face is visible, press Enter to continue."
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      frame = cv2.resize(frame, dsize=(0, 0), fx=0.7, fy=0.7)
      if not ret:
        print "Something is wrong..."
        return
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # TODO: our own haar cascade
      faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
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
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      frame = cv2.resize(frame, dsize=(0, 0), fx=0.7, fy=0.7)
      if not ret:
        print "Something is wrong..."
        return
      down_steps = (frame.shape[0] - h) / 10
      right_steps = (frame.shape[1] - w) / 10
      ssds = {}
      for i in range(down_steps):
        for j in range(right_steps):
          cand_roi = frame[10*i:10*i+h, 10*j:10*j+w]
          ssds[(i, j)] = ((face_roi - cand_roi) ** 2).sum()
      best_i, best_j = min(ssds, key=lambda k: ssds[k])
      best_roi = frame[10*best_i:10*best_i+h, 10*best_j:10*best_j+w]
      cv2.imshow("best roi", best_roi)
      cv2.waitKey(5)

def main():
  FacialRecognition().run()
        
if __name__ == "__main__":
  main()
