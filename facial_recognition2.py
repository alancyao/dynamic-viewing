import cv2
from threading import Thread, Lock
from facial_recognition import FacialRecognition, WINDOW_AMT
from kalman_filter import KalmanFilter1, KalmanFilter2
import numpy as np
from check_face import check_face

NUM_PYR = 2
WINDOW_AMT = 8
assert (float(WINDOW_AMT) / (2 ** NUM_PYR)).is_integer(), "WINDOW_AMT must remain an integer at all pyramid levels."
START_FACE_DIST = 310

class FacialRecognition2:
    def __init__(self, kalman_filter=KalmanFilter1(0.1,0.1), do_rot=True, do_scale=True):
        self.stop = False
        self.facial_recognition = FacialRecognition()
        self.facial_recognition.calibrate()
        self.kalman_filter = kalman_filter
        self.params = None
        self.lock = Lock()
        self.do_rot = do_rot
        self.do_scale = do_scale
        self.ready = False
        self.last_output = None
        Thread(target=self.start_loop, args=()).start()
        while not self.ready:
            pass
    def calibrate(self):
        self.facial_recognition.calibrate()

    def end(self):
        self.stop = True
        self.facial_recognition.end()

    def run(self):
        while True:
            self.get_face()
            if cv2.waitKey(1) & 0xFF == 10:
                cv2.destroyWindow("frame")
                self.facial_recognition.ig.keep_going = False
                cv2.waitKey(1)
                break

    def start_loop(self):
        while not self.stop:
            best_i, best_j, frame, interp_shape, interp_rot = self.facial_recognition.get_face(self.do_rot, self.do_scale)
            ztrans = self.facial_recognition.camera_f - self.facial_recognition.camera_f * interp_shape[0] / self.facial_recognition.w
            self.params = (best_i, best_j, frame, interp_shape, interp_rot, ztrans)
            self.ready = True
    def get_face(self, do_rot=True, do_scale=True):
        if self.params is not None:
            (best_i, best_j, frame, interp_shape, interp_rot, ztrans) = self.params
            self.has_face = check_face(frame, interp_shape, interp_rot, best_i, best_j)
            print self.has_face
            self.old_params = self.params
            self.params = None
            coords = [best_i, best_j, ztrans]
        else:
            (best_i, best_j, frame, interp_shape, interp_rot, ztrans) = self.old_params
            coords = None
        self.kalman_filter.Update(coords)
        best_i, best_j, ztrans = self.kalman_filter.Pos()
        best_i = int(best_i); best_j = int(best_j)
        cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + self.facial_recognition.w, WINDOW_AMT*best_i + self.facial_recognition.h), color=(255,255,255), thickness=2)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        return best_i, best_j, frame, interp_shape, interp_rot, ztrans

    def get_transforms(self, do_rot=True, do_scale=True, do_trans=True):
        best_i, best_j, frame, interp_shape, interp_rot, ztrans = self.get_face(do_rot, do_scale)
        if (not self.has_face and self.last_output is not None):
            return self.last_output
        # Rotation amount
        center = (np.array((WINDOW_AMT*best_j, WINDOW_AMT*best_i)) + np.array((WINDOW_AMT*best_j + self.facial_recognition.w, WINDOW_AMT*best_i + self.facial_recognition.h))) / 2.0
        disp = center - self.facial_recognition.start_center
        rot = np.arctan(disp/START_FACE_DIST) * (180 / np.pi) # change to actual face dist
        self.last_output = np.array(rot), ztrans, interp_rot
        return np.array(rot), ztrans, interp_rot

def main():
    f = FacialRecognition2(KalmanFilter1(0.1,0.1,0.9))
    f.run()

if __name__ == "__main__":
    main()
