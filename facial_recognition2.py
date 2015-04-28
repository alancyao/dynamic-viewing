import cv2
from threading import Thread, Lock
from facial_recognition import FacialRecognition, WINDOW_AMT
from kalman_filter import KalmanFilter1, KalmanFilter2

class FacialRecognition2:
    def __init__(self, kalman_filter, do_rot=True, do_scale=True):
        self.stop = False
        self.facial_recognition = FacialRecognition()
        self.facial_recognition.calibrate()
        self.kalman_filter = kalman_filter
        self.params = None
        self.lock = Lock()
        self.do_rot = do_rot
        self.do_scale = do_scale
        self.ready = False
        Thread(target=self.start_loop, args=()).start()
        while not self.ready:
            pass
    def calibrate(self):
        self.facial_recognition.calibrate()
    def end(self):
        self.stop = True
    def run(self):
        while True:
            best_i, best_j, frame, interp_shape, interp_rot, ztrans = self.get_face()
            cv2.rectangle(frame, (WINDOW_AMT*best_j, WINDOW_AMT*best_i), (WINDOW_AMT*best_j + self.facial_recognition.w, WINDOW_AMT*best_i + self.facial_recognition.h), color=(255,255,255), thickness=2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 10:
                cv2.destroyWindow("frame")
                self.ig.keep_going = False
                break

    def start_loop(self):
        while not self.stop:
            best_i, best_j, frame, interp_shape, interp_rot = self.facial_recognition.get_face(self.do_rot, self.do_scale)
            ztrans = self.facial_recognition.camera_f - self.facial_recognition.camera_f * interp_shape[0] / self.facial_recognition.w
            with self.lock:
                self.params = (best_i, best_j, frame, interp_shape, interp_rot, ztrans)
            self.ready = True
    def get_face(self):
        with self.lock:
            if self.params is not None:
                (best_i, best_j, frame, interp_shape, interp_rot, ztrans) = self.params
                self.old_params = self.params
                self.params = None
                coords = [best_i, best_j, ztrans]
            else:
                (best_i, best_j, frame, interp_shape, interp_rot, ztrans) = self.old_params
                coords = None
        self.kalman_filter.Update(coords)
        best_i, best_j, ztrans = self.kalman_filter.Pos()
        return int(best_i), int(best_j), frame, interp_shape, interp_rot, ztrans

def main():
    f = FacialRecognition2(KalmanFilter2(0.1,0.1))
    f.run()

if __name__ == "__main__":
    main()
