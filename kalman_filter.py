import time
import numpy as np

class KalmanFilter1:
    """
    First order kalman filter.
    Assumes that all dynamics are locally linear (constant velocity).
    Models kinematic motion.
    """
    def __init__(self, q, r):
        self._trajectory = [np.array([0,0,0,0,0,0])]
        self.last_time = time.time()
        self.C = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ])
        self.Q = q*np.eye(6)
        self.R = q*np.eye(3)
        self.P = np.eye(6)
    def GetTimeDelta(self):
        current_time = time.time()
        T = current_time - self.last_time
        self.last_time = current_time
        return T
    def Update(self,observation):
        T = self.GetTimeDelta()
        A = np.matrix([
            [1, T, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, T, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, T],
            [0, 0, 0, 0, 0, 1],
        ])
        x_hat = A*np.matrix(self._trajectory[-1]).T
        P_hat = np.einsum('ij,jk,kl', A, self.P, A.T) + self.Q
        if observation is not None:
            S = self.C*P_hat*self.C.T+self.R
            K = P_hat*self.C.T*np.linalg.inv(S)
            e = np.matrix(observation).T - self.C*x_hat
            x_plus = x_hat+K*e
            P_plus = (np.eye(6)-K*self.C)*P_hat
        else:
            x_plus = x_hat
            P_plus = P_hat
        x_plus = np.asarray(x_plus.T)
        self._trajectory.append(x_plus[0])
        self.P = P_plus
    def Get(self):
        return self._trajectory[-1]
    def Pos(self):
        state = self.Get()
        return [state[0], state[2], state[4]]

class KalmanFilter2:
    """
    Second order kalman filter.
    Assumes that all dynamics are constant acceleration.
    More flexible than KalmanFilter1, but more susceptible to noise and
    divergence.
    """
    def __init__(self, q, r):
        self._trajectory = [np.array([0,0,0,0,0,0,0,0,0])]
        self.last_time = time.time()
        self.C = np.matrix([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
        ])
        self.Q = q*np.eye(9)
        self.R = q*np.eye(3)
        self.P = np.eye(9)
    def GetTimeDelta(self):
        current_time = time.time()
        T = current_time - self.last_time
        self.last_time = current_time
        return T
    def Update(self,observation):
        T = self.GetTimeDelta()
        A = np.matrix([
            [1, T, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, T, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, T, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, T, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, T, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, T],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        x_hat = A*np.matrix(self._trajectory[-1]).T
        P_hat = np.einsum('ij,jk,kl', A, self.P, A.T) + self.Q
        if observation is not None:
            S = self.C*P_hat*self.C.T+self.R
            K = P_hat*self.C.T*np.linalg.inv(S)
            e = np.matrix(observation).T - self.C*x_hat
            x_plus = x_hat+K*e
            P_plus = (np.eye(9)-K*self.C)*P_hat
        else:
            x_plus = x_hat
            P_plus = P_hat
        x_plus = np.asarray(x_plus.T)
        self._trajectory.append(x_plus[0])
        self.P = P_plus
    def Get(self):
        return self._trajectory[-1]
    def Pos(self):
        state = self.Get()
        return [state[0], state[3], state[6]]

def main():
    observations = [[0,0,0],
                    [1,0,1],
                    [4,0,2],
                    [9,0,3],
                    None, #[16,0,4],
                    None, #[25,0,5],
                    [36,0,6],
                    ]
    k2 = KalmanFilter2(0.1,0.1)
    k1 = KalmanFilter1(0.1,0.1)
    for observation in observations:
        k1.Update(observation)
        k2.Update(observation)
        print k1.Pos(), k2.Pos()
        time.sleep(1)


if __name__=='__main__':
    main()
