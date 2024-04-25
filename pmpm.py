import numpy as np
import cv2
import math

WINDOW_SIZE = 35
MAX_DISPARITY = 60
PLANE_PENALTY = 120

class PatchMatch:
    # 미완성
    def __init__(self, alpha, gamma, tau_c, tau_g):
        self.alpha = alpha
        self.gamma = gamma
        self.tau_c = tau_c
        self.tau_g = tau_g

        self.window_size = 35
        self.max_disparity = 60
        self.plane_penalty = 120 

    
    def inside(self, x, y, lbx, lby, ubx, uby):
        return lbx <= x < ubx and lby <= y < uby
    
    # 미완성
    def set(self, img1: np.ndarray, img2: np.ndarray):
        self.views[0] = img1
        self.views[1] = img2
        self.rows, self.cols, _ = img1.shape

        wmat_sizes = [self.rows, self.cols, WINDOW_SIZE, WINDOW_SIZE]
        self.weights[0] = np.zeros(wmat_sizes, dtype=np.float32)
        self.weights[1] = np.zeros(wmat_sizes, dtype=np.float32)



    # 수식 (5)
    def dissimilarity(self, pp, qq, pg, qg):
        cost_c = np.linalg.norm(pp - qq, ord = 1) #Color difference
        cost_g = np.linalg.norm(pg - qg, ord = 1) #Gray-value gradient difference
        cost_c = min(cost_c, self.tau_c)
        cost_g = min(cost_g, self.tau_g)

        return (1- self.alpha) * cost_c + self.alpha * cost_g


    def plane_match_cost(self, p, cx, cy, ws ,cpv):
        sign = -1 + 2*cpv
        cost = 0
        half = ws // 2

        f1, f2 = self.views[cpv], self.views[1-cpv]
        g1, g2 = self.views[cpv], self.views[1-cpv]
        w1 = self.weights[cpv]

        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if not self.inside(x, y, 0, 0, f1.shape[1], f1.shape[0]):
                    continue

                dsp = self.disparity(x,y,p)

                #dsp 가 범위에 벗어나면, plane_penalty 추가
                if dsp < 0 or dsp > self.max_disparity:
                    cost += self.plane_penalty
                
                else: 
                    match = x + sign * dsp
                    x_match = max(0, min(f1.shape[1]-1, int(match)))






