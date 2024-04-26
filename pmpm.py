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
    
    def compute_greyscale_gradient(frame, gradient):
    # Implement the gradient computation logic here
        pass

    def disparity(x, y, p):
        return p[0] * x + p[1] * y + p[2]


    # 수식 (4)
    def weight(p, q, gamma=10.0):
        return np.exp(-np.linalg.norm(p - q, ord=1) / gamma)

    # 수식 (5) 중, ||I_q - I_q''||
    def vec_average(x, y, wx):
        return wx * x + (1 - wx) * y


    # 1단계
    def set(self, img1: np.ndarray, img2: np.ndarray):
        self.views[0] = img1
        self.views[1] = img2
        self.rows, self.cols, _ = img1.shape

        wmat_sizes = [self.rows, self.cols, WINDOW_SIZE, WINDOW_SIZE]
        self.weights[0] = np.zeros(wmat_sizes, dtype=np.float32)
        self.weights[1] = np.zeros(wmat_sizes, dtype=np.float32)

    # 2단계
    def process(self, iterations, reverse=False):
        for iter in range(iterations):
            iter_type = (iter % 2 ==0)

            for work_view in range(2):
                if iter_type: # 짝수 번쨰 (Forward pass)
                    self.process_pixel(work_view, iter, 0, self.rows, 0, self.cols, 1)
                else:
                    self.process_pixel(work_view, self.rows-1, -1, self.cols-1, -1, -1)
            
        self.planes_to_disparity(self.planes[0], self.planes[1])
        self.planes_to_disparity(self.planes[1], self.planes[0])


    #3단계 (미완성)
    def postprocess(self):
        pass



    # 수식 (5)
    def dissimilarity(self, pp, qq, pg, qg):
        cost_c = np.linalg.norm(pp - qq, ord = 1) #Color difference
        cost_g = np.linalg.norm(pg - qg, ord = 1) #Gray-value gradient difference
        cost_c = min(cost_c, self.tau_c)
        cost_g = min(cost_g, self.tau_g)

        return (1- self.alpha) * cost_c + self.alpha * cost_g

    # 수식 (3)
    def plane_match_cost(self, p, cx, cy, ws ,cpv):
        sign = -1 + 2*cpv
        cost = 0
        half = ws // 2

        f1, f2 = self.views[cpv], self.views[1-cpv]
        g1, g2 = self.grads[cpv], self.grads[1-cpv]
        w1 = self.weights[cpv]

        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if not self.inside(x, y, 0, 0, f1.shape[1], f1.shape[0]):
                    continue
                
                #a_f*q_x + b_f*q_b + c_f
                dsp = self.disparity(x,y,p)

                #dsp 가 범위에 벗어나면, plane_penalty 추가
                if dsp < 0 or dsp > self.max_disparity:
                    cost += self.plane_penalty
                
                else: 
                    # Target image 의 매칭 포인트
                    match = x + sign * dsp
                    x_match = max(0, min(f1.shape[1]-1, int(match)))

                    wm = 1 - (match - x_match)

                    #오른쪽으로 초과하는 거 방지
                    if x_match > f1.shape[1] - 2:
                        x_match = f1.shape[1] - 2

                    #왼쪽으로 초과하는 거 방지
                    if x_match < 0:
                        x_match = 0
                    

                    mcolo = self.vec_average(f2[y, x_match], f2[y, x_match + 1], wm)
                    mgrad = self.vec_average(g2[y, x_match], g2[y, x_match + 1], wm)

                    w = w1[cy, cx, y-cy+half, x-cx+half]
                    cost += w * self.dissimilarity(f1[y,x], mcolo, g1[y,x], mgrad) # 수식 (3) 중, w(p,q) * p(q, q-(a_f*q_x + b_f*q_y + c_f))

        return cost


    #weight 함수 통해서, self.weights 매트릭스 채워주기
    def precompute_pixels_weights(self, frame, weights, ws):
        half = ws // 2
        rows, cols = frame.shape[:2]

        for cx in range(cols):
            for cy in range(rows):
                for x in range(cx - half, cy + half + 1):
                    for y in range(cy - half, cy + half + 1):
                        if self.inside(x, y, 0, 0, cols, rows):
                            weights[cy, cx, y - cy + half, x - cx + half] = self.weight(frame[cy, cx], frame[y, x], self.gamma)


    #평면 방정식과 픽셀 좌표주고, dispairty 구하는 식
    def planes_to_disparity(self, planes, disp):
        rows, cols = self.rows, self.cols

        for y in range(rows):
            for x in range(cols):
                disp[y,x] = self.disparity(x, y, planes[y,x])


    
    def initialize_random_planes(self, planes, max_d):


