import numpy as np
import cv2
import math
from tqdm import tqdm

class Matrix2D:
    def __init__(self, rows=0, cols=0, default_value=None):
        self.rows = rows
        self.cols = cols
        self.data = [[default_value] * cols for _ in range(rows)]

    def __getitem__(self, index):
        row, col = index
        return self.data[row][col]
    
    def __setitem__(self, index, value):
        row, col = index
        self.data[row][col] = value


class Plane:
    def __init__(self, point=None, normal=None, coeff=None):
        if point is not None and normal is not None:
            self.point = point
            self.normal = normal
            a = - normal[0] / normal[2]
            b = - normal[1] / normal[2]
            c = np.dot(normal, point) / normal[2]
            self.coeff = np.array([a, b, c])
        
        elif coeff is not None:
            self.coeff = coeff
        
        else:
            self.point = None
            self.normal = None
            self.coeff = None
        
    def __getitem__(self, idx):
        return self.coeff[idx]

    def __call__(self):
        return self.coeff
    
    def get_point(self):
        return self.point

    def get_normal(self):
        return self.normal
    
    def get_coeff(self):
        return self.coeff
    
    def view_transform(self, x, y, sign):
        z = self.coeff[0] * x + self.coeff[1] * y + self.coeff[2]
        qx = x + sign * z
        qy = y
        p = np.array([qx, qy, z])
        return Plane(point = p, normal = self.normal)


class PatchMatch:

    ############################################## 공통 사용 함수
    def __init__(self, alpha, gamma, tau_c, tau_g):
        self.alpha = alpha
        self.gamma = gamma
        self.tau_c = tau_c
        self.tau_g = tau_g
        self.window_size = 35
        self.max_disparity = 60
        self.plane_penalty = 120
        self.views = [None, None]
        self.weights = [None, None]
        self.grads = [None, None]
        self.planes = [None, None]
        self.costs = [None, None]
        self.disps = [None, None]

    def inside(self, x, y, lbx, lby, ubx, uby):
        return lbx <= x < ubx and lby <= y < uby

    def dissimilarity(self, pp, qq, pg, qg):
        cost_c = np.linalg.norm(pp - qq, ord=1) #두 점의 색상간의 불일치
        cost_g = np.linalg.norm(pg - qg, ord=1) #두 점의 기하학적 거리
        cost_c = min(cost_c, self.tau_c)
        cost_g = min(cost_g, self.tau_g)
        return (1- self.alpha) * cost_c + self.alpha * cost_g

    # 수식 (4)
    def weight(self, p: np.ndarray, q: np.ndarray, gamma = 10.0):
        return np.exp(-np.linalg.norm(p - q, ord=1) / gamma)

    def disparity(self, x, y, p):
        return p[0] * x + p[1] * y + p[2]
    
    def vec_average(self, x, y, wx):
        return wx * x + (1 - wx) * y

    def plane_match_cost(self, p, cx, cy, ws, cpv): #평면 매칭 비용 계산
        sign = -1 + 2 * cpv
        cost = 0
        half = ws // 2

        f1, f2 = self.views[cpv], self.views[1-cpv]
        g1, g2 = self.grads[cpv], self.grads[1-cpv]

        w1 = self.weights[cpv]

        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if not self.inside(x, y, 0, 0, f1.shape[1], f1.shape[0]):
                    continue
                dsp = self.disparity(x,y,p)

                if dsp < 0 or dsp > self.max_disparity:
                    cost += self.plane_penalty
                
                else:
                    match = x + sign * dsp
                    x_match = max(0, min(f1.shape[1] - 1, int(match)))
                    wm = 1 - (match - x_match)

                    if x_match > f1.shape[1] - 2:
                        x_match = f1.shape[1] - 2
                    if x_match < 0:
                        x_match = 0
                    
                    mcolo = self.vec_average(f2[y, x_match], f2[y, x_match + 1], wm)
                    mgrad = self.vec_average(g2[y, x_match], g2[y, x_match + 1], wm)

                    w = w1[cy, cx, y - cy + half, x - cx + half]
                    cost += w * self.dissimilarity(f1[y, x], mcolo, g1[y, x], mgrad)

        return cost
    ############################################## 공통 사용 함수



    ############################################## 1. Set 과정
    def set(self, img1, img2):
        self.views[0] = img1
        self.views[1] = img2

        self.rows, self.cols = img1.shape[:2]

        # 픽셀 가중치 계산
        print("Precomputing pixels weight...")
        # self.weights[0] = np.zeros((self.rows, self.cols, self.window_size, self.window_size), dtype=np.float32)
        # self.weights[1] = np.zeros((self.rows, self.cols, self.window_size, self.window_size), dtype=np.float32)
        # self.precompute_pixels_weights(img1, self.weights[0], self.window_size)
        # self.precompute_pixels_weights(img2, self.weights[1], self.window_size)

        self.weights[0] = np.ones((self.rows, self.cols, self.window_size, self.window_size), dtype=np.float32) #{임시}
        self.weights[1] = np.ones((self.rows, self.cols, self.window_size, self.window_size), dtype=np.float32) #{임시}

        # 그레이스케일 영상의 그래디언트 계산
        print("Evaluating images gradient...")
        self.grads[0] = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.grads[1] = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.compute_greyscale_gradient(img1, self.grads[0])
        self.compute_greyscale_gradient(img2, self.grads[1])

        # 픽셀 평면 초기화
        print("Precomputing random planes...")
        self.planes[0] = Matrix2D(self.rows, self.cols)
        self.planes[1] = Matrix2D(self.rows, self.cols)
        self.initialize_random_planes(self.planes[0], self.max_disparity)
        self.initialize_random_planes(self.planes[1], self.max_disparity)

        # 초기 평면 비용 계산
        print("Evaluating initial planes cost...")
        # self.costs[0] = np.zeros((self.rows, self.cols), dtype=np.float32)
        # self.costs[1] = np.zeros((self.rows, self.cols), dtype=np.float32)
        # self.evaluate_planes_cost(0)
        # self.evaluate_planes_cost(1)

        self.costs[0] = np.random.uniform(0, 100, size=(self.rows, self.cols)) #{임시}
        self.costs[1] = np.random.uniform(0, 100, size=(self.rows, self.cols)) #{임시}

        # 좌우 시차 맵 초기화
        self.disps[0] = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.disps[1] = np.zeros((self.rows, self.cols), dtype=np.float32)


    #Adaptive Weight 계산해서 저장
    def precompute_pixels_weights(self, frame, weights, ws):
        half = ws // 2
        rows, cols = frame.shape[:2]

        for cx in tqdm(range(cols)):
            for cy in range(rows):
                for x in range(cx - half, cx + half + 1):
                    for y in range(cy - half, cy + half + 1):
                        if self.inside(x, y, 0, 0, cols, rows):
                            weights[cy, cx, y - cy + half, x - cx + half] = self.weight(frame[cy, cx], frame[y, x], self.gamma)



    def compute_greyscale_gradient(self, frame, grad):
        scale = 1
        delta = 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x_grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        y_grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        
        x_grad = x_grad / 8.0
        y_grad = y_grad / 8.0
        
        grad[:, :, 0] = x_grad
        grad[:, :, 1] = y_grad

    def initialize_random_planes(self, planes, max_d):
        rows, cols = planes.rows, planes.cols

        for y in range(rows):
            for x in range(cols):
                z = np.random.uniform(0, max_d)
                point = np.array([x, y, z])

                nx = np.random.uniform(-1, 1)
                ny = np.random.uniform(-1, 1)
                nz = np.random.uniform(-1, 1)
                normal = np.array([nx, ny, nz])
                normal /= np.linalg.norm(normal)

                planes[y, x] = Plane(point=point, normal=normal)
    
    def evaluate_planes_cost(self, cpv):
        rows, cols = self.rows, self.cols

        for y in tqdm(range(rows)):
            for x in range(cols):
                self.costs[cpv][y, x] = self.plane_match_cost(self.planes[cpv][y, x], x, y, self.window_size, cpv)

    ############################################## 1. Set 과정


    
    ############################################## 2. Process 과정

    def process(self, iterations, reverse=False):
        print("2. Processing")
        for iter in tqdm(range(iterations)):
            iter_type = (iter % 2 == 0)

            for work_view in range(2):
                if iter_type:
                    self.process_pixels(work_view, iter, 0, self.rows, 0, self.cols, 1)
                else:
                    self.process_pixels(work_view, iter, self.rows - 1, -1, self.cols - 1, -1, -1)

        self.planes_to_disparity(self.planes[0], self.disps[0])
        self.planes_to_disparity(self.planes[1], self.disps[1])


    def planes_to_disparity(self, planes, disp):
        rows, cols = self.rows, self.cols
        for y in range(rows):
            for x in range(cols):
                disp[y, x] = self.disparity(x, y, planes[y, x])


    def process_pixels(self, work_view, iter, y_start, y_end, x_start, x_end, step):
        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                self.process_pixel(x, y, work_view, iter)


    def process_pixel(self, x, y, cpv, iter):
        self.spatial_propagation(x, y, cpv, iter)
        self.plane_refinement(x, y, cpv, self.max_disparity / 2, 1.0, 0.1)
        self.view_propagation(x, y, cpv)

    
    def spatial_propagation(self, x, y, cpv, iter):
        rows, cols = self.rows, self.cols
        offsets = [(0, -1), (-1, 0)] if iter % 2 == 0 else [(0, 1), (1, 0)]

        old_plane = self.planes[cpv][y, x]
        old_cost = self.costs[cpv][y, x]

        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if not self.inside(nx, ny, 0, 0, cols, rows):
                continue

            p_neigh = self.planes[cpv][ny, nx]
            new_cost = self.plane_match_cost(p_neigh, x, y, self.window_size, cpv)

            if new_cost < old_cost:
                old_plane = p_neigh
                old_cost = new_cost

        self.planes[cpv][y, x] = old_plane
        self.costs[cpv][y, x] = old_cost
    

    def plane_refinement(self, x, y, cpv, max_delta_z, max_delta_n, end_dz):
        old_plane = self.planes[cpv][y, x]
        old_cost = self.costs[cpv][y, x]

        while max_delta_z >= end_dz:
            delta_z = np.random.uniform(-max_delta_z, max_delta_z)
            z = old_plane[0] * x + old_plane[1] * y + old_plane[2]
            new_point = np.array([x, y, z + delta_z])

            n = old_plane.get_normal()
            delta_n = np.random.uniform(-max_delta_n, max_delta_n, size=3)
            new_normal = n + delta_n
            new_normal /= np.linalg.norm(new_normal)

            new_plane = Plane(point=new_point, normal=new_normal)
            new_cost = self.plane_match_cost(new_plane, x, y, self.window_size, cpv)

            if new_cost < old_cost:
                old_plane = new_plane
                old_cost = new_cost

            max_delta_z /= 2.0
            max_delta_n /= 2.0

        self.planes[cpv][y, x] = old_plane
        self.costs[cpv][y, x] = old_cost


    def view_propagation(self,x, y, cpv):
        sign = -1 if cpv == 0 else 1
        view_plane = self.planes[cpv][y, x]

        mx, my = int(x + sign * self.disparity(x, y, view_plane)), y #int 로 바꿔줌 좌표 이산화 때문에

        if not self.inside(mx, my, 0, 0, self.cols, self.rows):
            return

        new_plane = view_plane.view_transform(x, y, sign) #평면의 방정식을 통해 변환 이전 좌표의 평면을 구하기 위함

        #import pdb; pdb.set_trace()
        old_cost = self.costs[1 - cpv][my, mx]
        new_cost = self.plane_match_cost(new_plane, mx, my, self.window_size, 1 - cpv)

        if new_cost < old_cost:
            self.planes[1 - cpv][my, mx] = new_plane
            self.costs[1 - cpv][my, mx] = new_cost


    def viewTransform(self, x, y, sign): 
        z = self.coeff[0] * x + self.coeff[1] * y + self.coeff[2]
        qx = x + sign * z
        qy = y
        p = np.array([qx, qy, z])
        return Plane(p, self.normal)

    ############################################## 2. Process 과정




    ############################################## 3. Post process 과정
    def postprocess(self):
        print("3. Post Processing")

        lft_validity = np.zeros((self.rows, self.cols), dtype=bool)
        rgt_validity = np.zeros((self.rows, self.cols), dtype=bool)

        for y in tqdm(range(self.rows)):
            for x in range(self.cols):
                x_rgt_match = max(0, min(self.cols - 1, int(x - self.disps[0][y, x])))
                lft_validity[y, x] = abs(self.disps[0][y, x] - self.disps[1][y, x_rgt_match]) <= 1

                x_lft_match = max(0, min(self.rows - 1, int(x + self.disps[1][y, x])))
                rgt_validity[y, x] = abs(self.disps[1][y, x] - self.disps[0][y, x_lft_match]) <= 1

        for y in tqdm(range(self.rows)):
            for x in range(self.cols):
                if not lft_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes[0], lft_validity)
                if not rgt_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes[1], rgt_validity)

        self.planes_to_disparity(self.planes[0], self.disps[0])
        self.planes_to_disparity(self.planes[1], self.disps[1])

        for y in tqdm(range(self.rows)):
            for x in range(self.cols):
                self.weighted_median_filter(x, y, self.disps[0], self.weights[0], lft_validity, self.window_size, False)
                self.weighted_median_filter(x, y, self.disps[1], self.weights[1], rgt_validity, self.window_size, False)



    def fill_invalid_pixels(self, y, x, planes, validity):
        x_lft = x - 1
        x_rgt = x + 1

        while x_lft >= 0 and not validity[y, x_lft]:
            x_lft -= 1

        while x_rgt < self.cols and not validity[y, x_rgt]:
            x_rgt += 1

        best_plane_x = x

        if 0 <= x_lft and x_rgt < self.cols:
            disp_l = self.disparity(x, y, planes[y, x_lft])
            disp_r = self.disparity(x, y, planes[y, x_rgt])
            best_plane_x = x_lft if disp_l < disp_r else x_rgt
        elif 0 <= x_lft:
            best_plane_x = x_lft
        elif x_rgt < self.cols:
            best_plane_x = x_rgt

        planes[y, x] = planes[y, best_plane_x]


    def weighted_median_filter(self, cx, cy, disparity, weights, valid, ws, use_invalid):
        half = ws // 2
        w_tot = 0
        disps_w = []

        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if self.inside(x, y, 0, 0, self.cols, self.rows) and (use_invalid or valid[y, x]):
                    w = weights[cy, cx, y - cy + half, x - cx + half]
                    w_tot += w
                    disps_w.append((w, disparity[y, x]))

        disps_w.sort()
        med_w = w_tot / 2.0

        w = 0
        for dw in disps_w:
            w += dw[0]
            if w >= med_w:
                if dw == disps_w[0]:
                    disparity[cy, cx] = dw[1]
                else:
                    disparity[cy, cx] = (disps_w[disps_w.index(dw) - 1][1] + dw[1]) / 2.0
                break

    ############################################## 3. Post process 과정


    ############################################## 4. 마무리 과정

    def get_left_disparity_map(self):
        return self.disps[0]

    def get_right_disparity_map(self):
        return self.disps[1]