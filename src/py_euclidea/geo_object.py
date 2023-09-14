import numpy as np
from skimage import draw
import cairo
import string

epsilon = 0.00001
def eps_zero(npa):
    result = np.abs(npa) < epsilon
    if not isinstance(result, bool): result = result.all()
    return result
def eps_identical(npa1, npa2):
    return eps_zero(npa1-npa2)
def eps_smaller(x1, x2):
    return x1 + epsilon < x2
def eps_bigger(x1, x2):
    return eps_smaller(x2, x1)

def vector_perp_rot(vec):
    return np.array((vec[1], -vec[0]))

def corners_to_rectangle(corners):
    size = corners[1] - corners[0]
    return list(corners[0])+list(size)

class GeoObject:
    def get_name(self):
        raise Exception("Not implemented")
    def identical_to(self, x):
        if not type(self) == type(x): return False
        return eps_identical(x.data, self.data)
    #def __eq__(self, other):
    #    return self.identical_to(other)

    def dist_from(self, np_point):
        raise Exception("Not implemented")
    def draw(self, cr, corners, scale, visualisation=False):
        raise Exception("Not implemented")
    def scale(self, scale):
        raise Exception("Not implemented")
    def shift(self, shift):
        raise Exception("Not implemented")
    def get_bounding_box(self):
        raise Exception("Not implemented")
    def copy(self):
        raise Exception("Not implemented")
    def get_mask(self,corners, out_size, scale, mask_size):
        raise Exception("Not implemented")


class Point(GeoObject):
    def get_name(self):
        return "Point"
    def __init__(self, coor):
        self.a = np.array(coor)
        self.data = self.a
    def copy(self):
        return Point(self.data)
    def dist_from(self, np_point):
        return np.linalg.norm(self.a - np_point)
    def get_bounding_box(self):
        return np.array([self.a, self.a])

    def set_index(self, index):
        self.index = index

    def set_hidden(self, hidden):
        self.hidden = hidden

    def draw(self, cr, corners, scale, visualisation=False, point_index=0):
        #cr.arc(self.a[0], self.a[1], 10, 0, 2*np.pi)
        #cr.set_source_rgb(1,1,1)
        #cr.fill()
        width = 5
        cr.arc(self.a[0], self.a[1], width/scale, 0, 2*np.pi)
        #cr.set_source_rgb(0,0,0)
        cr.fill()
        if point_index > 0:
            self.draw_label(cr, scale, point_index)
    def draw_label(self, cr, scale, point_index):

        ALPHABET = list(string.ascii_uppercase)
        cr.set_source_rgb(0, 0, 0)  # Set color to black
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(30)
        cr.move_to(self.a[0] + 10 / scale, self.a[1])
        cr.show_text(ALPHABET[point_index - 1])
    def scale(self, scale):
        self.a *= scale
    def shift(self, shift):
        self.a += shift

    def __repr__(self):
        return "Point({}, {})".format(*self.a)
    def __str__(self):
        return self.__repr__()
    def get_mask(self,corners,out_size,scale, mask_size):
        mask = np.zeros(out_size)
        coors = (self.a / scale).astype(np.int32)

        center_row = coors[0]
        center_col = coors[1]
        radius = 2 * mask_size

        # rr, cc = draw.ellipse(center_row, center_col, radius, radius, shape=mask.shape)
        rr, cc = draw.circle(coors[0], coors[1], radius=2*mask_size, shape=mask.shape)
        mask[rr, cc] = 1
        return mask

class PointSet(GeoObject):
    def contains(self, np_point):
        raise Exception("Not implemented")
    def closest_on(self, np_point):
        raise Exception("Not implemented")
    def set_index(self, index):
        self.index = index

    def set_hidden(self, hidden):
        self.hidden = hidden

class Circle(PointSet):
    def get_name(self):
        return "Circle"
    def __init__(self, center, r):
        assert(r > 0)
        self.c = np.array(center)
        self.r = r
        self.compute_data()

    def copy(self):
        return Circle(self.c, self.r)

    def compute_data(self):
        self.data = np.concatenate([self.c, [self.r]])
        self.r_squared = self.r**2
    def dist_from(self, np_point):
        center_dist = np.linalg.norm(self.c - np_point)
        return abs(self.r-center_dist)
    def get_bounding_box(self):
        bb_shift = np.array([self.r, self.r])
        return np.array([self.c - bb_shift, self.c + bb_shift])

    def contains(self, np_point):
        return abs(np.linalg.norm(np_point-self.c)-self.r) < epsilon

    def closest_on(self, np_point):
        vec = np_point-self.c
        vec *= self.r/np.linalg.norm(vec)
        return self.c + vec

    def draw(self, cr, corners, scale, visualisation=False):
        cr.arc(self.c[0], self.c[1], self.r, 0, 2*np.pi)
        #cr.set_source_rgb(0,0,0)
        if visualisation:
            cr.set_line_width(2/scale)
        else:
            cr.set_line_width(1/scale)
        cr.stroke()

    def scale(self, scale):
        self.c *= scale
        self.r *= scale
        self.compute_data()
    def shift(self, shift):
        self.c += shift
        self.compute_data()
    def get_mask(self,corners,out_size, scale,mask_size):
        mask = np.zeros(out_size, dtype=np.uint8)
        coors = np.array(self.c / scale, np.int32)
        rr, cc = draw.circle(coors[0], coors[1], radius=int(self.r / scale[0] + mask_size), shape=mask.shape)
        rr_c, cc_c = draw.circle(coors[0], coors[1], radius=int(self.r / scale[0] - mask_size), shape=mask.shape)
        mask[rr, cc] = 1
        mask[rr_c, cc_c] = 0
        return mask

class Line(PointSet):
    def get_name(self):
        return "Line"
    def __init__(self, normal_vector, c): # [x,y] in Line([a,b],c) <=> xa + yb == c
        assert((normal_vector != 0).any())
        normal_size = np.linalg.norm(normal_vector)
        normal_vector = normal_vector / normal_size
        c = c / normal_size
        self.n = normal_vector
        self.v = vector_perp_rot(normal_vector)
        self.c = c
        self.compute_data()
    def copy(self):
        return Line(self.n, self.c)
    def compute_data(self):
        self.data = np.concatenate([self.n, [self.c]])

    def get_bounding_box(self):
        return None
    def get_endpoints(self, corners):

        result = [None, None]
        boundaries = list(zip(*corners))
        if np.prod(self.n) > 0:
            #print('swap')
            boundaries[1] = boundaries[1][1], boundaries[1][0]

        for coor in (0,1):
            if self.n[1-coor] == 0: continue
            for i, bound in enumerate(boundaries[coor]):
                p = np.zeros([2])
                p[coor] = bound
                p[1-coor] = (self.c - bound*self.n[coor])/self.n[1-coor]
                #print(p)
                #print("({} - {}) * ({} - {} = {})".format(
                #    p[1-coor], boundaries[1-coor][0], p[1-coor], boundaries[1-coor][1],
                #    (p[1-coor] - boundaries[1-coor][0]) * (p[1-coor] - boundaries[1-coor][1]),
                #))
                if (p[1-coor] - boundaries[1-coor][0]) * (p[1-coor] - boundaries[1-coor][1]) <= 0:
                    result[i] = p

        if result[0] is None or result[1] is None: return None
        else: return result

    def dist_from(self, np_point):
        c2 = np.dot(self.n, np_point)
        return abs(c2-self.c)

    def contains(self, np_point):
        return abs(np.dot(np_point, self.n)-self.c) < epsilon

    def closest_on(self, np_point):
        c2 = np.dot(self.n, np_point)
        return np_point - (c2-self.c)*self.n

    def identical_to(self, x):
        if not isinstance(x, Line): return False
        if eps_identical(x.data, self.data): return True
        if eps_identical(x.data, -self.data): return True
        return False

    def draw(self, cr, corners, scale, visualisation=False):
        endpoints = self.get_endpoints(corners)
        if endpoints is None: return

        cr.move_to(*endpoints[0])
        cr.line_to(*endpoints[1])

        #cr.set_source_rgb(0,0,0)
        if visualisation:
            cr.set_line_width(2 / scale)
        else:
            cr.set_line_width(1/scale)
        cr.stroke()

    def scale(self, scale):
        self.c *= scale
        self.compute_data()
    def shift(self, shift):
        self.c += np.dot(shift, self.n)
        self.compute_data()

    def get_mask(self, corners, out_size, scale, mask_size):
        mask = np.zeros(out_size, dtype=np.uint8)
        end_points = np.array(self.get_endpoints(corners) / scale,np.int32)
        for i in range(-mask_size, mask_size):

            rr, cc = draw.line(int(np.clip(end_points[0][0] + i, 0, out_size[0]-1)),
                               int(np.clip(end_points[0][1], 0, out_size[1]-1)),
                               int(np.clip(end_points[1][0] + i, 0, out_size[0]-1)),
                               int(np.clip(end_points[1][1], 0, out_size[1]-1)))
            mask[rr, cc] = 1
            rr, cc = draw.line(int(np.clip(end_points[0][0], 0, out_size[0]-1)),
                               int(np.clip(end_points[0][1] + i, 0, out_size[1]-1)),
                               int(np.clip(end_points[1][0], 0, out_size[0]-1)),
                               int(np.clip(end_points[1][1] + i, 0, out_size[1]-1)))
            mask[rr, cc] = 1
        return mask

class Segment(Line):
    def __init__(self, p1, p2): # [x,y] in Line([a,b],c) <=> xa + yb == c
        assert((p1 != p2).any())
        normal_vec = vector_perp_rot(p1-p2)
        c = np.dot(p1, normal_vec)
        Line.__init__(self, normal_vec, c)
        self.end_points = np.array([p1, p2])
        self.length = np.linalg.norm(p1-p2)
    def copy(self):
        return Segment(self.end_points[0],self.end_points[1])

    def get_bounding_box(self):
        return np.array([
            np.min(self.end_points, axis = 0),
            np.max(self.end_points, axis = 0)
        ])
    def get_endpoints(self, corners):
        return self.end_points

    def contains(self, x):
        if not Line.contains(self, x): return False
        p1, p2 = self.end_points
        for x in (np.dot(p2-p1, x-p1), np.dot(p1-p2, x-p2)):
            if x < 0 and not np.isclose(x,0): return False
        return True

    def dist_from(self, x):
        p1, p2 = self.end_points
        if np.dot(p2-p1, x-p1) < 0: return np.linalg.norm(x-p1)
        elif np.dot(p1-p2, x-p2) < 0: return np.linalg.norm(x-p2)
        else: return Line.dist_from(self, x)

    def closest_on(self, np_point):
        cand = Line.closest_on(self, np_point)
        if self.contains(cand): return cand
        else: return min(self.end_points, key = lambda x: np.linalg.norm(x - np_point))

    def scale(self, scale):
        self.end_points *= scale
        self.length *= scale
        Line.scale(self, scale)
    def shift(self, shift):
        self.end_points += shift
        Line.shift(self, shift)

class Ray(Line):
    def __init__(self, start_point, vec):
        normal_vec = -vector_perp_rot(vec)
        c = np.dot(start_point, normal_vec)
        Line.__init__(self, normal_vec, c)
        self.start_point = np.array(start_point)
    def copy(self):
        return Ray(self.start_point, vector_perp_rot(self.n))

    def get_bounding_box(self):
        return np.array([self.start_point, self.start_point])
    def get_endpoints(self, corners):
        line_endpoints = Line.get_endpoints(self, corners)
        if line_endpoints is None: return None
        pos_endpoints = [
            point
            for point in line_endpoints
            if np.dot(self.v, point - self.start_point) > 0
        ]
        if len(pos_endpoints) == 0: return None
        elif len(pos_endpoints) == 1:
            return [self.start_point, pos_endpoints[0]]
        else: return pos_endpoints

    def contains(self, x):
        if not Line.contains(self, x): return False
        return np.dot(self.v, x-self.start_point) >= 0

    def dist_from(self, x):
        if np.dot(self.v, x - self.start_point) >= 0: return Line.dist_from(self, x)
        else: return np.linalg.norm(x - self.start_point)

    def closest_on(self, np_point):
        cand = Line.closest_on(self, np_point)
        if self.contains(cand): return cand
        else: return self.start_point

    def scale(self, scale):
        self.start_point *= scale
        Line.scale(self, scale)
    def shift(self, shift):
        self.start_point += shift
        Line.shift(self, shift)
