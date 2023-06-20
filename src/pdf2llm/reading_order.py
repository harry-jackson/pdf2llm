from __future__ import annotations
from itertools import chain
from typing import List, Tuple

def directional_gap_function(direction):
    def f(bboxes):
        coords = set()
        for bbox in bboxes:
            coords.add(bbox[direction])     # 1 = y0
            coords.add(bbox[direction + 2]) # 3 = y1
            coords.add(bbox[direction] + 1)  # y0 + 1
            coords.add(bbox[direction + 2] - 1) #y1 - 1

        coords = sorted(coords)
        gaps = []
        for i in range(len(coords)-1):
            if coords[i+1] - coords[i] > 1:  # if there is a gap of more than 1 pixel
                # Check if the gap is horizontal and does not intersect any rectangle
                is_gap = True
                for bbox in bboxes:
                    predicate = bbox[direction] <= coords[i] <= bbox[direction + 2] and bbox[direction] <= coords[i+1] <= bbox[direction + 2]
                    if predicate:
                        is_gap = False
                        break
                if is_gap:
                    gaps.append((coords[i], coords[i + 1]))

        return gaps
    return f

vertical_gaps = directional_gap_function(0)
horizontal_gaps = directional_gap_function(1)
    
def approx_equal(x, y, k = 1):
    return abs(round(x, 0) - round(y, 0)) <= k

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def bbox_to_section(bbox, gaps, direction):
    if len(gaps) == 0:
        return 0
    gaps = [-float('inf')] + gaps + [float('inf')]
    for i, (gap_0, gap_1) in enumerate(zip(gaps[:-1], gaps[1:])):
        if bbox[direction] > gap_0 and bbox[direction + 2] < gap_1:
            return i
    return 1/0
        
def bboxes_to_sections(bboxes, gaps, direction):
    res = [[] for x in range(len(gaps) + 1)]
    for bbox in bboxes:
        index = bbox_to_section(bbox, gaps, direction)
        res[index].append(bbox)
    return res

class PageDivider:
    def __init__(self, bboxes, lines = [], min_width = 0.2):
        self.Boxes = bboxes
        self.Lines = lines
        
        x_min = min([b[0] for b in bboxes])
        x_max = max([b[2] for b in bboxes])
        self.Dividing_Lines = [l for l in lines if (l[0] <= x_min + (x_max - x_min) * 0.25) and (l[2] >= x_max - (x_max - x_min) * 0.25)]

        # FIX - x_max - x_min and y_max - y_min here?
        page_width = x_max + 20
        page_height = max([b[3] for b in bboxes]) + 20
        self.Horizontal_Gaps = horizontal_gaps(bboxes)
        gaps_horizontal = [(g_0 + g_1)/2 for g_0, g_1 in self.Horizontal_Gaps]
        gaps_horizontal_extended = [0] + gaps_horizontal + [page_height]
        self.Heights = [b - a for a, b in zip(gaps_horizontal_extended[:-1], gaps_horizontal_extended[1:])]
        bbox_list = bboxes_to_sections(bboxes, gaps_horizontal, direction = 1)
        self.Blocks = bbox_list
        self.Page_Width = page_width
        self.Min_Width = min_width * page_width
        self.Memo = {}
        self.Distances = {}
        for i, (b_0, b_1) in enumerate(zip(self.Blocks[:-1], self.Blocks[1:])):
            highest_point = max([b[3] for b in b_0])
            lowest_point = min([b[1] for b in b_1])
            self.Distances[(i, i + 1)] = lowest_point - highest_point
            
    def to_tuple(self, i, X):
        return (i,) + tuple(X)
        
    def get_block(self, i):
        return self.Blocks[i]
        
    def already_computed(self, i, X):
        return self.to_tuple(i, X) in self.Memo
    
    def memoization(self, i, X):
        return self.Memo[self.to_tuple(i, X)]
    
    def memoize(self, i, X, score):
        self.Memo[self.to_tuple(i, X)] = score
          
    def too_narrow(self, gaps):
        l = 0
        for gap_0, gap_1 in gaps:
            if (gap_1 - l) < self.Min_Width:
                return True
            elif gap_0 - l < self.Min_Width:
                l  += self.Min_Width
            else:
                l = gap_0
                
        return (self.Page_Width - l) < self.Min_Width
            
    def filter_gaps(self, gaps):
        l = 0
        gaps = []
        for gap_0, gap_1 in gaps:
            if (gap_1 - l) < self.Min_Width:
                continue
            elif gap_0 - l < self.Min_Width:
                l += self.Min_Width
            else:
                l = gap_0
            gaps.append(l, gap_1)
        return gaps
        
    def x_cut(self, i):
        block = self.get_block(i)
        gaps = vertical_gaps(block)
        
        # Make a gap on the right if the rightmost block is
        # too far to the left. 
        # This is for pages with 2 columns where the second 
        # column ends too early. 
        max_x = max([b[2] for b in block])
        if max_x < self.Page_Width * 0.6:
            gaps.append((max_x, self.Page_Width))
        return gaps
        
    def height(self, i):
        return self.Heights[i]
        
    def intersect(self, a, b):
        intersections = []
        for a_start, a_end in a:
            for b_start, b_end in b:
                if max(a_start, b_start) < min(a_end, b_end):
                    intersections.append((max(a_start, b_start), min(a_end, b_end)))
        return intersections
        
    def dist(self, i, j):
        return self.Distances[(i, j)]
        
    def count_blocks(self):
        return len(self.Blocks)
        
    def BC(self, i, X):
        # i is an integer
        # X is a list
        if i >= len(self.Blocks):
            return 0
        
        if self.already_computed(i, X):
            return self.memoization(i, X)
        
        # possible X-cuts in block i alone
        Xi = self.x_cut(i)
        if len(X) == 0:
            # we are not required to share an X-cut
            # with previous block. 
            # either 1) we share the local Xi
            # vertical cut with next block:
            sc1 = self.BC(i + 1, Xi)
            
            if sc1 != 0:
                # if the sharing can be pursued, 
                # reward for height of block i
                sc1 = sc1 + self.height(i)
            
            # or 2) we cut horizontally between 
            # block i and block i+1
            sc2 = self.BC(i + 1, [])
            score = max(sc1, sc2)
            
        else:
            # we are required to share a 
            # vertical cut with previous block(s)
            X_prime = self.intersect(X, Xi)
            if len(X_prime) == 0 or self.too_narrow(X_prime):
                score = 0
                
            else:
                # reward the making of columns
                share_score = 1 / self.dist(i - 1, i) + self.height(i)
                
                # pursue the sharing or not?
                sc1 = self.BC(i + 1, X_prime) # if pursuing the sharing
                sc2 = self.BC(i + 1, []) # score if not pursuing
                
                score = share_score + max(sc1, sc2)
                
        self.memoize(i, X, score)
        return score
    
    def merge_blocks(self):
        mapping_table = {0: 0}
        j = 0
        new_blocks = [self.Blocks[0]]
        for i, block in enumerate(self.Blocks[:-1]):
            score_with_cut = self.BC(i + 1, [])
            score_without_cut = self.BC(i + 1, self.x_cut(i))
            g_0, g_1 = self.Horizontal_Gaps[i]
            dividing_line_exists = any([l[1] >= g_0 and l[3] <= g_1 for l in self.Dividing_Lines])

            if (score_with_cut >= score_without_cut) or (dividing_line_exists):
                new_blocks.append(self.Blocks[i + 1])
            else:
                new_blocks[-1] += self.Blocks[i + 1]
                j += 1
            mapping_table[i + 1] = j
        self.Blocks = new_blocks
        return mapping_table
                
    def divide_into_blocks(self, advanced = True, recursive = 2):
        if advanced:
            self.merge_blocks()
        res = []
        for block in self.Blocks:
            gaps_vertical = [(g_0 + g_1)/2 for g_0, g_1 in vertical_gaps(block)]
            small_blocks = bboxes_to_sections(block, gaps_vertical, direction = 0)
            for small_block in small_blocks:
                if recursive > 0:
                    min_x = min([b[0] for b in small_block])
                    max_x = max([b[2] for b in small_block])
                    
                    lines = [l for l in self.Lines if l[0] <= max_x and l[2] >= min_x]
                    subdivider = PageDivider(small_block, lines = lines)
                    if advanced:
                        subdivider.merge_blocks()
                    small_block = subdivider.divide(advanced = True, recursive = recursive - 1)
                #else:
                #    small_block = sorted(small_block, key = lambda x: x[0])
                res.append(small_block)
        return res
    
    def divide(self, advanced = True, recursive = 2):
        small_blocks = self.divide_into_blocks(advanced = advanced, recursive = recursive)
        res = []
        for small_block in small_blocks:
            for box in small_block:
                res.append(box)
        return res
            
