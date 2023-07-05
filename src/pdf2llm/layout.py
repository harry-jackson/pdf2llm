from __future__ import annotations
import fitz
import layoutparser as lp
import dataclasses
from typing import Dict, Tuple, List
from pdf2llm.reading_order import PageDivider, directional_gap_function

import re

re_units = re.compile('(trillion|million|billion|trn|bln|mln|tn|mn|bn|t|b|m|k)')
re_symbols = re.compile('(%|\(|\)|\-|\,|\.|\—|\–|\$|£|#|\s)')
re_years = re.compile("^((19|\')[7-9][0-9]|(20|\')[0-6][0-9])$")
re_whitespace = re.compile('\s+')
re_numbers = re.compile(r'-?\(?[0-9\.\,]+\)?( ?trillion\b| ?million\b| ?billion\b| ?trn\b| ?bln\b| ?mln\b| ?tn\b| ?mn\b| ?bn\b| ?t\b| ?b\b| ?m\b| ?k\b)?')

def correct_numeric_box(text: str, x0: float, x1: float) -> List[Tuple[str, float, float]]:
    box_width = x1 - x0
    character_width = box_width / len(text)
    return [(m.group(), x0 + character_width * m.start(), x0 + character_width * m.end()) for m in re_numbers.finditer(text)]

def remove_double_spaces(s: str) -> str:
    return re_whitespace.sub(' ', s).strip()

def classify_data_type(s: str) -> str:
    """Get the data type of a string. """
    s = s.lower().strip()
    if s == '':
        return 'blank'
    if re_years.search(s) != None:
        return 'year'
    s = re_units.sub('', s).strip()
    s = re_symbols.sub('', s).strip()
    if s == '':
        return 'symbol'
    elif s == 'na' or s == 'n/a' or s == 'nm' or s.isnumeric():
        return 'numeric'
    else:
        return 'string'

re_not_alpha = re.compile('[^a-zA-Z]')

def lower_case_letters_from_string(s: str) -> str:
    """Return just the letters from a string in lower case, in the same order they appear in the string. Letters can be repeated."""
    return re_not_alpha.sub('', s).lower()

def font_name_to_font_style(s: str) -> str:
    """Translate font name from fitz to a style (e.g. italic, bold)."""
    l = s.split('-')
    if len(l) == 1:
        return 'normal'
    elif 'bold' in l[1].lower():
        return 'bold'
    elif 'ital' in l[1].lower():
        return 'italic'
    else:
        return 'normal'

def sub_boxes_to_text(boxes: List[Box]) -> str:
    return remove_double_spaces(' '.join([box.text() for box in boxes]))

def sub_boxes_to_box_type(sub_boxes: List[str]) -> str:
    box_types = [sub_box.type() for sub_box in sub_boxes]
    if all([box_type == 'title' for box_type in box_types]):
        return 'title'
    elif all([box_type == 'footnote' for box_type in box_types]):
        return 'footnote'
    else:
        return 'text'

def styles_to_box_type(font_style: str, font_size: float, average_font_size: float) -> str:
    if font_size <= 5:
        return 'footnote'
    elif (font_size >= average_font_size + 2.0) or (abs(font_size - average_font_size) < 2.0 and font_style == 'bold'):
        return 'title'
    else: 
        return 'text'

def merge_lines(lines: List[Box]) -> List[Box]:
    if len(lines) == 0:
        return lines
    
    # Sort the lines by y0 and x0 coordinates
    lines.sort(key = lambda l: (l.y0, l.x0))

    merged_lines = []
    current_line = lines[0]

    for next_line in lines[1:]:
        prev_line = current_line

        # Check if rectangles have the same y0 and y1 coordinates and no gap between x coordinates
        if current_line.adjacent_x(next_line) and current_line.adjacent_y(next_line):
            # Merge the rectangles
            current_line = current_line.merge(next_line)
        else:
            # Add the current rectangle to the merged list and update the current rectangle
            merged_lines.append(current_line)
            current_line = next_line

    # Add the last rectangle to the merged list
    merged_lines.append(current_line)

    return merged_lines

def assign_to_group(box: Box, lists: List[List[Box]]) -> int:
    for i, li in enumerate(lists):
        for box_2 in li:
            if box.aligned_y(box_2):
                return i
    return -1

def split(boxes: List[Box]) -> List[Box]:
    res = []
    for box in boxes:
        box_index = assign_to_group(box, res)
        if box_index == -1:
            res.append([box])
        else:
            res[box_index].append(box)

    res_boxes = []
    for r in res:
        res_boxes.append(Box(box_type = 'text', sub_boxes = r))

    return res_boxes

re_cell_symbols = re.compile('[\#\£\$\€\%]')
re_cell_footnotes = re.compile('\([a-z]\)')

def process_cell_text(s: str) -> str:
    s = re_cell_symbols.sub('', s)
    s = re_cell_footnotes.sub('', s)
    s = s.strip()
    return s

def table_boxes_to_text(table_rows: List[Box]) -> str:
    row_text = []
    for row in table_rows:
        cells = row.boxes()
        cell_text = [cell.text() for cell in cells]
        if row.Row_Type == 'data':
            cell_text = [process_cell_text(s) for s in cell_text]
            cell_text = [s for s in cell_text if s != '']

            # FIX - put this somewhere more sensible
            for cell in cells:
                if process_cell_text(cell.text()) == '':
                    cell.Visible = False

        row_text.append('(' + '|'.join(cell_text) + ')')
    res = '\n'.join(row_text)
    return res

def table_header_cutoff(lines: List[Box]) -> int:
    if len(lines) == 0:
        return lines
    # FIX - pass the parent box as the argument to this function to make this easier
    line_lengths = [line.size_x() for line in lines]
    line_x1s = [line.x1 for line in lines]
    min_x0 = min([line.x0 for line in lines])
    max_length = max(line_lengths)
    for line_id, line in enumerate(lines):
        data_types = [classify_data_type(s.text()) for s in line.boxes()]
        n_valid_cells = sum([dc in ('numeric', 'year', 'string') for dc in data_types])
        n_numeric_cells = sum([dc == 'numeric' for dc in data_types])
        if n_valid_cells == 0:
            continue
        elif n_numeric_cells / n_valid_cells >= 0.5:
            break

    while line_id > 0:
        line_id -= 1
        if line_x1s[line_id] - min_x0 > 0.5 * max_length:
            break

    return line_id + 1

class Box:
    """A box containing text. Boxes can have sub boxes inside them."""

    def __init__(self,
                 box_type: str = '',
                 sub_boxes: List[Box] = [],
                 bbox: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0), 
                 text: str = '', 
                 font_name: str = '', 
                 font_size: float = -1.0, 
                 font_color: str = '',
                 average_font_size: float = -1.0,
                 row_type = 'data'):
        
        self.Row_Type = row_type
        self.Font_Style = font_name_to_font_style(font_name)
        self.Font_Size = font_size
        self.Font_Color = font_color
        self.Visible = True

        self.Leaf = sub_boxes == []
        if not self.Leaf:
            self.x0 = min([sub_box.x0 for sub_box in sub_boxes])
            self.y0 = min([sub_box.y0 for sub_box in sub_boxes])
            self.x1 = max([sub_box.x1 for sub_box in sub_boxes])
            self.y1 = max([sub_box.y1 for sub_box in sub_boxes])

            if box_type in ('image', 'drawing', 'table'):
                self.Type = box_type
            else:
                self.Type = sub_boxes_to_box_type(sub_boxes)
                    
            self.Data_Type = 'string'
        else:
            self.x0, self.y0, self.x1, self.y1 = bbox

            if box_type in ('image', 'drawing', 'table'):
                self.Type = box_type

            elif average_font_size > 0 and font_size > 0:
                self.Type = styles_to_box_type(self.Font_Style, self.Font_Size, average_font_size)
            
            else:
                self.Type = 'text'
            
            if self.Font_Color == (0, 0, 0):
                self.Data_Type = 'blank'
            else:
                self.Data_Type = classify_data_type(text)

        self.Text = text

        self.Sub_Boxes = sub_boxes
        
    def bbox(self) -> Tuple(float, float, float, float):
        return (self.x0, self.y0, self.x1, self.y1)

    def size_x(self) -> float:
        return self.x1 - self.x0
    
    def size_y(self) -> float:
        return self.y1 - self.y0
    
    def area(self) -> float:
        return self.size_x() * self.size_y()
    
    def intersection_x(self, box: Box):
        max_x0 = max(self.x0, box.x0)
        min_x1 = min(self.x1, box.x1)

        return max(min_x1 - max_x0, 0)
    
    def intersection_y(self, box: Box):
        max_y0 = max(self.y0, box.y0)
        min_y1 = min(self.y1, box.y1)

        return max(min_y1 - max_y0, 0)

    def intersection_area(self, box: Box):
        return self.intersection_x(box) * self.intersection_y(box)
    
    def intersects_x(self, box: Box) -> bool:
        return self.intersection_x(box) > 0
    
    def intersects_y(self, box: Box) -> bool:
        return self.intersection_y(box) > 0
    
    def intersects(self, box: Box) -> bool:
        return self.intersects_x(box) and self.intersects_y(box)
    
    def adjacent_x(self, box: Box, tolerance: float = 1.0) -> bool:
        return abs(box.x0 - self.x1) <= tolerance
    
    def adjacent_y(self, box: Box, tolerance: float = 1.0) -> bool:
        return abs(box.y0 - self.y1) <= tolerance
    
    def aligned_x(self, box: Box, threshold: float = 0.5) -> bool:
        return self.intersection_x(box) > min(self.size_x(), box.size_x()) * 0.5
    
    def aligned_y(self, box: Box, threshold: float = 0.5) -> bool:
        return self.intersection_y(box) > min(self.size_y(), box.size_y()) * 0.5
    
    def merge(self, box: Box) -> Box:
        if self.is_leaf() and box.is_leaf():
            bbox = (min(self.x0, box.x0), min(self.y0, box.y0), max(self.x1, box.x1), max(self.y1, box.y1))
            text = remove_double_spaces(self.text() + ' ' + box.text())
            return Box(box_type = self.type(), bbox = bbox, text = text)
        
        elif (not self.is_leaf()) and (not box.is_leaf()):
            return Box(sub_boxes = self.boxes() + box.boxes())
        
        else:
            raise Exception('Can only merge leaves with leaves or non-leaves with non-leaves.')


    def is_leaf(self) -> bool:
        return self.Sub_Boxes == []

    def type(self) -> str:
        return self.Type

    def boxes(self) -> List[Box]:
        return self.Sub_Boxes

    def is_title(self) -> bool:
        return self.Type == 'title'
    
    def is_footnote(self) -> bool:
        return self.Type == 'footnote'

    def is_text_box(self) -> bool:
        return self.type() not in ('image', 'drawing')

    def is_table(self):
        return self.type == 'table'

    def is_image_box(self) -> bool:
        return self.type() == 'image'
    
    def is_drawing_box(self) -> bool:
        return self.type() == 'drawing'

    def text_boxes(self) -> List[Box]:
        return [box for box in self.boxes() if box.is_text_box()]

    def image_boxes(self) -> List[Box]:
        return [box for box in self.boxes() if box.is_image_box()]
    
    def drawing_boxes(self) -> List[Box]:
        return [box for box in self.boxes() if box.is_drawing_box()]

    def is_table(self) -> bool:
        return self.type() == 'table'

    def text(self) -> str:
        if self.is_leaf():
            return self.Text
        elif self.is_table():
            return table_boxes_to_text(self.text_boxes())
        else:
            return sub_boxes_to_text(self.text_boxes())

    def text_length(self) -> int:
        return len(self.text())
    
    def leaves(self) -> List[Box]:
        if self.is_leaf():
            return [self]
        else:
            res = []
            for box in self.boxes():
                res += box.leaves()
            return res
        
    def add_sub_box(self, box: Box):
        self.Leaf = False
        self.Sub_Boxes = self.Sub_Boxes + [box]
        self.x0 = min(self.x0, box.x0)
        self.y0 = min(self.y0, box.y0)
        self.x1 = max(self.x1, box.x1)
        self.y1 = max(self.y1, box.y1)

    def side_lower(self) -> Box:
        return Box(box_type = 'drawing', bbox = (self.x0, self.y1, self.x1, self.y1))
    
    def side_upper(self) -> Box:
        return Box(box_type = 'drawing', bbox = (self.x0, self.y0, self.x1, self.y0))
    
    def side_left(self) -> Box:
        return Box(box_type = 'drawing', bbox = (self.x0, self.y0, self.x0, self.y1))
    
    def side_right(self) -> Box:
        return Box(box_type = 'drawing', bbox = (self.x1, self.y0, self.x1, self.y1))
    
    def set_box_type(self, new_box_type: str):
        self.Type = new_box_type

    def sort_horizontal(self):
        self.Sub_Boxes = sorted(self.Sub_Boxes, key = lambda box: (box.x0 + box.x1) / 2)

    def sort_vertical_start(self):
        self.Sub_Boxes = sorted(self.Sub_Boxes, key = lambda box: box.y1)

    def sort_vertical_mean(self):
        self.Sub_Boxes = sorted(self.Sub_Boxes, key = lambda box: (box.y0 + box.y1) / 2)

    def merge_sub_boxes(self):
        if len(self.boxes()) == 0:
            return
        
        res = []
        A = self.Sub_Boxes[0]
        for B in self.Sub_Boxes[1:]:
            if A.adjacent_x(B, tolerance = 2.0):
                A = A.merge(B)
            else:
                res.append(A)
                A = B

        res.append(A)
        self.Sub_Boxes = res

    def group_into_rows(self):
        
        self.sort_vertical_start()
        self.Sub_Boxes = split(self.Sub_Boxes)

        for sub_box in self.Sub_Boxes:
            sub_box.sort_horizontal()

        self.sort_vertical_mean()

    def sort_reading_order(self):
        # Note: this also gets rid of image and drawing boxes
        boxes = self.text_boxes() + self.image_boxes()
        drawing_boxes = self.drawing_boxes()

        lines = merge_lines([d.side_upper() for d in drawing_boxes] + [d.side_lower() for d in drawing_boxes])
        
        # FIX - this should work differently for tables I think. 
        #text_leaves = [leaf for leaf in self.leaves() if leaf.is_text_box()]
        lines = [line for line in lines if not any([line.intersects(box) for box in boxes])]

        box_tuples = [(round(box.x0, 1), round(box.y0, 1), round(box.x1, 1), round(box.y1, 1), i) for i, box in enumerate(boxes)]
        line_tuples = [(round(line.x0, 1), round(line.y0, 1), round(line.x1, 1), round(line.y1, 1)) for line in lines]
        if len(box_tuples) == 0:
            self.Sub_Boxes = []
            return
        divider = PageDivider(box_tuples, lines = line_tuples)
        new_box_tuples = divider.divide()

        new_ids = [x[-1] for x in new_box_tuples]
        order_dict = {id: index for index, id in enumerate(new_ids)}
        id_boxes = [(i, box) for i, box in enumerate(boxes)]
        sorted_boxes = sorted(id_boxes, key=lambda x: order_dict[x[0]])
        sorted_boxes = [box for _, box in sorted_boxes if box.is_text_box()]

        self.Sub_Boxes = sorted_boxes
    
    def format_table(self, drawing_boxes):
        assert self.type() == 'table'
        
        # check if it's a text table
        is_text_table = True
        # remove blanks, symbols and footnotess
        self.Sub_Boxes = [box for box in self.leaves() if (box.Data_Type not in ('blank', 'symbol')) and (box.Font_Size > 5.9)]
        if len(self.boxes()) == 0:
            return
        for box in self.Sub_Boxes:
            box.Sub_Boxes = []

        new_sub_boxes = []

        for box in self.boxes():
            if box.Data_Type == 'numeric':
                corrected_boxes = correct_numeric_box(box.text(), box.x0, box.x1)
                for cb in corrected_boxes:
                    new_box = Box(box_type = 'numeric', sub_boxes = [], bbox = (cb[1], box.y0, cb[2], box.y1),
                                  text = cb[0], font_name = box.Font_Style, font_size = box.Font_Size, font_color = box.Font_Color)
                    new_sub_boxes.append(new_box)
            else:
                new_sub_boxes.append(box)

        self.Sub_Boxes = new_sub_boxes

        n_numeric_cells = sum([box.Data_Type == 'numeric' for box in self.boxes()])
        n_valid_cells = sum([box.Data_Type in ('numeric', 'year', 'string') for box in self.boxes()])
        if n_valid_cells > 0:
            is_text_table = n_numeric_cells / n_valid_cells < 0.1

        self.group_into_rows()
        
        if is_text_table:
            cutoff = len(self.boxes())
        else:
            cutoff = table_header_cutoff(self.boxes())


        header = Box(box_type = 'text', sub_boxes = self.boxes()[:cutoff])
        data = Box(box_type = 'text', sub_boxes = self.boxes()[cutoff:]) 
        
        header_tuples = [(round(box.x0, 1), round(box.y0, 1), round(box.x1, 1), round(box.y1, 1), {'text': box.text(), 'size': box.Font_Size}) for i, box in enumerate(header.leaves())]

        lines = merge_lines([d.side_upper() for d in drawing_boxes] + [d.side_lower() for d in drawing_boxes])
        line_tuples = [(round(line.x0, 1), round(line.y0, 1), round(line.x1, 1), round(line.y1, 1)) for line in lines]

        divider = PageDivider(header_tuples, lines = line_tuples, min_width = 0, all_lines = True)
        blocks = divider.divide_into_blocks(advanced = False)

        header = Box(box_type = 'text', bbox = (header.x0, header.y0, header.x1, header.y1))
        new_boxes = []
        for block in blocks:
            new_box = Box(box_type = 'text',
                          row_type = 'header', 
                          bbox = (min([b[0] for b in block]), min([b[1] for b in block]), max([b[2] for b in block]), max([b[3] for b in block])), 
                          text = remove_double_spaces(' '.join([b[4]['text'] for b in block])),
                          font_size = max(b[4]['size'] for b in block))
            new_boxes.append(new_box)
        
        header = Box(box_type = 'text', sub_boxes = new_boxes)

        header.group_into_rows()
        for row_box in header.boxes():
            row_box.Row_Type = 'header'

        for row_box in data.boxes():
            row_box.Row_Type = 'data'

        # get columns
        find_vertical_gaps = directional_gap_function(0)
        data_boxes = []
        for row in data.boxes():
            if len(row.boxes()) == 1:
                continue
            proportion_numeric = len([b for b in row.boxes() if b.Data_Type == 'numeric']) / len(row.boxes())
            if proportion_numeric < 0.5:
                continue
            data_boxes += [b.bbox() for b in row.boxes()]
            
        vertical_gaps = find_vertical_gaps(data_boxes)

        vertical_gaps = [(0.0, self.x0)] + vertical_gaps + [(self.x1, float('inf'))]
        column_bounds = [(a[1], b[0]) for a, b in zip(vertical_gaps[:-1], vertical_gaps[1:])]

        column_boxes = [Box(bbox = (x0, self.y0, x1, self.y1)) for x0, x1 in column_bounds]

        self.Header_Cutoff = len(header.boxes())
        
        if len(data.boxes()) > 0:
            self.Sub_Boxes = header.merge(data).boxes()
        else:
            self.Sub_Boxes = header.boxes()

        for row in self.Sub_Boxes:
            if len(row.boxes()) >= len(column_boxes):
                continue
            for i, column in enumerate(column_boxes):
                if not any([column.intersects(box) for box in row.boxes()]):
                    empty_box = Box(box_type = 'numeric', bbox = (column.x0, row.y0, column.x1, row.y1), text = '-')
                    row.Sub_Boxes = row.boxes()[:i] + [empty_box] + row.boxes()[i:]

        
        


        


              