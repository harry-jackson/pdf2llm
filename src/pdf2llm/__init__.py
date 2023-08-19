from __future__ import annotations
from pdf2llm.layout import Box, classify_data_type, correct_numeric_box
from pdf2llm.contents import contents_list
from pdf2llm.search import VectorStore, make_simple_search_function, make_boolean_search_function, make_regex_search_function
import fitz
import json
import numpy as np
from PIL import Image
from typing import List, Optional, Set
import os
import re
import openai
from dataclasses import dataclass
from time import sleep
from huggingface_hub import hf_hub_download
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import torch

re_not_alpha = re.compile('[^a-zA-Z]')

def flatten(l):
    return [x for b in l for x in b]

def get_block_spans(block):
    if 'lines' not in block:
        return []
    return flatten([line['spans'] for line in block['lines']])

def get_page_spans(blocks):
    block_spans = [get_block_spans(block) for block in blocks]
    return flatten(block_spans)

def normalize_block_text(block):
    spans = get_block_spans(block)
    if len(spans) == 0:
        return ''
    s = ' '.join([span['text'] for span in spans])
    return re_not_alpha.sub('', s).lower()

def list_count(li: list) -> dict:
    res = {}
    for l in li:
        if l in res:
            res[l] += 1
        else:
            res[l] = 1
    return res

def page_average_font_size(blocks: dict) -> float:
    total_size = 0
    span_count = 0
    for block in blocks:
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                total_size += span['size']
                span_count += 1
    if span_count == 0:
        return 0
    return total_size / span_count

def derotate_bbox(bbox, derotation_matrix):
    x0, y0, x1, y1 = bbox
    x0, y0 = list(fitz.Point(x0, y0) * derotation_matrix)
    x1, y1 = list(fitz.Point(x1, y1) * derotation_matrix)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def get_page_blocks(page):
    blocks = json.loads(page.get_text(option = 'json', sort = True))['blocks']

    for block in blocks:
        block['bbox'] = derotate_bbox(block['bbox'], page.rotation_matrix)
        if 'lines' in block:
            for line in block['lines']:
                line['bbox'] = derotate_bbox(line['bbox'], page.rotation_matrix)

                if 'spans' in line:
                    for span in line['spans']:
                        span['bbox'] = derotate_bbox(span['bbox'], page.rotation_matrix)
    return blocks

def page_to_box(page, blocks, drawings, tables: List[List[float]] = []) -> Box:

    average_font_size = page_average_font_size(blocks)
    text_boxes = []
    table_boxes = []
    for t in tables:
        table_boxes.append(Box(box_type = 'table', bbox = tuple(t)))

    excluded_block_ids = []
    # find true extent of tables
    for block_id, block in enumerate(blocks):
        if 'lines' not in block:
            continue
        
        block_spans = get_block_spans(block)
        block_y0 = min([s['bbox'][1] for s in block_spans])

        block_y1 = max([s['bbox'][3] for s in block_spans])

        new_box = Box(box_type = 'text', 
                        bbox = tuple(block['bbox']))
        
        for table_box_id, table_box in enumerate(table_boxes):

            if table_box.intersection_area_percentage(new_box) > 0.25:
                table_boxes[table_box_id] = table_box.merge_x(new_box)

    for block_id, block in enumerate(blocks):

        if 'lines' not in block:
            continue

        else:
            sub_boxes = []
            for line in block['lines']:
                # filter out text lines at an angle
                if False and page.rotation == 0 and abs(line['dir'][1]) > 0.1:
                    continue
                for span in line['spans']:
                    new_box = Box(box_type = 'text', 
                                bbox = tuple(span['bbox']), 
                                text = span['text'], 
                                font_name = span['font'], 
                                font_size = span['size'], 
                                font_color = span['color'], 
                                average_font_size = average_font_size)
                    add_to_sub_boxes = True
                    for table_box_id, table_box in enumerate(table_boxes):
                        
                        if (table_box_id, block_id) not in excluded_block_ids and table_box.intersection_area_percentage(new_box) > 0.25:
                            table_box.add_sub_box(new_box)
                            
                            add_to_sub_boxes = False
                            break
                    
                    if add_to_sub_boxes:
                        sub_boxes.append(new_box)

        if len(sub_boxes) > 0:
            text_boxes.append(Box(sub_boxes = sub_boxes))

    table_boxes = [box for box in table_boxes if len(box.boxes()) > 0]
    for table_box in table_boxes:
        table_box.trim_border()

    all_text_boxes = table_boxes + text_boxes

    drawing_boxes = []
    
    for d in drawings:
        drawing_boxes.append(Box(box_type = 'drawing', bbox = tuple(d['rect'])))
    image_boxes = []
    for block_id, block in enumerate(blocks):
        if block['type'] == 1:
            # image box
            image_box = Box(box_type = 'image',
                          bbox = tuple(block['bbox']))
            
            if image_box.size_x() < 0.1 * page.rect.width and image_box.size_y() < 0.1 * page.rect.height:
                # small picture, like an icon or something - ignore
                continue

            elif any([image_box.intersects(text_box) for text_box in all_text_boxes]):
                # treat as a drawing (later we take only the lower and upper sides) rather than an image
                new_box.set_box_type('drawing')
                drawing_boxes.append(new_box)
                continue

            else:
                # add to the images
                image_boxes.append(image_box)

    for table_box in table_boxes:
        dbs = [box for box in drawing_boxes if box.intersects(table_box)]
            
        table_box.format_table(dbs)

    # for drawings we take the lower and upper bounds to find the horizontal lines
    all_boxes = all_text_boxes + image_boxes + drawing_boxes
    res = Box(sub_boxes = all_boxes)
    
    return res

def get_footer_strings(pages: List, min_count: int = 5) -> Set[str]:
    # FIX - what about multiple footers?
    # FIX - headers?
    footer_candidates = []
    for page in pages:
        for block in reversed(page):
            if 'lines' not in block:
                continue
            candidate = normalize_block_text(block)
            if candidate != '':
                footer_candidates.append(candidate)
            break

    counts = list_count(footer_candidates)
    res = set()
    for k in counts.keys():
        # FIX: min with number of pages
        if k != '' and counts[k] >= max(min_count, int(np.ceil(len(pages) / 5))):
            res.add(k)
    return res

def box_to_pdf_data(box, page_number, section_title) -> dict:
    res = []
    for i, text_box in enumerate(box.text_boxes()):
        new_dict = {
            'id': 'p' + str(page_number) + 'b' + str(i),
            'page': page_number,
            'section': section_title,
            'text': text_box.text(),
            'text': text_box.text(),'type': text_box.type(),
            'bbox': [text_box.x0, text_box.y0, text_box.x1, text_box.y1]
        }

        table_dict = []
        if text_box.is_table:
            for row in text_box.boxes():
                for cell in row.boxes():
                    table_dict.append({'text': cell.text(), 
                                       'bbox': [cell.x0, cell.y0, cell.x1, cell.y1],
                                       'row_type': cell.Row_Type,
                                       'font_size': cell.Font_Size,
                                       'visible': cell.Visible
                                       })
        new_dict['table_data'] = table_dict
        res.append(new_dict)

    return res

double_blanks = re.compile('\s+')
def horizontal_coverage(row, x0, x1):
    all_chars = ''
    for box in row.Sub_Boxes:
        all_chars += box.text()
    string_length = len(all_chars)
    
    blank_length = sum([len(s) for s in double_blanks.findall(all_chars)])
    blank_adjustment = 1 - (blank_length / string_length)
    return blank_adjustment * sum([box.size_x() for box in row.Sub_Boxes if box.text() != '']) / (x1 - x0)

def row_type(row, x0, x1):
    boxes = row.Sub_Boxes

    # footnotes
    if len(boxes) == 2:
        median_size = float(max([cell.Font_Size for cell in boxes]))
        boxes = [cell for cell in boxes if cell.Font_Size > median_size - 1.5]
    elif len(boxes) > 2:
        median_size = float(np.median([cell.Font_Size for cell in boxes]))
        boxes = [cell for cell in boxes if cell.Font_Size > median_size - 1.5]
        
    if len(boxes) == 1:
        if row.x0 < x0 + (x1 - x0) * 0.2:
            return 'unknown'
        else:
            return 'header'
    else:
        data_types = [classify_data_type(s.text()) for s in boxes]
        n_valid_cells = sum([dc in ('numeric', 'year', 'string') for dc in data_types])
        n_numeric_cells = sum([dc == 'numeric' for dc in data_types])
        if n_valid_cells == 0:
            return 'unknown'
        elif n_numeric_cells / n_valid_cells >= 0.5:
            return 'data'
        else:
            return 'header'
    
    


def detect_tables(page, blocks, model):
    pix = page.get_pixmap(matrix = page.rotation_matrix)
    spans = get_page_spans(blocks)

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    width, height = image.size
    feature_extractor = DetrFeatureExtractor()
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)

    results = feature_extractor.post_process_object_detection(outputs, threshold=0.01, target_sizes=[(height, width)])[0]

    bboxes = results['boxes'].tolist()
    scores = results['scores'].tolist()

    bboxes = [bbox for score, bbox in sorted(zip(scores, bboxes), reverse = True)]

    boxes = [Box(box_type = 'table', bbox = bbox) for bbox in bboxes]

    span_boxes = [Box(box_type = 'text', 
                                bbox = tuple(span['bbox']), 
                                text = span['text'], 
                                font_name = span['font'], 
                                font_size = span['size'], 
                                font_color = span['color']) for span in spans]
    
    span_boxes = sorted(span_boxes, key = lambda box: box.y0)

    res = []
    table_candidates = []

    while len(boxes) > 0:
        next_box = boxes.pop(0)
        boxes = [box for box in boxes if next_box.intersection_area_percentage(box) <= 0.25]
        res.append(next_box)
        table_candidates.append(next_box)

    block_boxes = [Box(bbox = block['bbox']) for block in blocks]
    block_boxes = sorted(block_boxes, key = lambda box: box.y0)
    # take tables in vertical order
    res = sorted(res, key = lambda box: box.y0)
    table_candidates = sorted(table_candidates, key = lambda box: box.y0)

    # remove non-tables
    tables_to_remove = []
    for table_index, table in enumerate(table_candidates):
        for span_box in span_boxes:
            if span_box.intersection_area(table) > 0.25:
                table.add_sub_box(span_box)
        table.group_into_rows()
        data_types = []
        for row in table.boxes():
            if horizontal_coverage(row, table.x0, table.x1) > 0.75:
                data_types.append('text')
            else:
                data_types.append(row_type(row, table.x0, table.x1))

        if len(data_types) == 0 or len([d for d in data_types if d == 'data']) / len(data_types) < 0.1:
            tables_to_remove.append(table_index)
            
    res = [r for i, r in enumerate(res) if i not in tables_to_remove]

    # extending tables downward
    for box_index, box in enumerate(res):
        # make box from bottom of table to end of page
        extended_box = Box(box_type = 'table', bbox = (box.x0, box.y1, box.x1, page.rect.height))

        # cut box with top of any boxes lower
        for lower_box in res[(box_index + 1):]:
            if lower_box.y0 <= extended_box.y0:
                continue
            elif lower_box.y0 > extended_box.y1:
                break
            elif extended_box.intersection_area_percentage(lower_box) > 0.25:
                extended_box = Box(box_type = 'table', bbox = (extended_box.x0, extended_box.y0, extended_box.x1, lower_box.y0))
                break
        
        # cut box with any blocks that overlap and extend a certain amount horizontally outside the box
        
        for block_box in block_boxes:
            if block_box.y0 <= extended_box.y0:
                continue
            elif block_box.y0 > extended_box.y1:
                break
            elif (extended_box.intersection_x(block_box) > 0.25 * extended_box.size_x()) and (block_box.intersection_x(extended_box) < 0.75 * block_box.size_x()):
                extended_box = Box(box_type = 'table', bbox = (extended_box.x0, extended_box.y0, extended_box.x1, block_box.y0))
                break

        # group extended_box into rows
        for span_box in span_boxes:
            if span_box.intersection_area(extended_box) > 0.25:
                extended_box.add_sub_box(span_box)

        extended_box.group_into_rows()

        y = extended_box.y0
        last_index = 0
        for i, row in enumerate(extended_box.Sub_Boxes):

            # cut box with any vertical gaps greater than x% of page
            if row.y0 - y > 0.05 * page.rect.height or horizontal_coverage(row, extended_box.x0, extended_box.x1) > 0.75:

                break
            else:
                y = row.y1
                last_index = i + 1
        
        extended_box.Sub_Boxes = extended_box.Sub_Boxes[:last_index]
        extended_box.y1 = y

        # from bottom of box, find first line that is sparse numeric row
        # cut box at bottom of that line
        #for row in extended_box.Sub_Boxes:
        #    print([box.text() for box in row.boxes()])
        #    print(row_type(row, extended_box.x0, extended_box.x1))
        row_types = [row_type(row, extended_box.x0, extended_box.x1) for row in extended_box.Sub_Boxes]
        if all([row_type != 'data' for row_type in row_types]):
            continue
            1
        else:
            index = max([row_index for row_index, row_type in enumerate(row_types) if row_type == 'data'])
            extended_box.y1 = extended_box.Sub_Boxes[index].y1
            extended_box.Sub_Boxes = extended_box.Sub_Boxes[:(index + 1)]

        # check blocks 16, 25, 27
        for block in blocks:
            if 'lines' not in block:
                continue
            if not extended_box.intersects(Box(bbox = block['bbox'])):
                continue
            block_spans = get_block_spans(block)
            span_intersects = [extended_box.intersection_area(Box(bbox = span['bbox'])) > 0.1 for span in block_spans]
            if any(span_intersects) and not all(span_intersects):
                extended_box.y1 = min(extended_box.y1, block['bbox'][1])
                extended_box.Sub_Boxes = [box for box in extended_box.Sub_Boxes if box.y1 <= extended_box.y1]
                

        # merge box with table
        box.y1 = extended_box.y1
        box.Sub_Boxes += extended_box.Sub_Boxes
        res[box_index] = box

    # extending tables upward
    for box_index, box in enumerate(res):
        # make box from bottom of table to end of page
        extended_box = Box(box_type = 'table', bbox = (box.x0, 0, box.x1, box.y0))

        # cut box with top of any boxes higher
        for higher_box in res[:box_index]:
            if higher_box.y1 > extended_box.y1:
                continue
            elif higher_box.y0 > extended_box.y1:
                break
            elif extended_box.intersection_area_percentage(higher_box) > 0.25:
                extended_box = Box(box_type = 'table', bbox = (extended_box.x0, higher_box.y1, extended_box.x1, extended_box.y1))
                break
        
        # cut box with any blocks that overlap and extend a certain amount horizontally outside the box
        for block_box in block_boxes:
            if block_box.y1 > extended_box.y1:
                continue
            elif block_box.y0 > extended_box.y1:
                break
            elif (extended_box.intersection_x(block_box) > 0.25 * extended_box.size_x()) and (block_box.intersection_x(extended_box) < 0.75 * block_box.size_x()):
                extended_box = Box(box_type = 'table', bbox = (extended_box.x0, block_box.y1, extended_box.x1, extended_box.y1))
                break
        
        # group extended_box into rows
        for span_box in span_boxes:
            if span_box.intersection_area(extended_box) > 0.25:
                extended_box.add_sub_box(span_box)

        extended_box.group_into_rows()

        y = extended_box.y1
        last_index = len(extended_box.Sub_Boxes)
        for i, row in reversed(list(enumerate(extended_box.Sub_Boxes))):
            #if 'using the proportional amortization method such that the' in row.boxes()[0].text():
            #    1
            #    print(horizontal_coverage(row, extended_box.x0, extended_box.x1))
            #    print(row_type(row, extended_box.x0, extended_box.x1))
            # cut box with any vertical gaps greater than x% of page
            if row.y1 - y > 0.05 * page.rect.height or horizontal_coverage(row, extended_box.x0, extended_box.x1) > 0.75:

                break
            else:
                y = row.y0
                last_index = i
        
        extended_box.Sub_Boxes = extended_box.Sub_Boxes[last_index:]
        extended_box.y1 = y

        # from bottom of box, find first line that is sparse numeric row
        # cut box at bottom of that line
        #for row in extended_box.Sub_Boxes:
        #    print([box.text() for box in row.boxes()])
        #    print(row_type(row, extended_box.x0, extended_box.x1))
        row_types = [row_type(row, extended_box.x0, extended_box.x1) for row in extended_box.Sub_Boxes]
        if all([row_type == 'unknown' for row_type in row_types]):
            
            continue
            1
        else:
            index = min([row_index for row_index, row_type in enumerate(row_types) if row_type != 'unknown'])
            extended_box.y0 = extended_box.Sub_Boxes[index].y0
            extended_box.Sub_Boxes = extended_box.Sub_Boxes[index:]

        # check blocks 16, 25, 27
        for block in blocks:
            if 'lines' not in block:
                continue
            if not extended_box.intersects(Box(bbox = block['bbox'])):
                continue
            block_spans = get_block_spans(block)
            span_intersects = [extended_box.intersection_area(Box(bbox = span['bbox'])) > 0.1 for span in block_spans]
            if any(span_intersects) and not all(span_intersects):
                extended_box.y0 = max(extended_box.y0, block['bbox'][3])
                extended_box.Sub_Boxes = [box for box in extended_box.Sub_Boxes if box.y0 >= extended_box.y0]

        # merge box with table
        box.y0 = extended_box.y0
        box.Sub_Boxes = extended_box.Sub_Boxes + box.Sub_Boxes
        res[box_index] = box

    res = sorted(res, key = lambda box: box.y0)
    # merge tables
    for table_index, table in enumerate(res):
        if table_index == 0:
            continue
        merge = False
        table.Sub_Boxes = [box for row in table.boxes() for box in row.boxes()]
        table.group_into_rows()
        # only merge if there is no header
        for row in table.boxes():
            
            rt = row_type(row, table.x0, table.x1)

            if rt == 'header':
                merge = False
                break
            elif rt == 'data':
                merge = True
                break
        if merge:
            # try to find table to match
            for higher_table_index, higher_table in enumerate(res[:table_index]):
                # if horizontally aligned
                
                if higher_table.intersection_x(table) > 0.8 * min(higher_table.size_x(), table.size_x()):
                    # check overlapping
                    if higher_table.y1 >= table.y0:
                        new_table = Box(bbox = (min(higher_table.x0, table.x0), min(higher_table.y0, table.y0), 
                                                max(higher_table.x1, table.x1), max(higher_table.y1, table.y1)))
                        res[table_index] = new_table
                        res[higher_table_index] = new_table
                    else:
                        # check nothing in the middle
                        merge_tables = True
                        gap_box = Box(bbox = (min(table.x0, higher_table.x0), higher_table.y1, max(table.x1, higher_table.x1), table.y0))
                        for span_box in span_boxes:
                            if span_box.intersection_area(gap_box) > 0.25:
                                if page.number == 21:
                                    print('something in the way')
                                merge_tables = False
                                break

                        if merge_tables:
                            if page.number == 21:
                                print('doing the merge')
                            new_table = Box(bbox = (min(higher_table.x0, table.x0), min(higher_table.y0, table.y0), 
                                                max(higher_table.x1, table.x1), max(higher_table.y1, table.y1)))
                            res[table_index] = new_table
                            res[higher_table_index] = new_table

    for table in res:
        for block in blocks:
            spans = get_block_spans(block)
            if len(spans) == 0:
                continue
            block_box = Box(bbox = (min([span['bbox'][0] for span in spans]),
                                    min([span['bbox'][1] for span in spans]),
                                    max([span['bbox'][2] for span in spans]),
                                    max([span['bbox'][3] for span in spans])))
                            
            if block_box.intersection_area(table) > 0 and block_box.intersects_y(table):
                if block_box.y0 < table.y0 - 1:
                    table.y0 = block_box.y1
                if block_box.y1 > table.y1 + 1:
                    table.y1 = block_box.y0

    return [b.bbox() for b in res if b.area() > 0]

class PDF:
    @staticmethod
    def from_fitz(document, tables: Optional[List[List]] = None, threshold = 0.7):
        page_blocks = [get_page_blocks(page) for page in document]
        page_drawings = [page.get_drawings() for page in document]
       
        # remove footers
        footer_strings = get_footer_strings(page_blocks)
        for page_number, page_block in enumerate(page_blocks):
            page_blocks[page_number] = [block for block in page_block if normalize_block_text(block) not in footer_strings]

        if tables == None:
            model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

            tables = [detect_tables(page, blocks, model) for page, blocks in zip(document, page_blocks)]

        boxes = []
        for page, blocks, drawings, table in zip(document, page_blocks, page_drawings, tables):
            # add a small border to the table in case it misses a heading
            #adjusted_table = [[t[0], t[1] - table_border, t[2], t[3]] for t in table]
            boxes.append(page_to_box(page, blocks, drawings, table))
        contents = contents_list(document, boxes)

        
        data = []
        for page_number, (box, section_title) in enumerate(zip(boxes, contents)):
            box.sort_reading_order()
            data += box_to_pdf_data(box, page_number, section_title)

        res = {'data': data, 'tables': tables}

        return PDF(res, vector_store = None)
    
    @staticmethod
    def from_file(file_path):
        with open(file_path, 'r', encoding = 'utf-8') as f:
            JSON = json.load(f)
        vector_store_path = file_path + '-vectorstore.pickle'
        if os.path.exists(vector_store_path):
            vs = VectorStore.load(vector_store_path)
        else:
            vs = None
        return PDF(JSON, vector_store = vs)
    
    def to_file(self, file_path):
        with open(file_path, 'w', encoding = 'utf-8') as f:
            json.dump(self.Data, f)
        if self.VectorStore != None:
            vector_store_path = file_path + '-vectorstore.pickle'
            self.VectorStore.save(vector_store_path)

    def __init__(self, data, vector_store = None):
        self.Data = data
        self.VectorStore = vector_store
    
    def generate_vector_store(self, embedding_function):
        vs = VectorStore(embedding_function)
        
        texts = [block['text'] for block in self.blocks()]
        ids = [block['id'] for block in self.blocks()]
        
        vs.add(texts, ids)
                
        self.VectorStore = vs

    def blocks(self):
        return self.Data['data']
    
    def text(self):
        return [x['text'] for x in self.blocks()]

    def filter_block_ids(self, predicate) -> PDF:
        res = PDF({
            'data': [b for b in self.blocks() if predicate(b)],
            'tables': self.Data['tables']
        })
        if res.VectorStore != None:
            block_ids = [b['id'] for b in res.blocks()]
            res.VectorStore = self.VectorStore.filter_vectors(block_ids)
        return res


    def pages(self, page_numbers):
        return self.filter_block_ids(lambda b: b['page'] in page_numbers)

    def page(self, page_number):
        return self.pages([page_number])

    def titles(self):
        return self.filter_block_ids(lambda b: b['type'] == 'title')
    
    def tables(self):
        return self.filter_block_ids(lambda b: b['type'] == 'table')
    
    def paragraphs(self):
        return self.filter_block_ids(lambda b: b['type'] == 'text')
    
    def footnotes(self):
        return self.filter_block_ids(lambda b: b['type'] == 'footnote')

    def non_tables(self):
        return self.filter_block_ids(lambda b: b['type'] != 'table')

    def ids(self, ids):
        return self.filter_block_ids(lambda b: b['id'] in ids)
    
    def id(self, id):
        return self.ids([id])

    def sections(self, section_headings):
        return self.filter_block_ids(lambda b: b['section'] in section_headings)
    
    def section(self, section_heading):
        return self.sections([section_heading])
    
    def search(self, query, case_sensitive = False):
        matcher = make_simple_search_function(query, case_sensitive)
        return self.filter_block_ids(lambda b: matcher(b['text']))
    
    def boolean_search(self, query, case_sensitive = False):
        matcher = make_boolean_search_function(query, case_sensitive)
        return self.filter_block_ids(lambda b: matcher(b['text']))
    
    def regex_search(self, query, case_sensitive = False):
        matcher = make_regex_search_function(query, case_sensitive)
        return self.filter_block_ids(lambda b: matcher(b['text']))
    
    def vector_search(self, query, n_results = 5):
        vector_store = self.VectorStore
        
        ids = vector_store.search(query, n_results)
        
        return self.ids(ids)

def openai_embedding(text, model="text-embedding-ada-002"):
    
    text = text.replace("\n", " ")
    if text == '':
        return [0.0] * 1536
    
    if len(text) > 8191:
        print('Truncating long text: ' + text[:200] + '...')
        text = text[:8191]

    repeats = 0
    while repeats < 60:
        try:
            res = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
            break
        except Exception as e:
            if type(e) == openai.error.InvalidRequestError:
                print('Skipping invalid text: ' + text)
                return [0.0] * 1536
            print(text)
            print(e)
            repeats += 1
            sleep(1)
    return res

def format_prompt(prompt, **kwargs):
    return([{'role': d['role'], 'content': d['content'].format(**kwargs)} for d in prompt])

@dataclass
class LLM:
    prompt:str
    model:str = 'gpt-3.5-turbo'
    max_tokens:int = 256
    temperature:float = 0

    def run(self, **kwargs):
        prompt = format_prompt(self.prompt, **kwargs)
        api_200 = False
        tries = 0
        while (not api_200) and (tries < 60):
            try:
                completion = openai.ChatCompletion.create(
                    model = self.model,
                    messages = prompt,
                    max_tokens = self.max_tokens,
                    temperature = self.temperature
                )
                api_200 = True
            except Exception as error:
                print(repr(error))
                tries += 1
                sleep(1)
                
        res = completion.choices[0].message.content
        
        return res