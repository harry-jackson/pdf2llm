from __future__ import annotations
from pdf2llm.layout import Box
from pdf2llm.contents import contents_list
from pdf2llm.search import VectorStore, make_simple_search_function, make_boolean_search_function, make_regex_search_function
import fitz
import json
import numpy as np
import layoutparser as lp
from typing import List, Optional, Set
import os
import re
import openai
from dataclasses import dataclass
from time import sleep

re_not_alpha = re.compile('[^a-zA-Z]')

def normalize_text(s):
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

def page_to_box(page, tables: List[List[float]] = []) -> Box:
    
    blocks = json.loads(page.get_text(option = 'json', sort = True))['blocks']

    average_font_size = page_average_font_size(blocks)
    text_boxes = []
    table_boxes = []
    for t  in tables:
        table_boxes.append(Box(box_type = 'table', bbox = tuple(t)))

    for block_id, block in enumerate(blocks):

        if 'lines' not in block:
            continue

        else:
            sub_boxes = []
            for line in block['lines']:
                # filter out text lines at an angle
                if abs(line['dir'][1]) > 0.1:
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
                    for table_box in table_boxes:
                        
                        if table_box.intersects(new_box):
                            table_box.add_sub_box(new_box)
                            
                            add_to_sub_boxes = False
                            break
                    
                    if add_to_sub_boxes:
                        sub_boxes.append(new_box)

        if len(sub_boxes) > 0:
            text_boxes.append(Box(sub_boxes = sub_boxes))

    all_text_boxes = table_boxes + text_boxes

    drawing_boxes = []
    
    for d in page.get_drawings():
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

def get_footer_strings(boxes: List[Box], min_count: int = 5) -> Set[str]:
    # FIX - what about multiple footers?
    # FIX - headers?
    footer_candidates = [normalize_text(box.text_boxes()[-1].text()) for box in boxes if len(box.text_boxes()) > 0]
    
    counts = list_count(footer_candidates)
    res = set()
    for k in counts.keys():
        # FIX: min with number of pages
        if k != '' and counts[k] >= min_count:
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

def detect_tables(page, model):
    pix = page.get_pixmap()
    pix = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
    layout = model.detect(pix)

    res = [[b.block.x_1, b.block.y_1, b.block.x_2, b.block.y_2] for b in layout._blocks if b.type == 'Table']

    return res

class PDF:
    @staticmethod
    def from_fitz(document, tables: Optional[List[List]] = None, model = None):
        table_border = 5
        if tables == None:
            if model == None:
                model = lp.Detectron2LayoutModel('lp/config_large.yml',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
            tables = [detect_tables(page, model) for page in document]

        boxes = []
        for page_num, page, table in zip(range(len(document)), document, tables):
            # add a small border to the table in case it misses a heading
            adjusted_table = [[t[0], t[1] - table_border, t[2], t[3]] for t in table]
            boxes.append(page_to_box(page, adjusted_table))
        contents = contents_list(document, boxes)

        footer_strings = get_footer_strings(boxes)
        data = []
        for page_number, (box, section_title) in enumerate(zip(boxes, contents)):
            box = Box(sub_boxes = [sub_box for sub_box in box.boxes() if (not box.is_text_box()) or (normalize_text(sub_box.text()) not in footer_strings)])
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