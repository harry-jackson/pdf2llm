from pdf2llm.layout import Box
import re
from typing import List

def contents_page(doc, pages_to_check: int = 10, min_links: int = 5) -> int:
    pages_to_check = min(pages_to_check, len(doc))
    for i in range(pages_to_check):
        page = doc[i]
        links = page.get_links()
        if len(links) >= min_links:
            return i
    return -1

re_page_from_link_name = re.compile('(?<=page=)[0-9]+|$')
def page_from_link_name(s: str) -> int:
    res = re_page_from_link_name.findall(s)[0]
    if res == '':
        return -1
    else:
        return int(res)
    
def process_link(l: dict) -> dict:
    if l['kind'] == 1:
        res = {'from': list(l['from']), 'page': l['page']}
    elif l['kind'] == 4:
        page = page_from_link_name(l['name'])
        if page == -1:
            return {}
        else:
            res = {'from': list(l['from']), 'page': page}
    else:
        return {}
    
    return res

def process_links(page, all_boxes) -> dict:
    links = page.get_links()
    links = [process_link(l) for l in links]
    links = [l for l in links if len(l) > 0]

    link_boxes = [Box(bbox = tuple(link['from'])) for link in links]
    
    for box in all_boxes:
        
        intersections = [(i, box.intersection_area(link_box)) for i, link_box in enumerate(link_boxes)]
        intersections = [(i, s) for i, s in intersections if s > 0]

        if len(intersections) == 0:
            continue

        link_id = sorted(intersections, key = lambda x: x[1], reverse = True)[0][0]
        if 'text' in links[link_id]:
            links[link_id]['text'] += box.text()
        else:
            links[link_id]['text'] = box.text()

    for i, link in enumerate(links):
        if 'text' not in link:
            links[i]['text'] = str(link['page'])
            
    res = {}
    for x in links:
        if x['page'] in res:
            res[x['page']] += ' ' + x['text']
        else:
            res[x['page']] = x['text']

    res = {value: key - 1 for key, value in zip(res.keys(), res.values())}
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
    return res
    
def contents_list(doc, boxes = List[Box]) -> List[str]:
    p = contents_page(doc)
    if p < 0:
        return [''] * len(doc)
    page = doc[p]
    page_boxes = boxes[p].leaves()
    links = process_links(page, page_boxes)
    if links == {}:
        return [''] * len(doc)
    N = len(doc)

    if list(links.values())[0] == 0:
        prev_key = list(links.keys())[0]
    else:
        prev_key = ''

    # Generate the list
    res = []
    prev_value = 0
    for key, value in list(links.items())[1:]:
        res.extend([prev_key] * (value - prev_value))
        prev_value = value
        prev_key = key

    res.extend([key] * (N - len(res)))
    assert len(res) == len(doc)
    return res