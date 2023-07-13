import fitz
from pdf2llm import derotate_bbox

def markup_pdf(doc, pdf):
    for page_number, page in enumerate(doc):
        blocks = pdf.page(page_number).blocks()
        
        page.clean_contents(sanitize = False)
        
        M = page.derotation_matrix
        for i, block in enumerate(blocks):
            block_box = derotate_bbox(block['bbox'], M)
            if block['type'] == 'table':
                j = 1
                
                page.draw_rect(rect = fitz.Rect(block_box), color = (1, 0, 0))
                for cell in block['table_data']:
                    if not cell['visible']:
                        continue
                    if cell['row_type'] == 'header':
                        cell_color = (0, 0, 1)
                    elif cell['row_type'] == 'data':
                        cell_color = (0, 1, 0)
                    else:
                        raise Exception('invalid row type')
                    cell_box = derotate_bbox(cell['bbox'], M)
                    page.draw_rect(rect = fitz.Rect(cell_box), color = cell_color)
                    page.insert_text(point = (cell_box[0], cell_box[1]), text = str(j), color = (1, 0, 0))
                    j += 1
            else:
                page.draw_rect(rect = fitz.Rect(block_box), color = (1, 0, 0))
            page.insert_text(point = (block_box[0], block_box[1]), text = str(i), color = (0, 0, 1))

    return doc