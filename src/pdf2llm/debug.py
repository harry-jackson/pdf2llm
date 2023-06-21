import fitz

def markup_pdf(doc, pdf):
    for page_number, page in enumerate(doc):
        blocks = pdf.page(page_number).blocks()

        page.clean_contents(sanitize = False)

        for i, block in enumerate(blocks):
            if block['type'] == 'table':
                j = 1
                page.draw_rect(rect = fitz.Rect(block['bbox']), color = (1, 0, 0))
                for cell in block['table_data']:
                    if not cell['visible']:
                        continue
                    if cell['row_type'] == 'header':
                        cell_color = (0, 0, 1)
                    elif cell['row_type'] == 'data':
                        cell_color = (0, 1, 0)
                    else:
                        raise Exception('invalid row type')
                    page.draw_rect(rect = fitz.Rect(cell['bbox']), color = cell_color)
                    page.insert_text(point = (cell['bbox'][0], cell['bbox'][1]), text = str(j), color = (1, 0, 0))
                    j += 1
            else:
                page.draw_rect(rect = fitz.Rect(block['bbox']), color = (1, 0, 0))
            page.insert_text(point = (block['bbox'][0], block['bbox'][1]), text = str(i), color = (0, 0, 1))

    return doc