import os
import fitz
import io
from PIL import Image

class OCR:
    def __init__(self, tessdata_path):
        os.environ["TESSDATA_PREFIX"] = tessdata_path

    def _normalize_bbox(bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
    ]

    def _load_image_array_to_pymupdf(image_array):
        # Convert the 2D array to a PIL image
        pil_image = Image.fromarray(image_array)

        # Convert the PIL image to bytes
        with io.BytesIO() as byte_io:
            pil_image.save(byte_io, format='PNG')
            byte_io.seek(0)
            data = byte_io.read()


        # Create a pixmap from the bytes data
        pixmap = fitz.Pixmap(data)

        # Create a new document
        doc = fitz.open()

        # Add the pixmap as a new page to the document
        page = doc.new_page(width=pixmap.width, height=pixmap.height)
        rect = fitz.Rect(0, 0, pixmap.width, pixmap.height)
        page.insert_image(rect, pixmap=pixmap)

        return page

    def perform_ocr(self, image_array):
        page = OCR._load_image_array_to_pymupdf(image_array)
        width = page.rect.width
        height = page.rect.height

        full_tp = page.get_textpage_ocr(flags=0, dpi=300, full=True)

        words = []
        bboxes = []

        for block in page.get_text("dict", textpage=full_tp)["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    words.append(span['text'])
                    bbox = OCR._normalize_bbox(span['bbox'], width, height)
                    bboxes.append(bbox)

        return words, bboxes