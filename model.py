import torch
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
from ocr import OCR
from preprocessing import OCRPreprocessor


class LayoutLM_QA_Model:
    def __init__(self, preprocessor: OCRPreprocessor, ocr: OCR, image_path: str):
        self.preprocessor = preprocessor
        self.ocr = ocr
        self.image_path = image_path
        self.preproc_img_arr = preprocessor.process_image_for_ocr(image_path)
        self.words, self.bboxes = ocr.perform_ocr(self.preproc_img_arr)
        assert len(self.words) == len(self.bboxes)

        self.tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
        self.model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")

    def updateImage(self, image_path):
        self.image_path = image_path
        self.preproc_img_arr = self.preprocessor.process_image_for_ocr(image_path)

    def answer_question(self, question):

        encoding = self.tokenizer(
            question.split(), self.words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
        )

        bbox = []
        for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
            if s == 1:
                bbox.append(self.bboxes[w])
            elif i == self.tokenizer.sep_token_id:
                bbox.append([1000] * 4)
            else:
                bbox.append([0] * 4)
        encoding["bbox"] = torch.tensor([bbox])

        word_ids = encoding.word_ids(0)
        outputs = self.model(**encoding)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
        answer = " ".join(self.words[start : end + 1])

        return answer
    

    