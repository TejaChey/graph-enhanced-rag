import re

from langchain_core.documents import Document


class TextCleaner:
    def remove_extra_whitespace(self, text):
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()

    def remove_special_characters(self, text, keep_chars=""):
        default_keep = r'.,;:!?()\[\]{}\-_\'\"/@#%&*+=<>|\\~`^'
        pattern = f'[^a-zA-Z0-9\\s{re.escape(default_keep)}{re.escape(keep_chars)}]'
        return re.sub(pattern, '', text)

    def clean_document(self, document):
        cleaned_content = self.remove_extra_whitespace(document.page_content)
        cleaned_content = self.remove_special_characters(cleaned_content)
        return Document(page_content=cleaned_content, metadata=document.metadata)

    def clean_documents(self, documents):
        return [self.clean_document(doc) for doc in documents]


if __name__ == "__main__":
    cleaner = TextCleaner()
    test_text = "This   is    a  test   text  with   extra    spaces."
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaner.remove_extra_whitespace(test_text)}")
