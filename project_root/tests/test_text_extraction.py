import unittest
from models.text_extraction_model import extract_text
from PIL import Image

class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        # Create a sample image for testing
        self.image = Image.new('RGB', (100, 100), color='blue')

    def test_extract_text(self):
        # Run the text extraction model
        text_data = extract_text(self.image)
        
        # Check that the extracted text is a list (even if it's empty)
        self.assertIsInstance(text_data, list)

if __name__ == "__main__":
    unittest.main()
