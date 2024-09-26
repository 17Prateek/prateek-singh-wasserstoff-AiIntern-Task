import unittest
from models.segmentation_model import segment_image
from PIL import Image
import numpy as np

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        # Create a sample image for testing
        self.image = Image.new('RGB', (100, 100), color='red')

    def test_segment_image(self):
        # Run the segmentation model
        output = segment_image(self.image)
        
        # Check that the output is not None
        self.assertIsNotNone(output)
        
        # Check that the output contains masks
        self.assertIn('masks', output[0])

if __name__ == "__main__":
    unittest.main()
