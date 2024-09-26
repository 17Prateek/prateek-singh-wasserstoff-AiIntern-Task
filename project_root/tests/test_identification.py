import unittest
from models.identification_model import identify_objects
from PIL import Image

class TestIdentification(unittest.TestCase):
    def setUp(self):
        # Create a sample image for testing
        self.image = Image.new('RGB', (100, 100), color='green')

    def test_identify_objects(self):
        # Run the object identification model
        output = identify_objects(self.image)
        
        # Check that the output is not None
        self.assertIsNotNone(output)
        
        # Check that there are object labels in the output
        self.assertIn('labels', output[0])

if __name__ == "__main__":
    unittest.main()
