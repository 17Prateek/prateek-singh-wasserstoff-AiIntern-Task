import unittest
from models.summarization_model import summarize_attributes

class TestSummarization(unittest.TestCase):
    def test_summarize_attributes(self):
        # Test a dummy text input
        text = "This is a long description of an object that needs to be summarized."
        summary = summarize_attributes(text)
        
        # Check that the summary is a string
        self.assertIsInstance(summary, str)
        
        # Check that the summary is not empty
        self.assertGreater(len(summary), 0)

if __name__ == "__main__":
    unittest.main()
