import unittest
from text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
        
    def test_clean_text(self):
        text = "Hello World! This is a test... 123 http://example.com"
        cleaned = self.processor.clean_text(text)
        self.assertEqual(cleaned, "hello world this is a test")
        
    def test_tokenize_and_lemmatize(self):
        text = "running dogs are running quickly"
        tokens = self.processor.tokenize_and_lemmatize(text)
        self.assertIn("run", tokens)  # "running" should be lemmatized to "run"
        self.assertIn("dog", tokens)  # "dogs" should be lemmatized to "dog"
        self.assertNotIn("are", tokens)  # "are" should be removed as stopword
        
    def test_process_text(self):
        text = "The quick brown foxes are jumping over the lazy dogs! http://test.com"
        result = self.processor.process_text(text)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['original_text'], text)
        self.assertTrue(len(result['tokens']) > 0)
        self.assertIn('quick', result['tokens'])
        self.assertIn('brown', result['tokens'])
        self.assertIn('fox', result['tokens'])
        self.assertNotIn('the', [t.lower() for t in result['tokens']])
        
    def test_handle_non_string(self):
        result = self.processor.process_text(None)
        self.assertFalse(result['success'])
        self.assertEqual(result['cleaned_text'], '')
        self.assertEqual(result['error'], 'Input must be a string')
        
if __name__ == '__main__':
    unittest.main() 