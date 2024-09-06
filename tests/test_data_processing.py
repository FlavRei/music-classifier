import unittest
from src.data_processing import load_audio_files

class TestDataProcessing(unittest.TestCase):

    def test_load_audio_files(self):
        data_path = './data/raw/'
        genres = ['classical', 'jazz'] 

        X, y = load_audio_files(data_path, genres)

        self.assertEqual(len(X), len(y), "The number of samples and labels must be the same.")
        self.assertGreater(len(X), 0, "Loading files should return non-empty data.")

if __name__ == '__main__':
    unittest.main()
