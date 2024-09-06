import unittest
from src.model_training import build_cnn

class TestModelTraining(unittest.TestCase):

    def test_build_cnn(self):
        input_shape = (100, 1)
        num_classes = 10
        model = build_cnn(input_shape, num_classes)
        self.assertEqual(len(model.layers), 7, "CNN model must have 7 layers.")
        self.assertEqual(model.output_shape[-1], num_classes, "CNN model output should match the number of classes.")

if __name__ == '__main__':
    unittest.main()
