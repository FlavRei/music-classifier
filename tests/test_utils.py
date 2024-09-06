import unittest
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.utils import save_model, load_model

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.model = Sequential([
            Dense(10, input_shape=(5,), activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def test_save_model(self):
        filename = 'test_model.h5'
        save_model(self.model, filename)
        self.assertTrue(os.path.exists(filename), "The template file was not created.")

        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
