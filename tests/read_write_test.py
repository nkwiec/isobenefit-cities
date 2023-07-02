from unittest import TestCase

import numpy as np

from src.land_map import Land
from src.simulation import update_map_snapshot, save_snapshot


class TestReadWrite1(TestCase):
    def test_read_write(self):
        land = Land(10, 10)
        test_image_path = 'fixtures/test_land_map.png'
        land.set_configuration_from_image("C:/Users/nkwie\Projects/testingOFqgisplug/small pixel grid read_write.png")
        canvas = np.ones(shape=(20, 10, 4))
        update_map_snapshot(land, canvas)
        save_snapshot(canvas, output_path='results', step=0)
        self.assertEqual(1, 1)

