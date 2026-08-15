import unittest

import numpy as np

from umi.common.timestamp_util import match_nearest_timestamps


class TestTimestampUtil(unittest.TestCase):
    def test_maps_60_hz_trajectory_to_even_120_hz_frames(self):
        video_timestamps = np.arange(120, dtype=np.float64) / 120.0
        trajectory_timestamps = np.arange(60, dtype=np.float64) / 60.0

        indices = match_nearest_timestamps(
            reference_timestamps=video_timestamps,
            query_timestamps=trajectory_timestamps,
            max_delta=(1.0 / 120.0) * 0.51,
            name='test')

        np.testing.assert_array_equal(indices, np.arange(0, 120, 2))

    def test_rejects_duplicate_reference_matches(self):
        video_timestamps = np.arange(4, dtype=np.float64) / 60.0
        trajectory_timestamps = np.array([0.0, 0.001])

        with self.assertRaisesRegex(ValueError, 'duplicate'):
            match_nearest_timestamps(
                reference_timestamps=video_timestamps,
                query_timestamps=trajectory_timestamps,
                max_delta=1.0,
                name='test')


if __name__ == '__main__':
    unittest.main()
