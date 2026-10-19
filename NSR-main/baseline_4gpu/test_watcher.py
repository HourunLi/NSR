import unittest

from .watch_and_launch import parse_gpu_rows


class WatcherTests(unittest.TestCase):
    def test_parse_gpu_rows(self):
        rows = parse_gpu_rows(
            "0, GPU-a, NVIDIA H20, 4, 0\n3, GPU-b, NVIDIA H20, 511, 5\n"
        )
        self.assertEqual([row["index"] for row in rows], [0, 3])
        self.assertEqual(rows[1]["memory"], 511)
        self.assertEqual(rows[1]["util"], 5)

    def test_malformed_row_rejected(self):
        with self.assertRaises(ValueError):
            parse_gpu_rows("0, GPU-a, 4")


if __name__ == "__main__":
    unittest.main()
