#!/usr/bin/env python3
"""Check comparison joins and reject incompatible historical evaluation settings."""
import copy
import json
import unittest

import compare_results as comparison


class ComparisonTests(unittest.TestCase):
    def test_extended_checkpoints_are_included_in_comparison(self):
        for mode in comparison.SOURCES:
            self.assertEqual(comparison.configured_checkpoints(mode), ['20k', '40k', '60k', '80k', '100k'])

    def test_comparison_and_protocol_guards(self):
        for mode, source in comparison.SOURCES.items():
            full = comparison.keyed(json.loads((comparison.REPO / 'experiments/8' / source /
                                                'results/restart_fid_summary.json').read_text()))
            subset = {key: copy.deepcopy(row) for key, row in full.items()
                      if key[0] in ('20k', '40k', '60k')}
            for key, row in subset.items():
                row['fid'] = float(row['fid']) + (0.5 if key[2] == 0 else 0.25)
            table = comparison.build_comparison(subset, full)
            self.assertEqual(len(table), 81)
            for row in table:
                self.assertAlmostEqual(row['subset_improvement'] - row['full_improvement'],
                                       0 if row['restart_k'] == 0 else 0.25)
            first = next(iter(subset))
            for field, value in [('num_images', 1), ('main_steps', 999), ('seeds', '1-5000')]:
                changed = copy.deepcopy(subset)
                changed[first][field] = value
                with self.assertRaises(ValueError):
                    comparison.build_comparison(changed, full)
            changed = copy.deepcopy(subset)
            changed[first]['protocol']['fid_ref']['sha256'] = 'a'*64
            with self.assertRaises(ValueError):
                comparison.build_comparison(changed, full)
            with self.assertRaises(ValueError):
                comparison.keyed([full[first], full[first]])


if __name__ == '__main__':
    unittest.main()
