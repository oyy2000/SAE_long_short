import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from length_budget_distill.legacy_replication import gate_decision
from length_budget_distill.legacy_sae_continuation import require_replication
from length_budget_distill.factorial import file_sha256


class ReplicationGateTests(unittest.TestCase):
    settings = {'historical_short_accuracy': 0.7042, 'tolerance': 0.01}

    def test_historical_reproduction_passes(self):
        self.assertEqual(gate_decision(.7042, .6937, .6732, .6714, self.settings)['status'], 'passed')

    def test_current_failed_pilot_does_not_unlock_sae(self):
        self.assertEqual(gate_decision(.6478, .65, .6530, .6714, self.settings)['status'], 'failed')

    def test_high_accuracy_alone_is_not_registered_gate(self):
        self.assertEqual(gate_decision(.705, .70, .71, .6714, self.settings)['status'], 'failed')

    def test_tolerance_boundary(self):
        self.assertEqual(gate_decision(.6942, .69, .68, .6714, self.settings)['status'], 'passed')

    def test_missing_replication_blocks_sae(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('length_budget_distill.legacy_sae_continuation.source_hashes'), patch('length_budget_distill.legacy_sae_continuation.root_for', return_value=Path(temp)):
                with self.assertRaises(FileNotFoundError): require_replication({})

    def test_gate_content_and_hash_both_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'replication/analysis').mkdir(parents=True)
            report = root/'replication/analysis/replication_report.json'
            report.write_text(json.dumps({'gate': {'status': 'passed'}}))
            marker = root/'replication/REPLICATION_COMPLETE'
            marker.write_text('status=complete\ngate_status=passed\nreport_sha256=wrong\n')
            with patch('length_budget_distill.legacy_sae_continuation.source_hashes'), patch('length_budget_distill.legacy_sae_continuation.root_for', return_value=root):
                with self.assertRaises(RuntimeError): require_replication({})
                marker.write_text(f'status=complete\ngate_status=passed\nreport_sha256={file_sha256(report)}\n')
                self.assertIn('replication_report_sha256', require_replication({}))
                report.write_text(json.dumps({'gate': {'status': 'failed'}}))
                marker.write_text(f'status=complete\ngate_status=failed\nreport_sha256={file_sha256(report)}\n')
                with self.assertRaises(RuntimeError): require_replication({})


if __name__ == '__main__':
    unittest.main()
