"""Independent checks for the two trained-model RTL simulator reports.

This script checks reproducibility and rejects incomplete or misleading reports.
It cannot establish FPGA board execution or power efficiency.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SAMPLES = 50
EXPECTED_TOKENS = 400
EXPECTED_ACCURACY = 49 / 50


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_report(path, trace):
    path, trace = Path(path), Path(trace)
    report = json.loads(path.read_text())
    rows = [line.strip() for line in trace.read_text().splitlines()]
    if len(rows) != EXPECTED_TOKENS:
        raise AssertionError(f'{trace}: expected {EXPECTED_TOKENS} outputs, got {len(rows)}')
    for i, row in enumerate(rows):
        if len(row) != 32 or any(c not in '0123456789abcdefABCDEF' for c in row):
            raise AssertionError(f'{trace}: malformed 128-bit word {i}')
    if report['samples'] != EXPECTED_SAMPLES or report['rtl_scan_transfers'] != EXPECTED_TOKENS:
        raise AssertionError(f'{path}: incorrect workload size')
    if report['trace_sha256'] != sha256(trace):
        raise AssertionError(f'{path}: trace checksum mismatch')
    if report['rtl_vs_integer_max_logit_abs_error'] != 0.0:
        raise AssertionError(f'{path}: RTL vs integer logit mismatch')
    if report['rtl_vs_integer_prediction_changes'] != 0:
        raise AssertionError(f'{path}: predicted class mismatch')
    predictions = report['per_sample']
    if len(predictions) != EXPECTED_SAMPLES:
        raise AssertionError(f'{path}: incomplete sample report')
    for i, sample in enumerate(predictions):
        if sample['sample'] != i or sample['rtl_prediction'] != sample['integer_prediction']:
            raise AssertionError(f'{path}: misordered or incorrect prediction at {i}')
    computed = sum(p['rtl_prediction'] == p['label'] for p in predictions) / len(predictions)
    if abs(computed - report['rtl_injected_accuracy']) > 1e-6:
        raise AssertionError(f'{path}: accuracy inconsistent with per-sample predictions')
    if abs(computed - EXPECTED_ACCURACY) > 1e-6:
        raise AssertionError(f'{path}: checkpoint/dataset accuracy unexpectedly changed ({computed})')
    return report


def verify(build_dir):
    b = Path(build_dir)
    names = ('iverilog', 'verilator')
    reports = [validate_report(b / f'rtl_classifier_{name}.json',
                               b / f'precision_scan_outputs_{name}.txt') for name in names]
    if reports[0]['checkpoint_sha256'] != reports[1]['checkpoint_sha256']:
        raise AssertionError('Simulators used different checkpoints')
    if reports[0]['trace_sha256'] != reports[1]['trace_sha256']:
        raise AssertionError('Icarus and Verilator produced different scan output traces')
    if reports[0]['per_sample'] != reports[1]['per_sample']:
        raise AssertionError('Icarus and Verilator predictions differ')
    print(f'PASS: two simulator traces match; {EXPECTED_TOKENS} scan outputs; '
          f'{EXPECTED_SAMPLES} predictions; accuracy {EXPECTED_ACCURACY:.2%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', default='build')
    verify(parser.parse_args().build_dir)
