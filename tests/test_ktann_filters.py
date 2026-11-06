"""Typed filter and wire boundaries that do not require a bridge process."""

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from vectordb_bench.backend.clients.ktann.ktann import KTANN, _encode_insert
from vectordb_bench.backend.filter import FilterOp, IntFilter, non_filter


class FilterTests(unittest.TestCase):
    def test_invalid_filter_does_not_replace_valid_state(self):
        client = KTANN.__new__(KTANN)
        client._with_scalar_labels = True
        client.prepare_filter(IntFilter(int_value=7))
        expected = {"type": "num_ge", "value": 7}
        for field, value in [("wrong", "A"), ("labels", 1), ("labels", "界" * 342)]:
            with self.assertRaises((ValueError, TypeError)):
                client.prepare_filter(SimpleNamespace(type=FilterOp.StrEqual, label_field=field, label_value=value))
            self.assertEqual(client._filter, expected)
        for field, value in [("wrong", 1), ("id", True), ("id", 1 << 63), ("id", -(1 << 63) - 1)]:
            with self.assertRaises((ValueError, TypeError)):
                client.prepare_filter(SimpleNamespace(type=FilterOp.NumGE, int_field=field, int_value=value))
            self.assertEqual(client._filter, expected)
        client.prepare_filter(non_filter)
        self.assertIsNone(client._filter)
        client._with_scalar_labels = False
        with self.assertRaises((ValueError, TypeError)):
            client.prepare_filter(SimpleNamespace(type=FilterOp.StrEqual, label_field="labels", label_value="A"))

    def test_binary_label_tail_and_boundaries(self):
        fields = {"ids": [1, 2], "vectors": [[1.], [2.]]}
        body = b"KTI" + struct.pack("!II", 2, 1) + struct.pack("<qqff", 1, 2, 1., 2.)
        self.assertEqual(_encode_insert(fields), body + b"\x00")
        self.assertEqual(_encode_insert({**fields, "labels": ["", "é"]}),
                         body + b"\x01\x00\x00\x00\x02" + "é".encode())
        for labels in [["A"], ["A", None], ["A", "界" * 342], ["A", "\ud800"]]:
            with self.assertRaises((ValueError, TypeError)):
                _encode_insert({**fields, "labels": labels})

    def test_canonical_label_cli_validates_dataset_and_proportion(self):
        from vectordb_bench.backend.clients.ktann.cli import KTANN as command
        runner = CliRunner()
        args = ["--socket-path", "/tmp/unused", "--dataset-identity", "cohere-labels",
                "--case-type", "LabelFilterPerformanceCase", "--label-percentage", "0.01"]
        with patch("vectordb_bench.backend.clients.ktann.cli.run") as run:
            result = runner.invoke(command, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(run.call_args.kwargs["label_percentage"], 0.01)
            run.reset_mock()
            for extra in [["--label-percentage", "0.123"], ["--dataset-with-size-type", "Small Cohere (768dim, 100K)"]]:
                result = runner.invoke(command, args + extra)
                self.assertEqual(result.exit_code, 2, result.output)
                run.assert_not_called()
