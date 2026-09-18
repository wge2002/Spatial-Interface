"""Fresh-run, non-destructive port handling, and root identity regressions."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

from spatial_interface.experiment import parser, plan, TASKS
from spatial_interface.identity import snapshot
from spatial_interface.ports import require_ports_free


class PublicCLI(unittest.TestCase):
    def test_current_method_and_model_are_explicit(self):
        result = plan(parser().parse_args([]))
        self.assertEqual((result['interface'], result['feedback']), ('direct_geometry', 'grounded'))
        self.assertEqual((result['model'], result['effort']), ('gpt-6-astra', 'medium'))
        self.assertEqual(result['automatic_retries'], 0)

    def test_smoke_cannot_use_a_full_experiment_budget(self):
        result = plan(parser().parse_args(['--smoke', '--timeout', '9999']))
        self.assertEqual(result['timeout_s'], 180)
        self.assertEqual(result['kind'], 'integration-smoke')

    def test_all_seven_tasks_and_manual_rainbow_are_preserved(self):
        self.assertEqual(set(TASKS), {'stack', 't_block', 'rainbow', 'libero_goal/0', 'libero_goal/4', 'libero_goal/7', 'libero_goal/8'})
        self.assertFalse(TASKS['rainbow']['detector'])

    def test_bad_port_and_qwen_defaults_are_rejected(self):
        for args in [['--base-port', '65534'], ['--timeout', '0'], ['--harness', 'qwen']]:
            with self.assertRaises(ValueError):
                plan(parser().parse_args(args))

    def test_occupied_listener_survives(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                require_ports_free([port])
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                connection, _ = listener.accept()
                connection.close()


class RootIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.root / 'method.py').write_text('VALUE = 1\n')
        (self.root / 'methods').mkdir()
        self.registry = {'releases': {'si-r1': {'anchor': 'unique-root-commit',
            'files': {'method.py': hashlib.sha256((self.root / 'method.py').read_bytes()).hexdigest()}}},
            'client_profiles': {'si-codex-r1': {'files': {}}}}
        (self.root / 'methods/registry.json').write_text(json.dumps(self.registry))
        self.git('add', '.')
        self.git('commit', '-m', 'Initial fixture')

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL, text=True).strip()

    def test_snapshot_works_with_exactly_one_root_commit(self):
        result = snapshot(root=self.root)
        self.assertEqual(result['source_commit'], result['release_anchor'])
        self.assertEqual(self.git('rev-list', '--all', '--count'), '1')

    def test_dirty_method_is_rejected(self):
        (self.root / 'method.py').write_text('VALUE = 2\n')
        with self.assertRaisesRegex(ValueError, 'uncommitted'):
            snapshot(root=self.root)

    def test_unknown_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown release'):
            snapshot('does-not-exist', root=self.root)

    def test_committed_redefinition_cannot_change_initial_identity(self):
        self.registry['releases']['si-r1']['files'] = {}
        (self.root / 'methods/registry.json').write_text(json.dumps(self.registry))
        self.git('add', '.')
        self.git('commit', '-m', 'Invalid redefinition')
        with self.assertRaisesRegex(ValueError, 'root anchor'):
            snapshot(root=self.root)
