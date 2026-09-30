"""Unit tests for changelog detection helpers (no network)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import changelog


class TestChangelog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_file = self.root / "changelog_state.json"
        self.pending_file = self.root / "changelog_pending.md"
        self._patches = [
            mock.patch.object(changelog, "STATE_FILE", self.state_file),
            mock.patch.object(changelog, "PENDING_FILE", self.pending_file),
            mock.patch.object(changelog, "REPO_ROOT", self.root),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def test_has_pending_when_notes_exist(self):
        self.pending_file.write_text("- feature x\n", encoding="utf-8")
        with mock.patch.object(changelog, "get_head_sha", return_value="abc"):
            self.assertTrue(changelog.has_pending_changes({}))

    def test_skip_same_head_suppresses_prompt(self):
        self.pending_file.write_text("- feature x\n", encoding="utf-8")
        state = {"last_prompted_sha": "abc", "last_broadcast_sha": ""}
        with mock.patch.object(changelog, "get_head_sha", return_value="abc"):
            self.assertFalse(changelog.has_pending_changes(state))

    def test_new_head_triggers_prompt(self):
        state = {"last_broadcast_sha": "old", "last_prompted_sha": "old"}
        with mock.patch.object(changelog, "get_head_sha", return_value="new"):
            self.assertTrue(changelog.has_pending_changes(state))

    def test_fallback_draft(self):
        text = changelog._fallback_changelog("a1b2c3d Add crypto menu", "- قیمت ارز دیجیتال")
        self.assertIn("ارز", text)

    def test_mark_broadcast_clears_pending(self):
        self.pending_file.write_text("- note\n", encoding="utf-8")
        changelog.mark_broadcast("deadbeef")
        self.assertEqual(self.pending_file.read_text(encoding="utf-8"), "")
        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["last_broadcast_sha"], "deadbeef")
        self.assertEqual(state["last_changelog_sha"], "deadbeef")

    def test_skip_clears_pending_and_watermark(self):
        self.pending_file.write_text("- skipped feature\n", encoding="utf-8")
        changelog.mark_skipped("sha_a")
        self.assertEqual(self.pending_file.read_text(encoding="utf-8"), "")
        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["last_changelog_sha"], "sha_a")
        self.assertEqual(state["last_prompted_sha"], "sha_a")
        with mock.patch.object(changelog, "get_head_sha", return_value="sha_a"):
            self.assertFalse(changelog.has_pending_changes(state))

    def test_new_head_after_skip_still_prompts(self):
        """New deploy after skip should prompt (new commits), without old pending bullets."""
        changelog.mark_skipped("sha_a")
        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        with mock.patch.object(changelog, "get_head_sha", return_value="sha_b"):
            self.assertTrue(changelog.has_pending_changes(state))
        ctx = changelog.build_change_context(state)
        self.assertEqual(ctx["since_sha"], "sha_a")


if __name__ == "__main__":
    unittest.main()
