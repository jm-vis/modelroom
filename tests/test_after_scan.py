"""Tests of the repository's own step after the scan: the public hygiene check.

The steps are `.githooks/after-scan/pre-commit` and `.githooks/after-scan/pre-push`; the hooks of the module run them
after a clean scan, and `.githooks/after-scan.required` names both. The hygiene check itself has its own tests
(`tests/test_public_hygiene.py`); here a stand-in for `uv` on the path records the call, so the order (scan first,
hygiene after a clean scan, refusal on red) is proved without the project environment. The helpers are those of the
module's hook test. Skipped with a reason unless `GITLEAKS_HOOK_TESTS=1` and `GITLEAKS_TEST_BINARY` are set
(`GITLEAKS_REQUIRED=1` fails instead).
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gitleaks_hooks as hooks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def setUpModule():
    hooks.setUpModule()


def tearDownModule():
    hooks.tearDownModule()


class HygieneAfterScan(hooks.HookCase):
    def setUp(self):
        super().setUp()
        self.repo, self.remote = hooks.build_normal()
        steps = self.repo / ".githooks" / "after-scan"
        steps.mkdir(parents=True, exist_ok=True)
        for name in ("pre-commit", "pre-push"):
            shutil.copy2(ROOT / ".githooks" / "after-scan" / name, steps / name)
            (steps / name).chmod(0o755)
        shutil.copy2(ROOT / ".githooks" / "after-scan.required", self.repo / ".githooks" / "after-scan.required")
        self.rec = hooks.WORK / ("rec" + hooks.rnd(6, hooks.LOWER))
        self.rec.mkdir()
        self.bin = hooks.WORK / ("bin" + hooks.rnd(6, hooks.LOWER))
        self.bin.mkdir()
        self.status = self.rec / "rc.txt"
        self.status.write_text("0", encoding="utf-8")
        posix = str(self.rec).replace("\\", "/")
        fake = self.bin / "uv"
        fake.write_bytes(('#!/bin/sh\nrec="' + posix + '"\nprintf "%s\\n" "$*" >> "$rec/uv_calls.txt"\n'
                          'exit "$(cat "$rec/rc.txt")"\n').encode("utf-8"))
        fake.chmod(0o755)

    def env(self) -> dict:
        return hooks.env_for({"PATH": os.pathsep.join([str(self.bin), os.environ.get("PATH", "")])})

    def calls(self) -> list[str]:
        path = self.rec / "uv_calls.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def stage_clean(self) -> None:
        hooks.write(self.repo, "clean" + hooks.rnd(4, hooks.LOWER) + ".txt", "clean\n")
        hooks.git(self.repo, "add", "-A")

    def test_the_hygiene_check_runs_after_a_clean_scan_and_a_clean_commit_passes(self):
        self.stage_clean()
        done = hooks.commit_hooked(self.repo, "control", env=self.env())
        self.assertEqual(done.returncode, 0, "the clean commit was refused")
        self.assertTrue(hooks.has_line(hooks.text_of(done), r"gitleaks_gate: pre-commit: clean"), "the scan did not run")
        self.assertEqual(self.calls(), ["run --frozen python tests/test_public_hygiene.py"], "the hygiene check did not run once")

    def test_a_red_hygiene_check_refuses_the_commit_after_the_scan(self):
        self.stage_clean()
        self.status.write_text("1", encoding="utf-8")
        before = hooks.count(self.repo)
        done = hooks.commit_hooked(self.repo, "control", env=self.env())
        self.assert_refused(done.returncode, "commit with a red hygiene check")
        lines = hooks.text_of(done).splitlines()
        self.assertTrue("pre-commit: public hygiene check failed, commit refused" in lines, "the fixed line is missing")
        self.assertTrue("gitleaks_gate: pre-commit: clean" in lines, "the scan did not pass first")
        self.assertEqual(hooks.count(self.repo), before, "a commit was made")

    def test_a_bait_is_refused_by_the_scan_and_the_hygiene_check_is_not_started(self):
        hooks.write(self.repo, "leak.py", f'GITHUB_TOKEN = "{hooks.github_token()}"\n')
        hooks.git(self.repo, "add", "-A")
        done = hooks.commit_hooked(self.repo, "leak", env=self.env())
        self.assert_refused(done.returncode, "commit with a bait")
        self.assert_refused_by_finding(hooks.text_of(done), "pre-commit", "bait")
        self.assertEqual(self.calls(), [], "the hygiene check ran after a red scan")

    def test_the_push_runs_the_hygiene_check_after_a_clean_scan_and_refuses_on_red(self):
        self.stage_clean()
        hooks.git(self.repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "plain")
        done = hooks.git(self.repo, "push", "origin", "main", env=self.env(), check=False)
        self.assertEqual(done.returncode, 0, "the clean push was refused")
        self.assertEqual(self.calls(), ["run --frozen python tests/test_public_hygiene.py"], "the hygiene check did not run once")
        hooks.write(self.repo, "more.txt", "more\n")
        hooks.git(self.repo, "add", "-A")
        hooks.git(self.repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "more")
        self.status.write_text("1", encoding="utf-8")
        done = hooks.git(self.repo, "push", "origin", "main", env=self.env(), check=False)
        self.assert_refused(done.returncode, "push with a red hygiene check")
        self.assertTrue("pre-push: public hygiene check failed, push refused" in hooks.text_of(done).splitlines(),
                        "the fixed line is missing")

    def test_both_steps_are_named_in_the_required_file_and_a_missing_one_refuses(self):
        self.assertEqual((ROOT / ".githooks" / "after-scan.required").read_bytes(), b"pre-commit\npre-push\n",
                         "the required-steps file is not the one of this repository")
        (self.repo / ".githooks" / "after-scan" / "pre-commit").unlink()
        self.stage_clean()
        done = hooks.commit_hooked(self.repo, "control", env=self.env())
        self.assert_refused(done.returncode, "commit without the named step")
        self.assertTrue("pre-commit: the step required by .githooks/after-scan.required is missing, commit refused"
                        in hooks.text_of(done).splitlines(), "the fixed line of the missing step is missing")


if __name__ == "__main__":
    unittest.main()
