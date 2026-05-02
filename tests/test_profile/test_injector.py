"""画像模块 — 注入器测试"""
import os
import tempfile
import sys
import importlib

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base)

spec = importlib.util.spec_from_file_location(
    "profile_injector", os.path.join(base, "core", "profile", "injector.py"))
injector_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(injector_mod)
ProfileInjector = injector_mod.ProfileInjector


def _make_profile_dir(base_dir, user_id, files):
    profile_dir = os.path.join(base_dir, "user_profile", user_id)
    os.makedirs(profile_dir, exist_ok=True)
    for fname, content in files.items():
        with open(os.path.join(profile_dir, fname), "w", encoding="utf-8") as f:
            f.write(content)
    return profile_dir


class TestInjectorBasic:
    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            injector = ProfileInjector(tmpdir)
            assert injector.read_profile_tags("nonexistent") == []

    def test_reads_profile_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {
                "preferred_name.md": "小貔貅",
                "public_identity.md": "后端工程师",
            })
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            types = {t["type"] for t in tags}
            assert "preferred_name" in types
            assert "public_identity" in types

    def test_skips_non_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {
                "preferred_name.md": "小貔貅",
                "readme.txt": "not a profile",
            })
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            assert len(tags) == 1
            assert tags[0]["type"] == "preferred_name"


class TestOptOut:
    def test_opt_out_blocks_reading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {
                "preferred_name.md": "小貔貅",
                ".opt_out": "true",
            })
            injector = ProfileInjector(tmpdir)
            assert injector.read_profile_tags("U1") == []

    def test_dot_files_skipped_as_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {
                "preferred_name.md": "小貔貅",
                ".hidden.md": "should be skipped",
            })
            injector = ProfileInjector(tmpdir)
            types = {t["type"] for t in injector.read_profile_tags("U1")}
            assert ".hidden" not in types


class TestConflictResolution:
    def test_same_type_returns_latest(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = _make_profile_dir(tmpdir, "U1", {
                "preferred_name.md": "小貔貅",
            })
            tag_path = os.path.join(profile_dir, "preferred_name.md")
            with open(tag_path, "w", encoding="utf-8") as f:
                f.write("大貔貅")
            now = time.time()
            os.utime(tag_path, (now, now))
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            names = [t for t in tags if t["type"] == "preferred_name"]
            assert len(names) == 1
            assert names[0]["value"] == "大貔貅"


class TestInject:
    def test_inject_merges_into_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {"preferred_name.md": "小貔貅"})
            injector = ProfileInjector(tmpdir)
            result = injector.inject("U1", [{"scope": "raw", "content": "普通记忆"}])
            assert len(result) == 2
            assert result[1]["scope"] == "user_profile"

    def test_inject_respects_max_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {f"tag_{i}.md": f"val_{i}" for i in range(10)}
            _make_profile_dir(tmpdir, "U1", files)
            injector = ProfileInjector(tmpdir)
            result = injector.inject("U1", [], max_inject=3)
            injected = [m for m in result if m["scope"] == "user_profile"]
            assert len(injected) == 3

    def test_inject_skips_when_no_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            injector = ProfileInjector(tmpdir)
            result = injector.inject("U1", [{"scope": "raw"}])
            assert len(result) == 1


class TestDecay:
    def test_stale_tag_marked(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = _make_profile_dir(tmpdir, "U1", {"preferred_name.md": "小貔貅"})
            stale_time = time.time() - 86400 * 100
            os.utime(os.path.join(profile_dir, "preferred_name.md"), (stale_time, stale_time))
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            assert len(tags) == 1
            assert tags[0]["stale"] is True

    def test_fresh_tag_not_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {"preferred_name.md": "小貔貅"})
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            assert tags[0]["stale"] is False

    def test_inject_marks_stale_content(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = _make_profile_dir(tmpdir, "U1", {"preferred_name.md": "小貔貅"})
            stale_time = time.time() - 86400 * 100
            os.utime(os.path.join(profile_dir, "preferred_name.md"), (stale_time, stale_time))
            injector = ProfileInjector(tmpdir)
            result = injector.inject("U1", [])
            items = [m for m in result if m["scope"] == "user_profile"]
            assert "[过期]" in items[0]["content"]


class TestVersionHeader:
    def test_write_tag_adds_version_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            injector = ProfileInjector(tmpdir)
            filepath = os.path.join(tmpdir, "test.md")
            injector.write_tag(filepath, "小貔貅")
            content = open(filepath, "r", encoding="utf-8").read()
            assert content.startswith("<!-- profile_v")
            assert "小貔貅" in content

    def test_read_strips_version_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            injector = ProfileInjector(tmpdir)
            filepath = os.path.join(tmpdir, "test.md")
            injector.write_tag(filepath, "小貔貅")
            raw = open(filepath, "r", encoding="utf-8").read()
            stripped = injector._strip_header(raw)
            assert stripped == "小貔貅"
            assert "<!--" not in stripped

    def test_future_version_header_does_not_crash(self):
        header = "<!-- profile_v99 -->"
        content = header + "\n" + "小貔貅"
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_profile_dir(tmpdir, "U1", {"preferred_name.md": content})
            injector = ProfileInjector(tmpdir)
            tags = injector.read_profile_tags("U1")
            assert len(tags) == 1
            assert tags[0]["value"] == "小貔貅"
