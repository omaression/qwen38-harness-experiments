"""Pure-stdlib unified diff applier tests (diff_apply.py)."""

from __future__ import annotations

from llm_eval_harness.diff_apply import apply_unified_diff


def test_exact_single_and_multi_file_patch() -> None:
    files = {
        "a.py": "one\ntwo\nthree\nfour\n",
        "b.py": "alpha\nbeta\ngamma\n",
        "untouched.py": "keep\n",
    }
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,4 +1,4 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
        " four\n"
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+delta\n"
        " gamma\n"
    )
    updated, status = apply_unified_diff(files, diff)
    assert status == "exact"
    assert updated == {
        "a.py": "one\nTWO\nthree\nfour\n",
        "b.py": "alpha\nbeta\ndelta\ngamma\n",
        "untouched.py": "keep\n",
    }


def test_add_delete_and_rename_files() -> None:
    files = {
        "old.py": "bye\nworld\n",
        "empty_del.py": "",
        "move_me.py": "hello\n",
    }
    diff = (
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+line1\n"
        "+line2\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-bye\n"
        "-world\n"
        "diff --git a/empty_new.py b/empty_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/empty_new.py\n"
        "diff --git a/empty_del.py b/empty_del.py\n"
        "deleted file mode 100644\n"
        "--- a/empty_del.py\n"
        "+++ /dev/null\n"
        "diff --git a/move_me.py b/moved.py\n"
        "similarity index 100%\n"
        "rename from move_me.py\n"
        "rename to moved.py\n"
    )
    updated, status = apply_unified_diff(files, diff)
    assert status == "exact"
    assert updated == {
        "old.py": None,
        "empty_del.py": None,
        "move_me.py": None,
        "moved.py": "hello\n",
        "new.py": "line1\nline2\n",
        "empty_new.py": "",
    }


def test_no_newline_at_end_of_file() -> None:
    files = {"a.txt": "first\nsecond"}
    diff_add_nl = (
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " first\n"
        "-second\n"
        "\\ No newline at end of file\n"
        "+second\n"
    )
    updated, status = apply_unified_diff(files, diff_add_nl)
    assert status == "exact" and updated == {"a.txt": "first\nsecond\n"}

    diff_drop_nl = (
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1 +1 @@\n"
        "-hi\n"
        "+HI\n"
        "\\ No newline at end of file\n"
    )
    updated2, status2 = apply_unified_diff({"b.txt": "hi\n"}, diff_drop_nl)
    assert status2 == "exact" and updated2 == {"b.txt": "HI"}


def test_offset_fuzz_within_and_beyond_50_lines() -> None:
    lines = [f"line {i}" for i in range(1, 80)]
    orig = "\n".join(lines) + "\n"
    # Target block is actually at line 56, header claims line 20 (offset +36 <= 50).
    ok_diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -20,3 +20,3 @@\n"
        " line 56\n"
        "-line 57\n"
        "+LINE 57\n"
        " line 58\n"
    )
    updated, status = apply_unified_diff({"f.py": orig}, ok_diff)
    assert status == "fuzz" and updated is not None and "LINE 57\n" in updated["f.py"]

    # Offset +55 > 50 succeeds as fuzz when the context match is unique across the file,
    # and fails when the far match is ambiguous (repeated).
    far_lines = [f"line {i}" for i in range(1, 100)]
    far_diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -10,3 +10,3 @@\n"
        " line 65\n"
        "-line 66\n"
        "+LINE 66\n"
        " line 67\n"
    )
    far_updated, far_status = apply_unified_diff({"f.py": "\n".join(far_lines) + "\n"}, far_diff)
    assert far_status == "fuzz" and far_updated is not None and "LINE 66\n" in far_updated["f.py"]

    dup_far_lines = far_lines + ["line 65", "line 66", "line 67"]
    res, reason = apply_unified_diff({"f.py": "\n".join(dup_far_lines) + "\n"}, far_diff)
    assert res is None and "does not match" in reason


def test_ambiguous_hunk_and_trailing_whitespace_fuzz() -> None:
    repeated = "x = 1\ny = 2\n" * 4
    ambig = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -2,2 +2,2 @@\n"
        " x = 1\n"
        "-y = 2\n"
        "+y = 3\n"
    )
    res, reason = apply_unified_diff({"f.py": repeated}, ambig)
    assert res is None and "ambiguous" in reason

    ws_file = "def f():   \n    return 1\t \n"
    ws_diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    updated, status = apply_unified_diff({"f.py": ws_file}, ws_diff)
    assert status == "fuzz" and updated == {"f.py": "def f():   \n    return 2\n"}


def test_blank_context_line_without_leading_space() -> None:
    files = {"a.py": "a = 1\n\nb = 2\n"}
    diff = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "\n"
        "-b = 2\n"
        "+b = 3\n"
    )
    updated, status = apply_unified_diff(files, diff)
    assert status == "exact" and updated == {"a.py": "a = 1\n\nb = 3\n"}


def test_security_and_invalid_diff_rejections() -> None:
    files = {"a.py": "x = 1\n"}
    bad_diffs = [
        "",
        "not a diff at all\n",
        "--- a/../secret.py\n+++ b/../secret.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        "--- a//etc/passwd\n+++ b//etc/passwd\n@@ -1 +1 @@\n-x\n+y\n",
        "--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-x\n+y\n",
        "--- a/dir\\file.py\n+++ b/dir\\file.py\n@@ -1 +1 @@\n-x\n+y\n",
        "diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+target\n",
        "diff --git a/sub b/sub\nnew file mode 160000\n",
        "diff --git a/bin.dat b/bin.dat\nBinary files a/bin.dat and b/bin.dat differ\n",
        "--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-x\n+y\n",
        "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x = 2\n",
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 999\n+x = 2\n",
    ]
    for bad in bad_diffs:
        res, reason = apply_unified_diff(files, bad)
        assert res is None and reason
