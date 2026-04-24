"""Rule-based caveman compressor tests."""

from tokamak.caveman import compress_rules


def test_drops_filler_and_pleasantries():
    src = "Sure, let me think. Basically, the bug is just an off-by-one in the loop."
    out = compress_rules(src, level="lite").compressed
    for word in ["sure", "let me think", "basically", "just"]:
        assert word.lower() not in out.lower()
    assert "off-by-one" in out
    assert "loop" in out


def test_preserves_code_blocks():
    src = (
        "Okay so basically wrap it in useMemo:\n"
        "```js\n"
        "const x = useMemo(() => ({ a: 1 }), []);\n"
        "```\n"
        "That should fix it actually."
    )
    out = compress_rules(src).compressed
    assert "useMemo(() => ({ a: 1 }), [])" in out
    assert "okay so" not in out.lower()
    assert "basically" not in out.lower()
    assert "actually" not in out.lower()


def test_preserves_paths_and_inline_code():
    src = "I think you should just check `/etc/passwd` because, basically, that's where it lives."
    out = compress_rules(src).compressed
    assert "`/etc/passwd`" in out
    assert "i think" not in out.lower()


def test_preserves_quoted_input_and_errors():
    src = 'Sure! The error message says "Connection refused: 127.0.0.1:5432" — basically Postgres is down.'
    out = compress_rules(src).compressed
    assert '"Connection refused: 127.0.0.1:5432"' in out
    assert "sure!" not in out.lower()


def test_replaces_verbose_phrases():
    src = "In order to fix this, we need to make use of useMemo, due to the fact that React re-renders."
    out = compress_rules(src).compressed
    assert "in order to" not in out.lower()
    assert "make use of" not in out.lower()
    assert "due to the fact that" not in out.lower()
    assert "use" in out


def test_full_drops_articles():
    src = "The bug is in the auth middleware where the token expiry check uses the wrong operator."
    out = compress_rules(src, level="full").compressed
    # "the" should be reduced
    assert out.lower().count(" the ") < src.lower().count(" the ")


def test_compression_reduces_tokens():
    src = (
        "Sure! I'd be happy to help you with that. The reason your component is "
        "re-rendering is basically that you're creating a new object reference "
        "each render. I think you should just wrap it in useMemo."
    )
    r = compress_rules(src)
    assert r.compressed_tokens < r.original_tokens
