"""Obfuscation operators T_family(text, rho) for the retrieval-evasion trade-off study.

Each operator transforms approximately a fraction `rho in [0,1]` of the eligible
units in `text`. `rho` is the paper's obfuscation-intensity parameter.
Selection is deterministic given (rho, seed) so runs are reproducible.
"""
import base64
import random

# Latin -> visually-confusable codepoints (Cyrillic / Greek). NFKC does NOT
# normalise these, which is why homoglyphs survive an NFKC defence.
HOMOGLYPHS = {
    'a': '\u0430', 'c': '\u0441', 'e': '\u0435', 'o': '\u043E', 'p': '\u0440',
    'x': '\u0445', 'y': '\u0443', 'i': '\u0456', 's': '\u0455', 'j': '\u0458',
    'A': '\u0410', 'B': '\u0412', 'C': '\u0421', 'E': '\u0415', 'H': '\u041D',
    'K': '\u041A', 'M': '\u041C', 'O': '\u041E', 'P': '\u0420', 'T': '\u0422',
    'X': '\u0425', 'Y': '\u0423',
}
ZERO_WIDTH = ['\u200b', '\u200c', '\u200d']  # ZWSP, ZWNJ, ZWJ
TAG_OFFSET = 0xE0000  # Unicode Tags block: ASCII 0x21..0x7e -> U+E0021..U+E007e


def _select(n, rho, seed):
    """Return a deterministic set of round(rho*n) indices in [0, n)."""
    rng = random.Random(seed)
    k = round(rho * n)
    idx = list(range(n))
    rng.shuffle(idx)
    return set(idx[:k])


def unicode_tags(text, rho=1.0, seed=0):
    chars = list(text)
    pos = [i for i, ch in enumerate(chars) if 0x21 <= ord(ch) <= 0x7e]
    chosen = _select(len(pos), rho, seed)
    for j, i in enumerate(pos):
        if j in chosen:
            chars[i] = chr(TAG_OFFSET + ord(chars[i]))
    return ''.join(chars)


def zero_width(text, rho=1.0, seed=0):
    rng = random.Random(seed + 7)
    chars = list(text)
    chosen = _select(len(chars), rho, seed)
    out = []
    for i, ch in enumerate(chars):
        out.append(ch)
        if i in chosen:
            out.append(rng.choice(ZERO_WIDTH))
    return ''.join(out)


def homoglyph(text, rho=1.0, seed=0):
    chars = list(text)
    pos = [i for i, ch in enumerate(chars) if ch in HOMOGLYPHS]
    chosen = _select(len(pos), rho, seed)
    for j, i in enumerate(pos):
        if j in chosen:
            chars[i] = HOMOGLYPHS[chars[i]]
    return ''.join(chars)


def _scramble(word, rng):
    pre = ''
    suf = ''
    core = word
    while core and not core[0].isalpha():
        pre += core[0]; core = core[1:]
    while core and not core[-1].isalpha():
        suf = core[-1] + suf; core = core[:-1]
    if len(core) <= 3:
        return word
    mid = list(core[1:-1]); rng.shuffle(mid)
    return pre + core[0] + ''.join(mid) + core[-1] + suf


def typoglycemia(text, rho=1.0, seed=0):
    rng = random.Random(seed + 13)
    words = text.split(' ')
    eligible = [i for i, w in enumerate(words)
                if sum(c.isalpha() for c in w) >= 4]
    chosen = _select(len(eligible), rho, seed)
    for j, i in enumerate(eligible):
        if j in chosen:
            words[i] = _scramble(words[i], rng)
    return ' '.join(words)


def base64_encode(text, rho=1.0, seed=0):
    n = len(text)
    k = round(rho * n)
    if k == 0:
        return text
    enc = base64.b64encode(text[:k].encode('utf-8')).decode('ascii')
    return (f"[decode base64] {enc} {text[k:]}").strip()


def hybrid(text, rho=1.0, seed=0, trigger="", inner="homoglyph"):
    """Clean retrieval trigger + obfuscated payload (Chang et al. split)."""
    fn = OBFUSCATORS[inner]
    return f"{trigger} {fn(text, rho=rho, seed=seed)}".strip()


OBFUSCATORS = {
    'unicode_tags': unicode_tags,
    'zero_width': zero_width,
    'homoglyph': homoglyph,
    'typoglycemia': typoglycemia,
    'base64': base64_encode,
    'hybrid': hybrid,
}
