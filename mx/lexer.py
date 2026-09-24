"""Lexer for the small MxLisp language.

Original tokens are preserved verbatim (including whitespace and comments)
with 1-based line/col and 0-based offset spans; the source text itself is
never overwritten after submission.
"""
from .errors import LexError

DELIMITERS = set("()'`,\";")
SYMBOL_TERMINATORS = DELIMITERS | set(" \t\r\n")


def tokenize(text, file_id="<input>"):
    tokens = []
    i = 0
    line, col = 1, 1
    n = len(text)

    def span(start, end, sl, sc):
        return {"file": file_id, "start": {"offset": start, "line": sl, "col": sc},
                "end": {"offset": end, "line": line, "col": col}}

    while i < n:
        c = text[i]
        sl, sc, start = line, col, i
        if c in " \t\r\n":
            j = i
            while j < n and text[j] in " \t\r\n":
                if text[j] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                j += 1
            tokens.append({"type": "ws", "value": text[i:j], "raw": text[i:j],
                           "span": span(start, j, sl, sc)})
            i = j
            continue
        if c == ";":
            j = i
            while j < n and text[j] != "\n":
                j += 1
                col += 1
            tokens.append({"type": "comment", "value": text[i:j], "raw": text[i:j],
                           "span": span(start, j, sl, sc)})
            i = j
            continue
        if c in "()":
            tokens.append({"type": "paren", "value": c, "raw": c,
                           "span": span(i, i + 1, sl, sc)})
            i += 1
            col += 1
            continue
        if c == '"':
            j = i + 1
            col += 1
            buf = []
            closed = False
            while j < n:
                ch = text[j]
                if ch == "\\":
                    if j + 1 >= n:
                        raise LexError("unterminated string escape", line, col, j)
                    buf.append(text[j + 1])
                    j += 2
                    col += 2
                    continue
                if ch == '"':
                    j += 1
                    col += 1
                    closed = True
                    break
                if ch == "\n":
                    raise LexError("newline inside string literal", line, col, j)
                buf.append(ch)
                j += 1
                col += 1
            if not closed:
                raise LexError("unterminated string literal", line, col, i)
            raw = text[i:j]
            tokens.append({"type": "string", "value": "".join(buf), "raw": raw,
                           "span": span(start, j, sl, sc)})
            i = j
            continue
        if c in "'`,":
            nxt = text[i:i + 2]
            if nxt == ",@":
                tokens.append({"type": "quote", "value": ",@", "raw": ",@",
                               "span": span(start, i + 2, sl, sc)})
                i += 2
                col += 2
            else:
                tokens.append({"type": "quote", "value": c, "raw": c,
                               "span": span(start, i + 1, sl, sc)})
                i += 1
                col += 1
            continue
        if c == "#":
            j = i + 1
            col += 1
            if j < n and text[j] in "tf":
                val = text[j] == "t"
                tokens.append({"type": "bool", "value": val, "raw": text[i:j + 1],
                               "span": span(start, j + 1, sl, sc)})
                j += 1
                col += 1
                i = j
                continue
            raise LexError("illegal token starting with '#'", line, col, i)
        if c == "-" and i + 1 < n and text[i + 1].isdigit():
            pass
        if c.isdigit() or (c == "-" and i + 1 < n and (text[i + 1].isdigit()
                                                       or text[i + 1] == ".")):
            j = i
            dots = 0
            if text[j] == "-":
                j += 1
            while j < n and (text[j].isdigit() or text[j] == "."):
                if text[j] == ".":
                    dots += 1
                j += 1
            raw = text[i:j]
            if dots > 1:
                raise LexError(f"illegal numeric literal {raw!r}", sl, sc, i)
            val = float(raw) if dots else int(raw)
            tokens.append({"type": "number", "value": val, "raw": raw,
                           "span": span(start, j, sl, sc)})
            col += j - i
            i = j
            continue
        if c == "|":
            raise LexError("escaped symbols using |...| are not supported", line, col, i)
        # symbol
        j = i
        while j < n and text[j] not in SYMBOL_TERMINATORS:
            j += 1
        raw = text[i:j]
        if not raw:
            raise LexError(f"unexpected character {c!r}", line, col, i)
        tokens.append({"type": "symbol", "value": raw, "raw": raw,
                       "span": span(start, j, sl, sc)})
        col += j - i
        i = j
    return tokens


def significant(tokens):
    return [t for t in tokens if t["type"] not in ("ws", "comment")]
