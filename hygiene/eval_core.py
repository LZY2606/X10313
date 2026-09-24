"""Interpreter for fully expanded core forms.

Runtime errors keep the offending syntax object so the UI can walk its birth
certificate / origin chain back to an original token span ("error source
reflection"). Evaluation is independent of identifier spelling: it consumes
expanded core, and unknown bindings surface as EvalError, not as silent
renames.
"""
from __future__ import annotations


class EvalError(Exception):
    def __init__(self, msg, obj=None):
        self.msg = msg
        self.obj = obj
        super().__init__(msg)


class Closure:
    def __init__(self, params, bodies, env, expander):
        self.params = params
        self.bodies = bodies
        self.env = env
        self.expander = expander


def _primitives():
    def check_nums(vals, obj):
        for v in vals:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise EvalError(f"arithmetic expected a number, got {v!r}", obj)

    def plus(vals, obj):
        check_nums(vals, obj)
        return sum(vals)

    def minus(vals, obj):
        check_nums(vals, obj)
        return vals[0] - sum(vals[1:]) if len(vals) >= 2 else -vals[0]

    def times(vals, obj):
        r = 1
        check_nums(vals, obj)
        for v in vals:
            r *= v
        return r

    def equal(vals, obj):
        return _cmp(vals, obj, lambda a, b: a == b)

    def lt(vals, obj):
        return _cmp(vals, obj, lambda a, b: a < b)

    def gt(vals, obj):
        return _cmp(vals, obj, lambda a, b: a > b)

    def _cmp(vals, obj, pred):
        check_nums(vals, obj)
        for a, b in zip(vals, vals[1:]):
            if not pred(a, b):
                return False
        return True

    def cons(vals, obj):
        if len(vals) != 2:
            raise EvalError("cons expects 2 arguments", obj)
        return ["pair", vals[0], vals[1]]

    def car(vals, obj):
        if len(vals) == 1 and isinstance(vals[0], list) and vals[0] and vals[0][0] == "pair":
            return vals[0][1]
        raise EvalError("car expects a pair", obj)

    def cdr(vals, obj):
        if len(vals) == 1 and isinstance(vals[0], list) and vals[0] and vals[0][0] == "pair":
            return vals[0][2]
        raise EvalError("cdr expects a pair", obj)

    def lst(vals, obj):
        r = ["null"]
        for v in reversed(vals):
            r = ["pair", v, r]
        return r

    def nullp(vals, obj):
        return len(vals) == 1 and isinstance(vals[0], list) and vals[0] == ["null"]

    def eqp(vals, obj):
        return len(vals) == 2 and vals[0] == vals[1]

    def display(vals, obj):
        return None

    return {"+": plus, "-": minus, "*": times, "=": equal, "<": lt, ">": gt,
            "cons": cons, "car": car, "cdr": cdr, "list": lst, "null?": nullp,
            "eq?": eqp, "display": display}


def evaluate(forms, expander):
    """Evaluate expanded top-level forms. Returns (results, trace)."""
    env = {"__prim__": _primitives()}
    trace = []
    results = []
    for form in forms:
        val = _eval(form, env, expander, trace)
        results.append(val)
    return results, trace


def _eval(obj, env, expander, trace):
    if not obj.is_list:
        if obj.symbol is None:
            return obj.val
        return _lookup(obj, env, expander)
    if not obj.val:
        raise EvalError("cannot evaluate an empty list ()", obj)
    head = obj.val[0].symbol
    if head == "quote":
        return _quote_value(obj.val[1])
    if head == "lambda":
        return Closure([p.symbol for p in obj.val[1].val], obj.val[2:], dict(env),
                       expander)
    if head == "let":
        new_env = dict(env)
        for clause in obj.val[1].val:
            name = clause.val[0].symbol
            new_env[name] = _eval(clause.val[1], env, expander, trace)
        return _eval(obj.val[2], new_env, expander, trace)
    if head == "if":
        test = _eval(obj.val[1], env, expander, trace)
        branch = obj.val[2] if test else (obj.val[3] if len(obj.val) > 3 else None)
        if branch is None:
            return None
        return _eval(branch, env, expander, trace)
    if head == "begin":
        val = None
        for sub in obj.val[1:]:
            val = _eval(sub, env, expander, trace)
        return val
    if head == "set!":
        name = obj.val[1].symbol
        env[name] = _eval(obj.val[2], env, expander, trace)
        return None
    if head == "define":
        env[obj.val[1].symbol] = _eval(obj.val[2], env, expander, trace)
        return None
    # application
    proc = _eval(obj.val[0], env, expander, trace)
    args = [_eval(a, env, expander, trace) for a in obj.val[1:]]
    trace.append({"oid": obj.oid, "proc": _proc_name(proc),
                  "argc": len(args), "span": obj.span.as_dict() if obj.span else None,
                  "birth": obj.birth.as_dict() if obj.birth else None})
    if isinstance(proc, Closure):
        if len(args) != len(proc.params):
            raise EvalError(f"procedure expects {len(proc.params)} arguments, got "
                            f"{len(args)}", obj)
        call_env = dict(proc.env)
        call_env.update(dict(zip(proc.params, args)))
        val = None
        for body in proc.bodies:
            val = _eval(body, call_env, expander, trace)
        return val
    if callable(proc):
        return proc(args, obj)
    raise EvalError(f"not applicable: {proc!r}", obj.val[0])


def _proc_name(proc):
    if isinstance(proc, Closure):
        return "closure:" + ",".join(proc.params)
    return "primitive"


def _lookup(obj, env, expander):
    sym = obj.symbol
    if sym in env:
        return env[sym]
    prim = env.get("__prim__", {}).get(sym)
    if prim is not None:
        return prim
    raise EvalError(f"unbound identifier {sym!r} (no visible binding in its scope set)",
                    obj)


def _quote_value(s):
    if s.is_list:
        if not s.val:
            return ["null"]
        head = s.val[0].symbol
        if head == "quote":
            return ["null"]
        out = ["null"]
        for e in reversed(s.val):
            out = ["pair", _quote_value(e), out]
        return out
    return s.val
