"""The shipped offline case pack “宏作用域镜：语法对象、卫生重命名与来源展开图”.

This is *domain material*, not a generic table.  Each case is a small
program plus a narrative describing the exact hygiene/provenance boundary it
probes.  The pack's initial version is complete (tokens, spans, scope sets,
definition/use sites, macro call stack, generated symbols and transform
steps are all retained by the pipeline); later versions add:

1. a parameter revision (macro budget),
2. a source change affecting only a local dependency,
3. a conflicting human decision.
"""
from __future__ import annotations

from typing import Any

PACK_ID = "macroscope-mirror-v1"
PACK_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Shared source programs (copy-pasteable small samples)
# ---------------------------------------------------------------------------

SRC_SHADOW = """\
;; 1) 同名局部变量：宏模板引入的 x 不得捕获调用点的 x
(define-syntax shadow-x
  (lambda (stx)
    (let ((xs (syntax-e stx)))
      `(lambda (x) ,(cadr xs)))))
((lambda (x) (shadow-x x)) 11)
"""

SRC_NESTED = """\
;; 2) 嵌套宏：outer 展开为 inner 调用，inner 再展开
(define-syntax inner (lambda (stx) `(quote inner-result)))
(define-syntax outer (lambda (stx) `(inner 9)))
(outer 1)
"""

SRC_DEFSITE = """\
;; 3) definition-site 引用：宏模板里的 free 标识符 helper 在定义点解析
(define helper (quote definition-site-helper))
(define-syntax use-helper
  (lambda (stx) `helper))
((lambda (helper) (use-helper)) (quote caller-helper))
"""

SRC_CAPTURE = """\
;; 4) 显式捕获：datum->syntax 以调用点标识符为上下文，捕获判定来自 scope 集
(define-syntax refer-it
  (lambda (stx)
    (let ((xs (syntax-e stx)))
      (let ((ctx (cadr xs)))
        (datum->syntax ctx (list (quote it)))))))
((lambda (it) (refer-it given it)) 13)
"""

SRC_RECURSE = """\
;; 5) 递归宏：countdown 自调用直到 0；预算不足时保留部分展开图
(define-syntax countdown
  (lambda (stx)
    (let ((xs (syntax-e stx)))
      (let ((n (cadr xs)))
        (if (= n 0)
            `done
            `(countdown ,(- n 1)))))))
(countdown 3)
"""

SRC_TEMP = """\
;; 6) 生成临时名：generate-temporaries 产出带 intro scope 的稳定符号
(define-syntax two-temps
  (lambda (stx)
    (let ((ts (generate-temporaries (list 1 2))))
      (let ((a (car ts)) (b (cadr ts)))
        `(lambda (,a ,b) ,a)))))
(two-temps)
"""

SRC_ERROR = """\
;; 7) 错误来源回映：宏用调用点子表达式报错，span 必须指回原始源码
(define-syntax require-positive
  (lambda (stx)
    (let ((xs (syntax-e stx)))
      (let ((e (cadr xs)))
        (if (< (syntax->datum e) 0)
            (error "argument must be positive" e)
            `(quote ok))))))
(require-positive -7)
"""

# A second macro *version* for the comparison view (recursion variant).
SRC_RECURSE_V2 = SRC_RECURSE.replace("(countdown 3)", "(countdown 4)")

CASES: list[dict[str, Any]] = [
    {
        "key": "shadow-local",
        "title": "同名局部变量 / shadowing",
        "scenario": "shadow-local",
        "source": SRC_SHADOW,
        "expect": "template x binds the introduced lambda parameter; "
                  "caller x stays bound to the caller lambda",
        "judgment": "capture-by-scope-relation",
    },
    {
        "key": "nested-macro",
        "title": "嵌套宏 / nested macros",
        "scenario": "nested-macro",
        "source": SRC_NESTED,
        "expect": "outer expands to an inner call which expands again "
                  "(macro call stack depth 2)",
        "judgment": "capture-by-scope-relation",
    },
    {
        "key": "definition-site",
        "title": "definition-site 引用",
        "scenario": "definition-site-reference",
        "source": SRC_DEFSITE,
        "expect": "the template identifier helper resolves at the macro "
                  "definition site, not the call site's helper",
        "judgment": "capture-by-scope-relation",
    },
    {
        "key": "explicit-capture",
        "title": "显式捕获 / datum->syntax",
        "scenario": "explicit-capture",
        "source": SRC_CAPTURE,
        "expect": "introduced symbol it inherits the use-site context's "
                  "scope set; context choice is a visible event",
        "judgment": "datum-context-visible",
    },
    {
        "key": "recursive-macro",
        "title": "递归宏 + 预算",
        "scenario": "recursive-macro",
        "source": SRC_RECURSE,
        "source_v2": SRC_RECURSE_V2,
        "params_v2": {"macro_budget": 2},
        "expect": "with budget 64 it terminates with `done`; with budget 2 "
                  "it is budget-exhausted but the partial graph is retained",
        "judgment": "bounded-recursion",
    },
    {
        "key": "generated-temp",
        "title": "生成临时名",
        "scenario": "generated-temporary-name",
        "source": SRC_TEMP,
        "expect": "two stable generated symbols carry intro scopes and "
                  "appear in the generated-symbol ledger",
        "judgment": "capture-by-scope-relation",
    },
    {
        "key": "error-remap",
        "title": "错误来源回映",
        "scenario": "error-source-remapping",
        "source": SRC_ERROR,
        "expect": "the compile-time error span points at the original -7 "
                  "token inside the macro call",
        "judgment": "bounded-recursion",
    },
]


def build_pack() -> dict[str, Any]:
    return {
        "pack_id": PACK_ID,
        "pack_version": PACK_VERSION,
        "title": "宏作用域镜：语法对象、卫生重命名与来源展开图",
        "coordinate_basis": "span.unit; character offsets with 1-based "
                            "line/column (Unicode code points)",
        "seed": 20260925,
        "cases": CASES,
        "narrative_steps": [
            "导入后先核对原 token、source span、scope set、definition/use "
            "site、宏调用栈、生成符号与转换步骤全部保留",
            "触发同名局部变量 / 嵌套宏 / definition-site / 显式捕获 / "
            "递归宏 / 临时名 / 错误回映边界",
            "第二分支：验证捕获由 scope 关系判定、datum->syntax 上下文"
            "可见、递归有预算且失败保留部分展开图",
            "在联动页面比较两个宏版本，接受/否定/调参都产生新事件",
        ],
    }


CASE_PACK = build_pack()
