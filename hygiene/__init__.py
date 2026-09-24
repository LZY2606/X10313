"""Local-first hygienic macro toolchain (scope-set model).

Modules:
  tokens    - original token stream with source spans (never rewritten)
  syntax    - syntax objects, scope sets, marks, introduction provenance
  parser    - token stream -> syntax objects
  patterns  - syntax-rules pattern matching & template instantiation
  expand    - scope-set hygienic expander, binding edges, partial expansion
  eval_core - small interpreter; runtime errors mapped back to origin tokens
  derive    - versioned derivation, fingerprints, dependency invalidation
  store     - SQLite append-only event store + export/import
  clock     - virtual time / fixed seed
  cases     - offline domain case pack replay
