# PR Conflict Analysis

The pull request that introduces the PDF-driven rule extractor rewrites the same
sections of `app.py` that currently define the static `RULES` dictionary on the
`main` branch. On `main`, the rulebook is still inlined as `RULES` near the top
of the file, so it occupies the same lines that the branch replaces with the new
`DEFAULT_RULES` helper and `get_rules()` session access.

Because both branches touch the same block of code, Git cannot merge them
automatically. Resolving the conflict requires choosing either the static rules
(`RULES`) from `main` or the dynamic configuration loader (`DEFAULT_RULES`
`+ get_rules()`) from the feature branch, then adapting the rest of the file to
use whichever structure remains.

