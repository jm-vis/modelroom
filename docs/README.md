---
typ: moc
---

# docs

The repository's own record, one folder per genre. A folder appears with its first real entry
and carries a `README.md` index with one line per entry. Nothing here is a template.

```
idea -> decision -> execution -> measurement -> operation -> incident
docs/rfc/  docs/adr/  docs/plans/  docs/slo/  docs/runbooks/  docs/postmortems/
                                               docs/playbooks/
```

| Folder | Appears when | Must contain |
|---|---|---|
| `rfc/` | an idea should be discussed before it binds | proposal, open questions |
| `adr/` | a decision whose reversal would be expensive | status, context, decision, consequences, alternatives and why not |
| `plans/` | work that spans more than one session | steps with status markers |
| `slo/` | the first number whose breach triggers action | target, window, reaction |
| `runbooks/` | a procedure is needed a second time | exact commands, verified on the system |
| `playbooks/` | a class of situations spanning two or more runbooks | order of runbooks |
| `postmortems/` | an SLO breach, data loss or outage over one hour | why it was not caught earlier |

An ADR is immutable; a changed situation gets a new ADR that supersedes the old one.

Entries so far: [`adr/`](adr/README.md). `assets/` holds the repository images (banner, social
preview) and is not a genre.
