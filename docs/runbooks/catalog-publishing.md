---
typ: runbook
---

# Publishing the catalog

How a proposal of the catalog scanner becomes the published table: set-up once, the setup test, the steps of a
publication, and what to do after each interruption. The tool is `scripts/catalog_publish.py` (a maintainer tool, not in
the wheel); the scanner that makes the proposal is `scripts/catalog_scan.py`. Every call below is run from the root of a
clone of the repository. Verified against the recorded answers of the test suite; the first run against the real dataset is
the setup test below.

## What the tool does and does not do

- It changes the table, the rejection list and the dataset only for the rows, accounts and rejections the maintainer
  names in the call. There is no "all", no default list, no prompt and no `--yes`.
- `--show`, `--apply`, `--complete`, `--materialize` and `--record-commit` make no request and need no token.
- The daily scan never calls `--apply`, `--materialize`, `--record-commit` or `--publish-pending`, and is never given
  the token path. Only `--verify` (no token) is part of it.
- The upload carries the bytes of the recorded commit, never the file in the working tree.
- A publication runs through one state in `decision.json` in the folder of its proposal:
  `approved`, `applied`, `committed`, `pushed`, `uploaded`, `published`. One publication at a time: while one is not
  `published`, `--apply` refuses every other proposal.

## Set-up (once, by hand)

1. Sign in to the hosting service with an own password and a second factor, not with a login of another provider.
2. Create two **public** datasets under the maintainer account: `jm-vis/modelroom-catalog` (the table) and
   `jm-vis/modelroom-catalog-probe` (the setup test; it only ever holds test data from `tests/fixtures/`).
3. Create an access token. Prefer a fine-grained token with write access to those two datasets only, if the service offers
   that for a personal account; note the result here. Otherwise a write token of the account.
4. Store the token in a file **outside every repository**, readable by you alone. The tool never reads or prints its
   content; it only passes its path on.
5. Keep a second file outside every repository with the maintainer's private words, one per line (the release gate's
   list). The upload refuses without it.
6. Export the two variables in the shell that publishes, never in a shell of the daily scan:
   `HF_TOKEN_PATH` (the token file) and `MODELROOM_RELEASE_PRIVATE_WORDS` (the words file). `HF_TOKEN` must stay unset;
   the tool removes it from the environment of the upload in any case.
7. Make sure `uvx` is available. The upload runs `uvx --from huggingface_hub==<version> hf upload ...` with the version
   pinned in `scripts/catalog_publish_rules.py` (`HF_HUB_VERSION`); the package has no dependency on it.
8. Chrome for the card's pictures: set `MODELROOM_CHROME` to the browser's executable, or leave it unset for the standard
   path under Program Files. Without a Chrome the card is only checked (`--check`), not rendered.

## Setup test (once, and again after a change of the pinned version or the token)

```
python scripts/catalog_publish.py --upload --probe tests/fixtures/catalog_v2_valid.json
python scripts/catalog_publish.py --verify --probe tests/fixtures/catalog_v2_valid.json
```

The first call refuses unless the file lies under `tests/fixtures/`, is tracked by git and `git status` shows it unchanged;
it uploads to the probe dataset only. The second runs **without** the token variables, reads the probe dataset the way modelroom does
(revision of the main line, then the file at that commit) and compares byte for byte. Expect
`catalog: published file equals the probe file`. A call of the first without `HF_TOKEN_PATH` must stop before any process
starts: try it once. Write the date and the result of both calls (and whether the token could be limited to the two
datasets) into the entry that records the set-up.

### Set-up record

- **2026-10-07.** Both datasets exist and are public. The token is fine-grained and limited to the two datasets
  (write access to their contents only; the service offers that for a personal account). Probe upload with
  `huggingface_hub==1.28.0`: commit `132b0b9566ae0b7d75e4fa166dd20d1e72531ba9` on the probe dataset; `--verify --probe`
  without the token variables: `catalog: published file equals the probe file`, exit 0. The upload without
  `HF_TOKEN_PATH` stopped with exit 2 and `HF_TOKEN_PATH is not set or names no readable file` before any process. Note
  for the caller: the hygiene check runs before the probe upload as well, so `MODELROOM_RELEASE_PRIVATE_WORDS` must be
  set for it too; without it the call ends with exit 2 and nothing is uploaded.

## A publication

1. **See the proposal.** `python scripts/catalog_publish.py --show <out>` names every pending publication first (day, state,
   commit), then prints the newest open proposal with every field as `field: old -> new`, whole, the full checksum of the
   file it was made from, and whether the proposal still fits the table (`base: current` or `base: stale`; a stale proposal
   is made again by the scanner).
2. **Decide, then name.** Only a number or candidate the maintainer names is touched:
   `python scripts/catalog_publish.py --apply <out>/<day>/proposal.json --proposal-sha <checksum> --approve 1,2,5 --reject 4 --admit K1`.
   A refusal ends with exit 2 and nothing written: wrong or missing checksum, another table than the one the proposal was
   made against, an unknown number, a number in two lists, a candidate in `--approve`, a repo of open kind in `--admit`, a
   decision that exists already, another publication pending. A rejected number puts its repo on the rejection list, a
   rejected candidate its account (a candidate of open kind: its repo). An admitted publisher is entered with its account
   as name and no region: fill in both by hand in a later change.
   **Deciding on candidates (rule of 2026-10-08).** The catalog is the public list of sources the maintainer trusts;
   admitting an account says "we watch this maker, its models appear by themselves". A name that is not in the catalog
   costs the user nothing: a typed name is never hidden and modelroom measures every model. So the risk sits on the
   side of admitting, not of rejecting; a rejected trending account comes back as a candidate after 30 days if the
   trend holds.
   - **Admit** an account with its own architecture or product line, several models or versions, at least 1000 likes
     or an established history, an open license, and a region that can be named.
   - **Reject** single persons, one-model accounts younger than 30 days, packagers (B2: only fine-tunes or
     quantisations of other makers' base models), and parts without a catalog kind (B3: VAEs, CLIP variants, bert
     implementations, rerankers).
   - **In doubt, reject**; the 30-day resubmission decides.
   What admitting risks: the reputation of the list (thin accounts dilute it), `custom_code` repos that the list then
   implicitly recommends, licenses such as `other` that mislead commercial users, the daily request budget (every
   watched account eats part of it), and a region that cannot be named (the publisher record then needs a later
   change by hand).
3. **A candidate of open kind** becomes a row only through the kind the maintainer names:
   `... --complete <proposal.json> --proposal-sha <checksum> --admit K3 --kind vision`. It rewrites `proposal.json` and
   `proposal.md` with new numbers and prints the new checksum; show it again and apply with the new checksum.
4. **Test, check, commit.** Run the test suite and the release gate's hygiene check, then commit `modelroom/catalog.json`
   and `modelroom/catalog-rejected.json`.
5. **Record the commit.** `python scripts/catalog_publish.py --record-commit <out>/<day> <full commit hash>` (state
   `committed`; it refuses a hash that is no commit, or a commit whose `catalog.json` or `catalog-rejected.json` is not
   the result of the decision; the same commit recorded again is confirmed without a write).
6. **Publish.** `python scripts/catalog_publish.py --publish-pending <out>/<day> [--remote forgejo]` pushes that commit to
   the remote (`git push <remote> <commit>:refs/heads/main`, no force), uploads its `catalog.json` bytes after the hygiene
   check, then compares the published file with the commit (`published`). Each step is repeatable.
6a. **The card follows the catalog.** `python scripts/catalog_card.py --write --render --upload` brings the numbers of
   `docs/dataset-card.md` and of the two pictures to the catalog (models, accounts, size in KB), renders the PNGs, and uploads
   the card and the banner to the dataset in one process (same token path and private words as step 6). Then commit and push
   `docs/dataset-card.md`, `docs/assets/catalog-*.svg` and `docs/assets/catalog-*.png`. The script never commits.
7. **Check later.** `python scripts/catalog_publish.py --verify [--commit <hash>]` compares the published file with the
   working tree or a commit and needs no token. A difference prints a line starting `ALARM` and exits 1.
   `python scripts/catalog_card.py --check` does the same for the card and the pictures; the daily run carries it.

## Resuming

Never run `--apply` again for a decision that exists. Read the state with `--show <out>` and continue from there.

| State | Meaning | Continue with |
|---|---|---|
| `approved` | the decision is fixed, the files may not be written yet | `--materialize <out>/<day>`, then step 4 |
| `applied` | both files hold the result, no commit yet | step 4 (a differing `catalog.json` means somebody changed it: stop and look), then step 5 |
| `committed` | commit recorded, not pushed | `--publish-pending` (pushes the same commit) |
| `pushed` | pushed, not uploaded, or the upload failed | `--publish-pending` (uploads the same bytes again) |
| `uploaded` | uploaded, the check failed or did not run | `--publish-pending` (checks only; it does not upload again) |
| `published` | done | nothing |

`--materialize` writes a file only when it still holds the base; one that holds neither the base nor the result is
somebody else's change, and the tool stops with exit 2 and the state `approved`. Exit 1 always means a process or a request
failed and the state stayed where it was; exit 2 means the call or an input was wrong and nothing was written.
