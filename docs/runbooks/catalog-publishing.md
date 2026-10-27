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

## The daily scan with `--packages` (AP9-KO, release 1.1.0)

The scanner can also check the GGUF packages of the catalog rows and read the model cards of the publishers. It proposes
the results as the fields `gguf_checks` and `setup_note` of a row (stage `A9`); they go through the same approval as every
other change. The contract is in `CONTRACTS.md`, "Package findings and the load check" (V1 to V7).

```
python scripts/catalog_scan.py --catalog modelroom/catalog.json --rejected modelroom/catalog-rejected.json --out <out> --packages --budget 700
```

- **The working catalog has to be schema 3 or 4.** On a schema-2 table the stage does not run and the proposal says so
  (`package checks: the working catalog is schema 2 ...`). Making the table schema 3 is one line, `"schema_version": 2` to `3`,
  and belongs to the first approval of findings (below); until then the daily run goes without `--packages`. **Schema 4
  carries the vLLM switches of a note** (`vllm_args`, release 1.2.0, below); on a schema-3 table the stage runs as before and
  proposes the note without them, with the note `the card names vLLM switches; the working catalog is schema 3 ...: left out`.
- **Hosts.** `huggingface.co`, `api.github.com` (the llama.cpp commit and its file `src/llama-arch.cpp`, two requests a run)
  and `*.hf.co` (the Hub's CDN, to which the read of a file's first bytes is redirected). Only the scanner may reach the last
  two; the runtime of modelroom does not (`http._ALLOWED_HTTPS_HOSTS` is unchanged). Reads only (`GET`).
- **Rotation.** Per run at most 40 rows of `kind` `text` or `vision`, never a row already tried that day: rows never tried
  first, then the one tried longest ago, then by name (`package_attempts` in `scan-state.json`; within a row the repositories
  and packages by `package_unit_attempts`). A second run on the same day takes the next rows and keeps the proposals of the first.
- **Budget.** The package stage spends at most 200 requests a run; the run's own budget (`--budget`, 300 by default) pays for
  the scan of the accounts first, some 130 requests. With the default the stage gets what is left: on 2026-10-14 that was one
  row. Use `--budget 700`. **Every row gets the same share.** After the one-time reads (the llama.cpp commit and file) the
  stage divides what is left by the rows it takes (`floor(rest / rows)`, 4 requests each at the default of 198 over 40 rows)
  and no row may send more than its share, so a row with many conversions (some ninety repositories for one popular model)
  cannot use the run: the run reaches its 40 rows, and a full turn through the text and vision rows takes about a week. A
  row needs at least 4 requests (model answer, card, one page of the list, one file list); when the budget is too small for
  that many rows the run takes fewer and says `budget ended: package checks stopped after n of 40 rows`. What a share cuts is
  no fault: the row counts as tried, the repositories it did not reach are read first next time, and the run says
  `n rows ended at their share`. Inside a row the repository tried longest ago (never tried first) comes first; only among
  repositories of the same day the publisher's own come before those of the catalog's listed packagers, and those before all
  others. A reserve of at least 8 requests (a tenth of the budget when that is more) is kept back from the shares, so that
  it pays the progress of one open unit; when the budget cannot give it next to the minimum of every row, the run takes fewer
  rows. It goes, with what the rows leave, to the cut row that got it longest ago (`package_reserve_days` in
  `scan-state.json`), so every cut row comes round and none keeps it; a list the share cut is continued where it ended, not
  read again. A stage budget below 12 requests gives one row all of it and no reserve. A row books the reserve only when it made progress
  with it. A list longer than the reserve carries (at the default 37 pages, some 1850 repositories) is not reached and ends
  every day the same way without a fault; it is the documented bound (CONTRACTS, V6).
- **Rate limit.** The Hub answers `429` to a client that asks in bursts. A unit that gets one fails alone (`package checks:
  HTTP 429` or `model card not read (HTTP 429)` in the notes), changes nothing it did not read, and is tried again by a later
  run; run the scan once a day, not twice within minutes.
- **A failed run changes nothing it did not read.** No new `absent`, no deletion, and every stored check keeps its day.
- **Reading the proposal.** A `gguf_checks` change lists all checks of the row, `absent` and `present`; only `absent` blocks an
  installation in modelroom. Each check names the repository, the revision, the file, the architecture the file states, where
  that came from (`hub_field` or `file_header`) and the llama.cpp commit it was compared with. A `setup_note` change points at
  the publisher's model card at one revision and names the package only for an explicit inference requirement.
- **Approving.** By number, as every change (`--approve 30,31,...`). The checks and the note of one row are two numbers.
  Removing a finding is a change too and goes the same way.

## Successors from the names (stage `A4`, decided 2026-10-14)

Every scan, with or without `--packages`, also runs the name rule of the search over the rows of the working catalog and
proposes successors (`CONTRACTS.md`, "How a successor is proven in the shipped catalog"). It needs no request.

```
python scripts/catalog_scan.py --catalog modelroom/catalog.json --rejected modelroom/catalog-rejected.json --out <out> --offline
```

- **`--offline`** asks nobody and moves no mark: the run makes only these proposals and carries the open ones. The daily scan
  makes them as well; use `--offline` to look at them on their own, or without a network. Run on a day that already has a
  proposal (a normal scan with the same `--out`), it takes that proposal over like an earlier day's: the open changes and
  candidates stay, only the name-rule successors are made again.
- **Reading a proposal.** Each is a number of its own, stage `A4`, with a line `basis: name rule: version 4.0 stands over
  version 3.5 in <publisher>/<family> (<kind>)` and the successor's Hub page as evidence. Read the pair, not only the
  numbers: the rule compares names, so it cannot see a variation that is no replacement (a LoRA, an editing or audio model
  of the same family, a text embedding for a vision embedding). Approve the numbers you can vouch for:
  `--approve 1,2,5`, as for every change. An approved row becomes `legacy` with that successor in the list.
- **A `latest` row** with a higher version gets no number: a note names it (`latest in the catalog, but ... stands over it by
  version; no successor proposed, decide separately`). Decide it apart, by hand.
- **Same variation.** A proposal needs the same variation words on the row and the successor (in any case), and the successor
  is the highest of that subgroup: a LoRA, an audio, an Omni or a Flash/MOPD build is never proposed for the plain model.
  The size may differ; look at it when you approve.
- **Rejecting** a number turns down the **pair**: `catalog-rejected.json` gets `{"hf_repo": <row>, "successor": <target>, "date":
  ...}` and that proposal never comes back. Another successor of the row (a new higher version) and every other field of the
  row are still proposed. Leave a number you merely doubt open: it is carried with its first day and made again each run while
  the rule holds.
- **Nothing is approved by the rule.** A scan changes no file of the table; only `--apply` with the numbers does. The
  publication then runs as under "A publication"; the new `successor` fields read in releases 1.0.0 and 1.1.0 alike.

### First approval of findings: the table becomes schema 3

The shipped table stays schema 2 and unchanged until findings are approved (decision of 2026-10-14). The steps, in this order:

1. Make the working table schema 3 and keep the scan against exactly that file: replace the one line `"schema_version": 2` by
   `"schema_version": 3` in `modelroom/catalog.json` and nothing else. A proposal is made against the bytes of the table;
   any other bytes make it `base: stale` in `--show`, and the scanner makes it again.
2. Scan with `--packages` against that table, `--show` the proposal, `--apply` the numbers of the findings with the
   checksum, run the tests, commit, then continue as under "A publication".
3. The publication makes **both** channel files from the table (**all three** since release 1.2.0, see below): `catalog-v3.json`
   as it is, and `catalog-v2.json` by projection without the findings (it stays what release 1.0.0 reads). `--publish-pending`
   uploads them in one commit and `--verify` compares both afterwards.
4. **Only then release 1.1.0.** The dataset's `catalog-v3.json` is published before the release, so a user of 1.1.0 meets the
   file at once. A 1.1.0 that meets none chooses between its shipped catalog and a valid cache of the new channel
   (`catalog-v3.cache.json`), says so, and goes on. It does not fall back to `catalog-v2.json` and ignores a newer cache of
   release 1.0.0. Until the file is there, such a user therefore loses the findings **and** the ordinary updates of the
   published table (models, successors, packages) that are newer than the shipped catalog; a valid newer v3 cache is kept.

### The switches of a card: the table becomes schema 4 (release 1.2.0)

Release 1.2.0 puts the switches that the `vllm serve` block of a card names for its model into the vLLM commands: of a closed
list (`--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`, `--kv-cache-dtype`), each with a checked
value, kept in the note as `vllm_args` (`CONTRACTS.md`, V7). A note with `vllm_args` needs schema 4, because release 1.1.0
refuses the key. In this order:

1. Make the working table schema 4: replace the one line `"schema_version": 3` by `"schema_version": 4` in
   `modelroom/catalog.json` and nothing else (a proposal is bound to the bytes of the table, see above).
2. A stored note is **not read again while its card stays at the revision it was read at** (`catalog_scan_packages`, `_card`).
   A note stored before 1.2.0 therefore gets its switches only with the next revision of its card; to have one read at once,
   its note has to leave the table first (a decision of the maintainer).
3. Scan with `--packages`, `--show`, `--apply` the numbers of the notes that now carry `vllm_args`, run the tests, commit.
4. The publication makes **three** channel files from the table: `catalog-v4.json` as it is, `catalog-v3.json` by projection
   without the switches (what release 1.1.0 reads; the note itself stays in it) and `catalog-v2.json` by projection without
   the findings (release 1.0.0). `--publish-pending` uploads them in one commit and `--verify` compares all three.
5. **Only then release 1.2.0.** The dataset's `catalog-v4.json` is published before the release, so a user of 1.2.0 meets the
   file at once. A 1.2.0 that meets none chooses between its shipped catalog and a valid cache of the new channel
   (`catalog-v4.cache.json`), says so, and goes on; it does not fall back to `catalog-v3.json`. Releases 1.0.0 and 1.1.0
   keep reading their own files and never meet `catalog-v4.json`.

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
   **Rows without a reading (AP9-KA-H).** A new row the grammar cannot read (section `Rows without a reading` of the
   proposal, computed from repo and size, not from a mark) is taken by `--apply` only with `--approve-unread <n>`, where
   `<n>` is also in `--approve` and named once; the number is then stored as `approved_unread` in `decision.json` (a
   decision without the field counts as an empty list). The exception entry in `SHIPPED_WITHOUT_READING`
   (`tests/test_search_release.py`) and the `--apply` belong in the same commit. The way through the maintainer's
   wrapper (`katalog-freigabe.ps1`, kept outside this repository) is its switch `-OhneLesart <n>`, used with
   `-Freigeben`: every number in it must also be in `-Zeilen`, none twice, and the wrapper passes
   `--approve <list> --approve-unread <list>`; `-Ohne` only notes a row left open in the journal line and is no way to
   take a row without a reading. Candidates open for more than 30 days
   are listed first under `## Candidates past 30 days (decide now)`.
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
   The new row carries the note `Kind <kind> set by the maintainer on <YYYY-MM-DD>.` (one of the eight kinds, nothing
   before or after). That form is product, not a schema-1 reason: `scripts/catalog_publish_rules.py` holds it as
   `MAINTAINER_KIND_NOTE`, and `test_every_reasoned_missing_statement_of_the_schema_1_file_is_a_note` leaves exactly
   that form out of its comparison; any other wording of a note stays in the comparison and has to be in `NOTES`.
4. **Test, check, commit.** Run the test suite and the release gate's hygiene check, then commit `modelroom/catalog.json`
   and `modelroom/catalog-rejected.json`. The maintainer's release wrapper runs the whole
   suite before the commit (about 8 minutes, the same stage for `-Freigeben` and `-Fortsetzen`); a red test stops the chain
   before the commit with the state `applied`, and after the fix it goes on with `-Fortsetzen`.
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
