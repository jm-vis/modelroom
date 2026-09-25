# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow Semver.

## [0.1.0] - 2026-09-25

The first release on the package index. The guided mode is the way in; the six subcommands
under it are what it calls. The entries under "The first cut" at the end of this section were
written on 2026-09-23 for the same version, which was never published. Entries stand in the
order they landed; where two of them name a number (a request budget, pages per account), the
later one is what ships.

### Added

- **The search takes whatever someone types** (`modelroom/search_word.py`, new;
  `modelroom/search_pages.py`, `modelroom/search.py`, `modelroom/guided_search.py`, new;
  CONTRACTS.md, "What was typed, read as one of three things"). Until now every answer went to the
  Hub as `search=<text>` word for word, so `unsloth/Qwen3.5-9B-GGUF` found nothing, `qwen 3.5 9b`
  found nothing, and an empty answer was the error `no model name to search for`. The input is
  parsed first, into one of three forms:
  - **a repository id** (`owner/name`): one request for that repository, shown as `typed`. It is
    always selectable -- the owner filter does not hide it and `publisher_unknown` does not apply,
    because someone who names a repository has said which account they want. Without a GGUF file
    (read off the repository's own `gguf` tag) the run says `<id> holds no GGUF file` and searches
    for the words of the name half instead, so the input is never simply refused;
  - **words**: the text cut at blanks, `-`, `_` and `:`, without the filler words `gguf`, `model`,
    `models` and `the`. One of them goes to the Hub (the longest letter run) and the rest narrow the
    answer locally, so `qwen 3.5 9b` and `Qwen3.5-9B` both find `unsloth/Qwen3.5-9B-GGUF`. A
    one-word search is never narrowed -- that is the search this package has always made;
  - **nothing at all**: a valid answer that means "show me what fits this machine". One page per
    model the catalog calls `latest`, its ten most downloaded GGUF builds, with the note `no search
    word: the catalog's current models, one page each`.
- **A word close to something the catalog knows leads to a list, not to an empty step**
  (`search_word.close_matches`, `guided_search.candidate_choices`; the new answer-file key
  `did_you_mean`). `qwn` offers `qwen`, picked from the catalog's family names, publisher accounts,
  model names and the letter runs of those names at `difflib` cutoff 0.75; picking one searches
  again, `none of these` keeps the answer as it was.
- **Two pages per account: its newest twenty, and its twenty most downloaded** (CONTRACTS.md,
  "Search over the Hugging Face API"). Measured live 2026-09-25:
  `author=unsloth&search=qwen&sort=createdAt&limit=20` answers with twenty repositories created
  after 2026-05 and `unsloth/Qwen3.5-9B-GGUF` (2026-02-28) is not among them at all, while the same
  account's `sort=downloads` page carries it in fourth place -- one sort order cannot find the plain
  build of a model whose account publishes often, whatever word is typed. The two answers stay
  **one** group under the account's label, and a group says how much of its answer an earlier page
  already had and how much the words of a multi-word input do not name.
- **`tests/search_testset.toml`**, and criterion 8 of the acceptance run
  (`scripts/searchset.py`, new): eight lines of "what someone types" and "which models must come
  back for it", asked **live** and reported case by case. Gate-free by decision -- the Hub's answer
  for a word changes by the hour, so a missing model is a `WARN` and never ends the run; what the
  tests gate is the shape of that report.
- **`Enter` takes the row under the pointer** in every list of the guided mode that marks
  (`modelroom/dialog.py`; CONTRACTS.md, "Guided mode"). `Enter` without the space bar took nothing
  at all, and a reader read the green bar as the selection ("I thought the bar was the selection",
  test round 2026-09-25). With something marked the answer is exactly what is marked. The
  instruction line says both cases -- `↑↓ move   Space marks   Enter takes the marked rows, or this
  one   Esc leave` -- and the rule is one key binding added after `questionary` has built the
  question, like `Esc`: the library is unchanged, and an answer file's `select = []` still means
  nothing.
- **The list of step 2 has a column head, a `Release` column and gray `legacy` rows**
  (`modelroom/guided_models.py`; CONTRACTS.md, "Guided mode", step 2). It had no head, so a reader
  had to guess what a cell meant, `latest` was invisible between two columns of numbers and `none
  known` read as a statement about the model (test round, 2026-09-25). The head is the first line of
  the list and no entry of it, the `Release` column carries `latest`, `legacy` or `–`, a `legacy` row
  is drawn gray as a whole, and the order puts `latest` above a release nobody knows and that above
  `legacy`. The 100 characters now cover the **whole** line, pointer and marker included.
- **A line of a list wraps instead of being cut** (`modelroom/dialog.py`). Measured 2026-09-25 in a
  window of 100 columns: the library's own list window does not wrap, and the last piece of the
  153-character instruction line of step 2 was simply gone.
- **`aliases`: the other tags of one Ollama manifest** (`modelroom/contracts.py`,
  `modelroom/ollama.py`; CONTRACTS.md, "Package"). `qwen3.5:9b` and `qwen3.5:9b-q4_K_M` are two
  names of one manifest -- the same digest, byte for byte -- and stood in the ranking as two
  packages of one size. There is one package per manifest digest now, named by the shortest of its
  tags, with the rest as `aliases`. The field is optional and empty by default, so a snapshot
  written before it reads unchanged and `schema_version` stays at 1. An approval given to a tag that
  is an alias now carries forward: it is about the same manifest under the same digest.
- **A step 5 with nothing to rank says so and draws its card** (`modelroom/views.py::show_nothing`;
  CONTRACTS.md, "Step 5, the render and the card"). The last step of a run that chose nothing was a
  head with nothing under it, and the reason stood on stderr where no reader of the run is (test
  round, 2026-09-25). It now notes `nothing to rank: no package in this folder yet`, draws the card
  with `result –`, and closes with `– Results         nothing to rank`; the exit code is unchanged.
  Only where the folder really holds no snapshot -- a render another process's lock stopped is no
  folder without packages, and step 5 then says nothing of its own.
- **The catalog names the publishers people actually search for** (`modelroom/catalog.toml`,
  README, "Publishers the search resolves"). It knew three accounts (`Qwen`, `deepseek-ai`,
  `utter-project`) and is at the same time the positive list of publishers, so a search for
  `mistral` in the test round of 2026-09-24 answered with repositories of which not one could be
  picked: every hit was `publisher_unknown`, because `mistralai` was no publisher of the catalog.
  It now holds 29 families under 22 publisher accounts, Europe first (`mistralai`,
  `utter-project`, `openGPT-X`), and per family the lines the publisher carries today in the sizes
  a single machine runs. Every row names the model page it was read from on 2026-09-25 and has at
  least one GGUF build of a listed packager or of the publisher itself; an Ollama name is only
  written when the manifest of that exact size answers 200, and `latest` only with the publisher
  page or collection that calls the model current plus the day it was checked -- 22 of the 62 rows
  say so, the rest stay `unknown`. Two lines (DeepSeek V3.2/V4, GLM-5.3-Flash) are far above those
  sizes and are listed anyway, so that a search resolves them and the computed fit answers instead
  of the list hiding them.
- **One drawer for the screen of the guided mode** (`modelroom/screen.py`, new; CONTRACTS.md,
  "Guided mode", "The screen"). Every line of a run is now a pattern -- a step head of 72
  characters, an answer line, a note, a balance with a check mark or a dash, the card of step 5,
  and `joined`, which puts short statements on one line and wraps between them at 100 characters.
  Each pattern is a list of `(class, text)` fragments, printed **with color** through
  `prompt_toolkit` and the one style of the dialog, or **without color** as exactly the same plain
  text through the run's `out`, with one leading space, as the start screen has it. So
  `modelroom --answers <file>` and a terminal run read as the same screen, and
  `tests/golden/guided-screen*.txt` is that screen as a fixture.
- **`<state>/search.json`**, the log of one search (`modelroom/state.py::write_search_log`,
  `search.SearchOutcome.search_log`; CONTRACTS.md, "Search log"). It names every page the search
  asked with the class it was asked by (`publisher`, `packager`, `open`), what it answered and
  whether it was full, plus the request count, the budget, how much resolved and every reason a
  repository is no model of the list. The screen keeps two sentences of it; the file answers "why
  did this account answer nothing", which is a question about the search and not about the line a
  reader is looking at.
- **A line to copy for the first package of this machine**, under the result table and behind a
  blank one: `install   #1  ollama pull hf.co/<repo>:<quant>` (`screen.install_name`,
  `views.terminal_blocks`). Nothing is called, nothing is looked up, and a package with no local
  Ollama name has no line. Installing is a work package of its own.
- **`modelroom hardware --same-machine`**, and "the same machine" in the guided mode's clone
  question, now **measure** (`modelroom/binding.py`, `modelroom/cli.py`, `modelroom/guided.py`;
  CONTRACTS.md, "Profile binding"). Until now the honest answer to "is this the same machine or a
  clone?" for a results folder whose profile file is gone was a dead end: the run said "put that
  profile file back", measured nothing, and every step after it was empty -- no fit in the size
  scale, no measurement, "nothing to render" at the end. The takeover rule gained the answer
  `rewrite`: the machine is measured again under the profile id the folder already binds it to,
  the binding stays that id, and `[machines.<name>].profile` is written as after a new profile.
  The two switches are two answers to one question: together they are exit `2` with a sentence
  naming both.
- **A fit from the size of a package**, for an architecture fit v1 cannot read
  (`modelroom/fit.py::fit_from_size` and `fit_from_parameters`, `Fit.basis`; CONTRACTS.md, "Fit
  from size (basis `size`)"). A hybrid or mixture architecture -- Qwen3.5, Qwen3.6, Qwen3.8,
  DeepSeek-R1 in its Qwen build -- used to leave every one of its packages under "architecture not
  covered by v1": 25 of 26 packages in a hand test of 2026-09-24, with a single package ranked.
  Such a package is now computed with the same formula and a KV cache taken from a constant
  (`KV_PER_TOKEN_SIZE_BASIS`, 147,456 bytes per token: 2 x 36 layers x 8 KV heads x 128 wide x 2
  bytes), capped at `good`, and every view says `from size` next to the class. That constant is one
  fixed assumption and no bound in either direction -- measured on a 5.29 GiB `Q4_K_M` package at
  131072 tokens it computes 24.32 GiB, against 22.32 for a 32-layer model and 27.32 for a 42-layer
  one -- which is what the cap at `good` and the words `from size` are for.
  `fit_from_parameters` answers the same question before anything is fetched, from a parameter
  count and `BYTES_PER_PARAMETER_Q4` (0.6, measured against two real packages of the fixtures).

### Fixed

- **A set-aside piece no longer makes a line of 101 characters** (`views._named`): it counted two of
  the three characters it adds -- the space in front and both brackets -- so a name list of just the
  right length passed the room check and the line came to 101 (second-model round, 2026-09-25).

### Changed

- **The owner filter is a preference, not a positive list** (`modelroom/search.py`,
  `modelroom/guided_models.py`; CONTRACTS.md, "Search over the Hugging Face API", README,
  "Publishers the search resolves"). With the filter on -- the default -- nothing changes: a base
  model whose account the catalog does not name stays unresolved with `publisher_unknown`. Say no to
  the filter and it resolves like any other, with its age `unknown` for want of evidence and its
  owner class `other`; the reason `publisher_unknown` then appears nowhere. The open `sort=downloads`
  page is twenty repositories instead of ten, because with the filter off that page is the only place
  a model of an account nobody listed shows up at all.
- **`search.json` is schema 2** (`modelroom/search.py`; CONTRACTS.md, "Search log"), for one new
  field: `mode` says whether the search was a word search, a typed repository id or the catalog pages
  of a search with no word. `requests` is the number of **pages** and is therefore no longer the
  length of `accounts`, which stays one entry per account.
- **Where a publisher of the catalog was asked and answered with nothing, the note names it**
  (`modelroom/guided_search.py`): `searched Hugging Face at the five listed packagers; the publisher
  deepseek-ai has no GGUF repository` in place of "at no publisher of this catalog that answered",
  which was true and told a reader nothing they could act on.
- **The search of step 2 lives beside the step** (`modelroom/guided_search.py`,
  `modelroom/search_age.py`, `modelroom/search_apply.py`, all new and all moved unchanged): the two
  search questions and the notes out of `guided.py`, the age verdict and the write-back into the
  configuration out of `search.py`. `search.py` re-exports `AgeVerdict`, `decide_age`, `apply_hits`,
  `family_name_for` and `write_configuration`, so no caller and no import changes.
- **The guided mode reads as a report, not as a record** (`modelroom/screen.py`,
  `modelroom/dialog.py`, `modelroom/guided*.py`, `modelroom/views.py`; CONTRACTS.md, "Guided mode").
  The test round of 2026-09-24 called the run "not laid out, hard to take in, one thing chained to
  the next, repetitions like the one about the context". Each step is a block now: a head of its
  own, one line per question with the answer **in words** (`this machine`, `Qwen3.5-9B, Qwen3-0.6B`,
  `L  32k  24,000 words  a report or a long contract`), at most two notes about what the run made of
  it, and one balance line -- a check mark, or a dash where the step did nothing. The dialog library
  writes no answer of its own any more: a list erases itself once it is answered
  (`erase_when_done`), so `done (2 selections)` and `[this machine (measure now)]` are gone. What a
  list has to say about itself moved into its instruction line: the keys behind the question, the
  glossary of a column as a gray line under the list, both gone with it (2026-09-25 -- the list
  window scrolls around the pointer, so the keys may not stand at its end). The
  run closes with a **card** of six rows -- folder, machine, models, context, speed, result -- over
  the result table. Gone from the screen, and still in the files they belong to: the profile's
  readings and its id, `wrote … as the writer`, the seven account lines with the budget, the reasons
  per repository, `fit at 8k context`, `checking …`, `The last column is computed …`, `kept context
  32768 in …`, the names of the cloud models, ten notes that said the same two sentences, `unknown`
  in every row, and `Written to … and …`.
- **Every question of the dialog erases itself**, a typed answer as well as a list
  (`modelroom/dialog.py`). Measured in the second-model round of 2026-09-24: `? What are you looking
  for? qwen` stayed on screen and the run wrote the same question again under it.
- **The result table says the memory pool where it said the basis** (`modelroom/views.py`). `Fit` is
  `good (RAM)` where the pool is system memory -- the same word the selection list of step 2 uses --
  and `Speed` is `–` where nothing was measured, not `unknown`. Under the table the notes are
  bundled, and since 2026-09-25 **every line carries the label of what it says**, in the card's own
  column: `shown     10 of 22 packages · 2 too tight (…)`, `memory    #1–4 fit into graphics memory,
  11.0 GB free after the reserve` with one line per pool, `basis     #1–10 computed from the package
  size, not its architecture`, and `speed     #1 measured 41.1 tok/s` or `nothing measured in the rows shown · say
  Yes in step 4 to measure an installed package`. Running text buried the install command a reader
  came for (test round, 2026-09-25). The head of each block is `<machine> · context L 32k` in place
  of `Ranking: <key> (<label>)` and the scenario line. The Markdown and JSON views are unchanged to
  the byte.
- **The fit of the selection list is computed at the context the scale starts on**
  (`guided_models.DEFAULT_CONTEXT`, 32768, one constant with `guided_context.DEFAULT_CONTEXT`). The
  list said `fit at 8k context` while the very next question started on `L 32k`; two numbers for one
  run are one too many.
- **The mark of the start screen is mark E in color** (`modelroom/intro.py`): squares of 24 pixels
  with a gap of 8 in both directions, a row of lower half blocks over a row of full blocks, six
  lines with the name, the description and the version on lines two to four. "Four columns, not a
  lot of little squares like in the banner" was the verdict on the old three rows of touching
  blocks. Without color the mark stays the three rows of `██ ▓▓ ░░` it was: a log is no place for a
  brand surface, and a half block cannot be shaded.
- **Step 2 of the guided mode offers models, not repositories**
  (`modelroom/guided_models.py`, new; CONTRACTS.md, "Guided mode", step 2, and "ModelChoice").
  The list of the hand test of 2026-09-24 had 120 lines of over 200 characters, repository ids,
  `unknown` in the size and age columns throughout and 40 grayed-out rows whose appended reason
  broke the columns -- and no statement at all about what a choice would cost. The list now holds
  one line per resolved base model and only the ones that can be picked: name, the fit its size
  allows on this machine (`good (RAM)` where the graphics memory is too small for it and the fit
  is against system memory), the size, the packager accounts that have it, their downloads
  together and the Ollama name, at most 100 characters -- every cell cut to its column, because
  `dialog.columns` pads and never cuts and these names come from a registry. A fit in graphics
  memory stands above one in system memory whatever its class, then the class, then the
  downloads: live, a 122B model had stood as `good` above a 9B `marginal` that fits the graphics
  card. Above it stand the two counts and the reasons the
  other repositories are no model; under it the context the fit was computed for and one line on
  what marking does. A chosen model is fetched from two repositories -- the publisher's own and
  the first listed packager's, of an account's builds the plain `<name>-GGUF` one -- because every
  repository costs the fetch about ten requests of the shared budget. `select` takes base model ids, and an answer file may name repository ids as
  it always could.
- **The result view of the terminal reads as the mockup does** (`modelroom/views.py`): `#`,
  `Model`, `Package` (`packager · quant`, cut with `…`), `Fit` (20 characters, with ` (from size)`
  where that is the basis), `Speed` and `Memory`, 100 characters wide, with the scenario as one
  short line under the step head. The snapshot time and the ranking rule are in the Markdown file,
  which is unchanged.
- **The mark of the start screen is drawn in one glyph and three colors** (`modelroom/intro.py`).
  Windows Terminal draws `▓▓` and `░░` as a coarse dot raster; where there is color, every cell is
  a full block now and its own color tells the three apart. Without color -- no terminal,
  `NO_COLOR`, `--answers`, or a console that cannot encode the block -- the shading glyphs stay
  exactly as they were.

- **The search asks one request per account, and the open lists only without the filter**
  (`modelroom/search_pages.py`, new; CONTRACTS.md, "Search over the Hugging Face API"). Until now
  the search made one request for the 50 most recently created GGUF repositories matching the word,
  and the positive list of packager accounts only *classified* whatever that answer happened to
  contain. Measured in an empty folder with `qwen` and with `deepseek`: the answer was 50
  third-party uploads of the last few days and not one repository of `unsloth`, `bartowski` or
  `Qwen` -- those accounts are older and fall out of the window, so nothing was selectable, twice.
  Now every publisher account of a matching catalog family and every account of the positive list
  gets a request of its own (`author=<account>`, `limit=20`), and age plays no part in it. Saying no
  to the owner filter adds two open lists on top of those groups -- the ten most downloaded and the
  ten newest repositories for that word (`limit=10` each) -- so a fresh fine-tune under an account
  nobody listed is visible, with its reason, instead of missing. `run_search` takes
  `open_pages` (default `False`); `HF_SEARCH_LIMIT` is gone, `HF_ACCOUNT_LIMIT` (20) and
  `HF_OPEN_LIMIT` (10) replace it. `SearchOutcome` now carries `groups` and prints one line per
  group above the summary; `hits` stays the same flat list in group order, so the selection list and
  the answer file's `select` key are unchanged.
- **The selection list shows the downloads** as its eighth fact, behind the age: `12.0M`, `68k`,
  `3551`, or `unknown` when the Hub answers no count (`SearchHit.downloads`, `expand=downloads`).
  It is an indication of what many people take and **no rank** -- the ranking rule alone orders the
  result.
- **The shared request budget of a guided run is 150, not 60** (`DEFAULT_GUIDED_BUDGET`). Measured
  live, `qwen` with the owner filter on spends 47 requests before the selection list is shown --
  seven account pages and one age lookup per distinct resolved base model -- and 60 left the fetch
  of two chosen repositories `incomplete (budget exhausted)`. A plain `modelroom fetch` keeps 400.

### Added

- The guided mode as a dialog for people who do not build this package (CONTRACTS.md, "Guided
  mode"). **A start screen** (`modelroom/intro.py`): the mark's pictogram, the version with the
  repository from the package metadata, and the three facts that are known before the first
  question -- the results folder, the local Ollama daemon with its version and its number of
  models, and this machine with its hardware in plain words. **Five numbered steps**, each with a
  head (`Step 3 of 5  Context`) and one line left behind when it is done. The fetch moved into
  step 2, ahead of the context question, so that question can count what was really fetched; an
  answer file is a table of keys and does not notice.
- **The context as a size scale** (`modelroom/guided_context.py`): six levels from `XS` 4096 to
  `XXL` 131072, each with the context, roughly how many words that is, an example, and how many of
  this folder's packages still fit the machine being checked. That last column is fit contract v1
  itself (`fit.count_fitting`, unchanged formula, one request, 16-bit KV cache) with the ranking
  rule's own "it fits" predicate, so the scale and the result table of one run cannot disagree. A
  level above the smallest `Architecture.max_context` of the configured base models is grayed out
  with that reason. The pointer starts on the context this folder kept, else on `L`; a kept context
  that is no level gets a `custom` line, and `enter a number` still leads to a number of tokens. An
  answer file answers `context` with a level (`"L"`) or a number as before.
- **One style for every question** (`modelroom/dialog.py`): a green pointer, the grayed-out entries
  with their reason, labels aligned in columns, and one instruction line under each list. Color
  only at a terminal, only without `NO_COLOR` and never under `--answers` (a run that answers from
  a file is read from a log); the drawn glyphs only where the console can encode them, else ASCII.
  `Esc` leaves a selection list the way an end of input does.
- `tests/conftest.py`: one fixture for the whole suite that moves `HOME` and `USERPROFILE` into the
  test's own folder, so a call that forgets to pass a path cannot reach a real home folder.
- `[guided].context`: the context a guided run's ranking was computed for is kept in
  `modelroom.toml` (CONTRACTS.md, "GuidedConfig"). The context question starts at the kept value
  instead of always at 8192, and writes the answer back whenever it differs from what the file
  holds by then (the file is read again first) -- the first time as well, and for 8192 as well,
  because a kept context is a decision and not a default. A
  `modelroom render --config <toml>` without a scenario of its own now computes with that context
  (`render_cmd.scenario_from_config`), so it shows the ranking the guided run showed and leaves a
  measurement taken at that context in measured group 0; before, it fell back to 8192 and a
  measurement of another context dropped into group 1. The field is optional, the configuration's
  `schema_version` stays `2`, and a configuration without it reads and renders exactly as before.
- The load test, stage 1 (CONTRACTS.md, "Load test (stage 1)"): the guided mode offers to measure
  the speed of packages the local Ollama daemon already has, runs measurement protocol v1 against
  them and writes each result as its own measurement file, which the same run's ranking then shows
  in measurement group 0 with `Speed tok/s (measured)`. Stage 1 downloads nothing, removes nothing
  and says nothing about disk space: there is no call of `/api/pull` or `/api/delete` anywhere in
  the package, and a test reads every module to prove it.
- `modelroom/daemon.py`: the transport for the Ollama daemon of this machine,
  `(method, path, body) -> Response`, against `http://127.0.0.1:11434` and nothing else. Its
  opener carries no proxy handler, no HTTPS handler and no redirect following, and it makes only
  `GET /api/version`, `GET /api/tags`, `GET /api/ps`, `POST /api/show` and `POST /api/generate`;
  10 s per short call, 600 s for one `generate`. A refused connection, a limit that was reached or
  an unreadable answer is a message, never a traceback. What answers on that port is taken to be
  this machine's daemon -- a documented limit.
- `modelroom/loadtest.py`: the inventory, the run and the file. An installed model counts as a
  package only when a digest says so -- the manifest digest for an Ollama package, the weight
  digest the daemon shows in `POST /api/show` for a Hugging Face one; a matching name without one
  is listed with its reason and never measured. `/api/ps` is read after every run, the warm-up
  included, and a name, digest or `context_length` that deviates at any observation makes the
  whole measurement `not comparable` with its reason -- stored, never ranked.
- `modelroom/guided_loadtest.py`: the guided mode's load test step around those two -- the two questions,
  the selection list of candidates, the progress lines and the one result line per measurement.
- Two answer-file keys for the load test, `load_test` and `load_test_packages` (CONTRACTS.md,
  "Guided mode"), and criteria (5) and (6) of `scripts/selftest.py` are now real: the prepared
  package is measured against the daemon of this machine, and the second start keeps that
  measurement in group 0 without writing a second file.
- `modelroom` with no subcommand is the guided mode (CONTRACTS.md, "Guided mode"): it asks where
  results should live, migrates and writes `modelroom.toml` with this device as the writer,
  measures this machine or imports a profile, searches Hugging Face, takes the selection and the
  context, fetches and renders -- calling exactly the functions the three commands call, with no
  second way of computing anything. Without an interactive terminal it prints the help and exits
  `2`; `modelroom --answers <file>` takes the dialog's answers from TOML instead (for a self-test
  or CI), Ctrl-C is exit `130` and an end of input exit `2`, neither leaving a half-written file.
- The terminal dialog is `questionary` (MIT) on `prompt_toolkit` (BSD), imported by
  `modelroom/dialog.py` and nowhere else, pinned after a gate in PowerShell 5.1 under both the
  Windows console host and Windows Terminal (AGENTS.md, "Terminal dialog library").
- `render` writes a JSON view next to the Markdown one, under the same stem: both come from one
  `document.RenderDocument`, so no output can state a number another one does not. The document
  is a contract of its own (`RankedEntry`, `SetAsideEntry`, `MachineRanking`, `RenderDocument`).
- `scripts/selftest.py`: the acceptance run of the guided mode, one step per criterion of the
  plan, twice through `--answers` against the pinned search answer, with the hardware measurement of
  this machine and the live search as a smoke that decides nothing. It moves its home folder into
  a temporary tree and refuses to start if the pointer file would land anywhere else.
- Schema 2 contracts for the guided mode (CONTRACTS.md, "Schema 2: profiles, measurements,
  guided mode"): hardware profile v2 keyed by a random `profile_id` with a source for every
  memory value, a GPU state and an llmfit cross-check; scenario, measurement records as their
  own files and measurement protocol v1 (`protocol_v1.toml`); export object; pointer file and
  profile takeover rule; relation check; shipped catalog (`catalog.toml`); fit on profile v2;
  ranking rule; search hit, requirement and note shapes.
- `modelroom migrate --config <file>`: moves schema-1 hardware profiles and configuration to
  schema 2 under the lock, keeps `*.v1.bak` backups, and says `nothing to do` on a second run.
- `modelroom export-profile --config <file> [--profile <id>] --out <file>` and `modelroom
  import-profile <file> --config <file>`: hand one machine's hardware profile and its
  measurements to another results folder (CONTRACTS.md, "Export and import"). The import runs
  under the lock, never overwrites a younger profile, takes measurements by `measurement_id`
  (idempotent / conflict / new), adds a `[machines.<name>]` entry with the reserves from
  `[defaults]` and `writer = false` (keeping the configuration as the user wrote it once as
  `<config>.bak`), lists broken files instead of loading them, and never changes this machine's
  own binding.
- `run_fetch` accepts a `RequestBudget` shared with the caller's own requests.
- Language standard (`AGENTS.md`, "Language standard"): US English and one word per idea, with
  `tests/test_language_standard.py` as the guard over every module under `modelroom/` and the
  prose files. It reports British spellings from a fixed list, a model's age stated as a
  comparison (`older`, `newer`, `outdated` -- the status is `latest`, `legacy` or `unknown`), a
  verdict on a packager (`untrusted`, `trustworthy` -- only the owner class is stated) and
  German, and carries a reasoned exception list for text being changed elsewhere.
- `modelroom/search.py`: the Hugging Face search behind the guided mode, over the transport this
  package already has and with no SDK dependency (CONTRACTS.md, "Search over the Hugging Face
  API"). One pinned request answers everything the resolution needs; a hit resolves only when
  the relation check proves it a `quantized` build of exactly one base model whose account the
  catalog knows as a publisher, and every unresolved hit is shown with its reason instead of
  being dropped. `decide_age` states `latest`/`legacy` from positive evidence only -- a valid
  `new_version` at the publisher repository, else the catalog's own statement, else `unknown`
  (CONTRACTS.md, "Latest and legacy evidence"). `apply_hits` turns resolved hits into families
  and owner-bound `repos` targets, `write_configuration` writes that configuration under the
  state lock once it reads back unchanged. Search, resolution, successor lookups and the fetch
  share one request budget, `DEFAULT_GUIDED_BUDGET = 60`.

- `modelroom/measure.py`: modelroom measures the machine itself (CONTRACTS.md, "Hardware
  measurement") -- physical RAM from `GlobalMemoryStatusEx`, `/proc/meminfo` or
  `sysctl hw.memsize`, VRAM from `nvidia-smi` for one adapter, the graphics adapter from
  `lspci -nn`/`Win32_VideoController.PNPDeviceID` against a positive list of display-only
  vendors, a cgroup v2 memory limit as a note, and the OS identifier behind a salted digest.
  Every source runs through an injected command or file layer with a 10 s timeout and a defined
  failure, and returns a value with its source or a reason, never an exception.

### Changed

- **Yes or no is a selection list** of `Yes` and `No` with the default under the pointer; the
  `(Y/n)` prompt is gone. An answer file still answers `true`/`false`.
- **Nothing is marked that costs time or changes a measurement.** The load test's list of installed
  models starts unmarked, and `this machine` in the machine list starts unmarked once this folder
  already holds a profile of it (`this machine (measure again, last measured <date>)`). Both
  answered `Enter` with work nobody had asked for.
- **A reason is said once, with a number.** The search prints one line per reason with the number of
  repositories behind it instead of one line per unresolved repository, and shows every hit in the
  list with its reason in plain words. The terminal view of the render groups `not covered` and
  `too tight` by reason with up to three names (`and 9 more`). The load test groups what it left out
  the same way. The Markdown view is unchanged to the byte.
- **`context_origin`** is `entered` for every context the guided dialog writes, 8192 included:
  `default` is what the three automation commands assume when nobody chose one. A later
  `modelroom render --config` of a folder that kept a context reads it back as `entered` too
  (`render_cmd.scenario_from_config`); `render_cmd.scenario_for`, which called 8192 `default`,
  is gone.
- A schema-1 configuration that is migrated on the way in says so in one plain sentence before the
  migration's own lines, and the run ends with the line that says where the result was written.
- `render` reads schema 2 and ranks: per `[machines.<name>]` it takes the profile that
  `profile` names, computes `compute_fit_v2` for **every** eligible package against one context
  for the whole document (`Scenario`, 8192 unless the guided mode passes another), and shows the
  full ranking per machine (top ten, with the total), a `not covered` block and a `too tight`
  block, each with its reason and one plain-language note per package built from named facts.
  The selection rule (one variant per packager), the no-recommendation row, the Installed column
  and the Speed column that read measurements embedded in a schema-1 profile are gone; speeds now
  come from the measurement files of that profile. A machine whose profile is a schema-1 file is
  shown as `legacy` with the fit rule's own reason and nothing is persisted; a hardware file that
  does not read is a `note:` line instead of ending the whole render with exit `3`.
- `fetch_with_config` takes the request budget the caller already spent on, so the guided run's
  search and its fetch share one budget.
- `scripts/release-smoke.py` no longer places a schema-1 profile for the renderer: it names the
  profile `modelroom hardware` just measured in `[machines.<name>].profile`, the one write the
  guided mode does, and then checks the ranking header, the not-covered block and that the JSON
  view agrees with the Markdown one.
- `fetch_with_config` takes the request budget the caller already spent on, so the guided run's
  search and its fetch share one budget.
- `scripts/release-smoke.py` no longer places a schema-1 profile for the renderer: it names the
  profile `modelroom hardware` just measured in `[machines.<name>].profile`, the one write the
  guided mode does, and then checks the ranking header, the not-covered block and that the JSON
  view agrees with the Markdown one.
- `modelroom hardware` writes a schema-2 profile `<state>/hardware/<profile_id>.json` from its
  own measurement and binds this machine in the pointer file; `--machine` is now optional and
  `--cpu-only`/`--new-identity` are new. `llmfit` is a cross-check of the physical RAM and the
  VRAM instead of the source: a missing one is `absent`, a failing one `error`, and neither
  fails the command (exit `2` for a missing `llmfit` is gone); an entered `--cpu-only` VRAM is
  not compared at all. The pointer file is read inside the lock and written as a merge onto a
  fresh read, and a new `profile_id` avoids every file name in the folder, so a schema-1 file is
  never overwritten. A profile that was written but could not be bound is exit `1`. The schema-1
  writer stays for `render` and `modelroom migrate` until the renderer switches.
- Configuration schema 2: `repos`, `[machines.<name>].profile`, `[defaults]`, `[updates]`,
  `[guided]`; `families` may be empty. Schema 1 still reads; the shipped example is schema 2.
- Prose converted to the language standard in `AGENTS.md`, `README.md`, `CONTRACTS.md` and
  `modelroom/contracts.py`: US spellings, and the `Approval` shape described as a human
  vouching for an exact content. No schema name, field name or literal value changed.
- `fetch` works from the shared target set `config.package_targets`, grouped by owner
  (`hf.py::target_owners`, replacing `candidate_owners`): one area per (base model, owner)
  covering all of that owner's targets, `complete` only when every one of them was worked
  through, otherwise `incomplete` with the first failure and the owner's previous stock
  untouched. An owner-bound `repos` target is fetched even when its owner is no configured
  packager. CONTRACTS.md, "Target set and areas".
- `decide_provenance` checks a Hugging Face package against that same target set and against
  `relation.check_relation`: a repository outside the set stays `unresolved (repo_name)`, and a
  repository that does not declare itself a `quantized` build of exactly this base model carries
  the relation check's own status as its reason (`base_model_tag`, `relation_unknown`,
  `derivative`, `metadata_conflict`). A `base_model:` tag alone is no longer enough, so a
  package that used to reach `metadata_ok` on the tag alone is now `relation_unknown` until a
  human `Approval` says otherwise.

### Fixed

- A search that resolves no model to pick from -- `mistral` with the owner filter on, whose
  59 repositories all belong to owners the catalog does not list as publishers -- ended the run
  with a traceback out of the dialog library, because the selection list was still asked with
  nothing in it (test round of 2026-09-24). The step now prints its count and its reasons, says
  that nothing was added, and goes on without the question (`modelroom/guided.py::_search_step`).
- `merge_snapshot`: a repository that moved from one base model to another between two runs
  could end up deactivated under its old base model when the old base model's area was merged
  after the new one; the stale step now leaves an identity alone that another area of the same
  run has published, so the result no longer depends on area order.
- `validate_hf_repo` (every `owner/name` in the contracts and the configuration) requires each
  half to start with a letter or digit; `Qwen/..`, `../x` or `Qwen/-a` used to pass and became
  path segments of the URLs this package builds.
- A configuration file that is not valid UTF-8 is a `ConfigError` naming the path (exit 2)
  instead of a traceback; a measurement file nested beyond what the JSON parser carries is listed
  as unreadable instead of crashing the reader.

### The first cut, 2026-09-23

The subcommands, the contracts and the release tooling, before the guided mode.

#### Release

- `scripts/release-check.py`: the release gate. Checks a clean working tree, a valid PEP 440
  version without a local part with its own section in this file and, with `--tag`, an
  annotated tag `v<version>` on HEAD; runs the negative list of `tests/test_public_hygiene.py`
  (the same module, not a copy) over every file of HEAD, every line ever added in the history
  (binary files as text, renames as additions) and the raw commit and tag objects; builds wheel
  and sdist offline from HEAD and checks their content and file classes. Findings name place
  and class, never the text found. Commit metadata may carry noreply addresses and the
  maintainer's own address, taken from `git config user.email` or
  `MODELROOM_RELEASE_MAINTAINER_EMAIL` at run time and never written into the repository. Exit
  `0` free, `1` findings, `2` cannot check.
- `scripts/release-smoke.py`: the delivery smoke test. Installs the built wheel into a fresh venv
  outside the repository, derives a customer configuration from `modelroom.example.toml`, runs
  `hardware`, `fetch` with the network blocked through an unreachable proxy (every area
  incomplete, exit `1`, packages kept, lock free) and `render` of a snapshot built from the
  recorded fixtures.
- README: installation from the package index and the two release commands; `AGENTS.md`: the
  release procedure. Neither script ships in the wheel or the sdist.

#### Documentation

- README brought to the current state: status alpha, requirements, a five-command quick start
  with the real flags (the planned `check` verb is gone), state files, the meaning and limits of
  the fit class, exit codes and network access, each taken from `CONTRACTS.md`/`AGENTS.md`.

#### Added

- Render (AP5): the `modelroom render --config ...` command (`modelroom/cli.py`,
  `cli.render_with_config`, same pattern as `fetch_with_config`/`hardware_with_config`) and the
  pure Markdown document builder it wraps (`modelroom/render.py::build_document`). A pure
  reader of the current snapshot and every configured machine's hardware profile: eligibility
  (`metadata_ok`/`approved` provenance, complete, active), the per-group-per-machine selection
  rule (largest good-or-better quant, else the smallest, one row per distinct package picked by
  any machine), the fit/installed/speed cell rules and a new minimal `Rating` contract
  (`modelroom/contracts.py`) for an optional market-index Stars column -- any exception from a
  `RatingSource` (including the new `render.RatingUnavailableError`) is treated the same way: a
  `Market rating unavailable: <message>` note near the top and every Stars cell `–`, still exit
  `0`. Runs under the same kernel lock as `fetch` and the same F13 schema-version-before-lock
  ordering; refuses to overwrite a rendered document that is already newer than the snapshot
  being rendered (exit `1`, file untouched). Writes atomically via the new
  `state.py::atomic_write_text` (`.pid.tmp` + `os.replace`, alongside `atomic_write_json`);
  both writers now pin LF line endings on every platform (measured 2026-09-22: the snapshot
  and the rendered document came out CRLF on Windows).
  Documented in `CONTRACTS.md` ("Render (AP5)", the `Rating` model, and the updated Exit codes
  table). Covered by `tests/test_render.py` (the pure builder: selection, fit/installed/speed
  cells, stars/rating, header round-trip) and `tests/test_cli.py` (the `render` verb end to
  end: the lock, every schema-version/no-snapshot/newer-document exit path, and
  `render_with_config`).
- Hardware (AP4): the `modelroom hardware --config ... --machine ...` command
  (`modelroom/cli.py`), the persisted per-machine hardware profile
  (`<state>/hardware/<machine>.json`, `HardwareSnapshot`/`InstalledModel`/`Measurement` in
  `modelroom/contracts.py`), a dependency-injected `llmfit` subprocess binding with a version
  gate (`modelroom/llmfit.py`), the local Ollama daemon inventory
  (`modelroom/ollama_local.py::fetch_installed_models`, `GET /api/tags`, never a command
  failure when the daemon is unreachable), and the pure fit computation contract v1
  (`modelroom/fit.py::compute_fit` -- weights/KV-cache sizing, GPU/RAM pool selection, the
  perfect/good/marginal/too_tight thresholds, and the CPU cap). `hardware` only requires the
  named machine to be configured, not a `writer`, and takes no lock (each machine writes only
  its own file). New fixtures (`llmfit_system_laptop.json`, `llmfit_version.txt`,
  `ollama_tags_local.json`, all recorded 2026-09-22) are documented in
  `tests/fixtures/README.md`; the models, the llmfit/Ollama field mappings and "Fit contract
  v1" (formula, thresholds, the exact "fit (computed, v1)" wording a renderer must use) are
  documented in `CONTRACTS.md`. Covered by `tests/test_contracts.py` (new models),
  `tests/test_llmfit.py`, `tests/test_ollama_local.py`, `tests/test_fit.py`,
  `tests/test_state.py` (hardware snapshot path/load/write) and `tests/test_cli.py` (the
  `hardware` verb end to end, including every exit-2/exit-3 path).
- The validated `EXAMPLES` of every contract model moved from `modelroom/contracts.py` to
  `modelroom/examples.py` (data next to the tests that use it; the contract module keeps the
  models and rules, and stays under the repository's file-size guard).

- Project scaffold: package metadata, MIT license, tool-neutral rules (`AGENTS.md`), contract
  discipline (`CONTRACTS.md`), documentation map, public hygiene test and git hooks that run it,
  single-source version test.
- Contracts (AP1): the Pydantic models for family/base-model/package/snapshot identity
  (`modelroom/contracts.py`), the provenance decision rules (`modelroom/provenance.py`), and
  quantization parsing and package ordering (`modelroom/quantization.py`), each documented in
  `CONTRACTS.md` with a validated example and covered by fixture-backed tests.
- Configuration (AP2): the `Configuration` model tree and the `modelroom.toml` reader
  (`modelroom/config.py`) -- `BaseModelConfig`, `FamilyConfig`, `MachineConfig`, `PathsConfig`,
  `LlmfitConfig`, cross-checked family/owner rules, and `load_config`/`Configuration.from_dict`
  with `schema_version` checked before field validation, same convention as `load_snapshot`.
  Shares its `hf_repo`/`repo_aliases`/`ollama_base`+`ollama_tag` validators with
  `BaseModelSpec` via three functions extracted from `modelroom/contracts.py`. Ships
  `modelroom.example.toml` at the repo root; documented in `CONTRACTS.md` under a new
  "Configuration" section and covered by `tests/test_config.py`.
- Fetch (AP3): the `modelroom fetch --config ... --machine ...` command
  (`modelroom/cli.py`, registered as the `modelroom` console script), a minimal stdlib HTTP
  transport with a per-run request budget (`modelroom/http.py`), Hugging Face and Ollama
  fetchers (`modelroom/hf.py`, `modelroom/ollama.py`) that assemble `Package`s from a
  packager's file tree or an Ollama library's manifests, and the run state that ties a fetch
  run together (`modelroom/state.py`): a lock file, the snapshot merge/version rules, atomic
  writes, and `run-status.json`. Every fetcher takes its `Transport` as a parameter --
  dependency injection throughout, fixture-backed and fully offline in tests (`tests/test_hf.py`,
  `tests/test_ollama.py`, `tests/test_state.py`, `tests/test_fetch.py`, `tests/test_cli.py`,
  `tests/test_http.py`). New fixtures and one measured deviation (Hugging Face returns `401`,
  not `404`, for a repo an anonymous caller cannot see) are documented in
  `tests/fixtures/README.md`; the runtime state files, area/merge/version semantics and the
  `parameters_b` fallback are documented in `CONTRACTS.md` under "Fetch runtime state (AP3)".

#### Fixed

- Fix-round 1, thirteen confirmed review findings against AP3+AP4:
  - **F1** `modelroom/state.py::acquire_lock` creates the lock file with
    `os.open(O_CREAT | O_EXCL)` instead of a `path.exists()`-then-`write_text` race, verifies a
    stale-lock takeover against a random `token` it re-reads after writing, and returns that
    `token`; `release_lock(path, token)` deletes only when the file still carries it.
    `modelroom/cli.py` adapted to pass the token through.
  - **F2** `modelroom/state.py::merge_snapshot` drops a package whose area is no longer among
    this run's `area_outcomes`, instead of keeping every old package forever regardless of
    whether its base model or area is still configured.
  - **F3** `modelroom/hf.py::fetch_hf_area` and `modelroom/ollama.py::fetch_ollama_area` run
    package assembly inside the same error handling as their tree/manifest fetch and validate
    every entry/layer's shape, so a malformed registry response (e.g. a tree entry or manifest
    layer that is not an object) ends only that area `incomplete`, never raises out of
    `run_fetch`.
  - **F4** a `weights`/`weights_shard` file/layer with no non-negative integer size is now a
    shape error (F3) rather than a silent `0`; `modelroom/fit.py::compute_fit` additionally
    returns `fit_class="unknown"`, `reason="a weight file has no size"` for any `Package` that
    still reaches it with a 0-byte weight file.
  - **F5** `modelroom/ollama.py::_fetch_manifest` computes the manifest digest as `sha256:` +
    sha256 of the `GET` body and makes no `HEAD` request at all any more (measured 2026-09-22:
    the body hash equals the registry's old `HEAD` header exactly for the recorded `9b`
    fixture).
  - **F6** `modelroom/fetch.py::run_fetch` builds a `previous_by_key` map from the old
    snapshot's packages and passes it into both fetchers, so an `Approval` bound to still-current
    content survives a fetch instead of being silently discarded every run.
  - **F7** `modelroom/config.py::load_config` rejects a `paths.state` that resolves (symlinks
    followed) outside the config file's own directory tree; `paths.markdown` is not confined.
  - **F8** `load_config` resolves its own `path` argument to an absolute path first, so a
    relative `--config` no longer fails to make `paths.state`/`paths.markdown` absolute.
  - **F9** `modelroom/fetch.py::run_fetch` stops fetching entirely once the request budget is
    exhausted; a base model this run never even started reading keeps its previous
    `BaseModelSpec` (or an unknown one with no previous snapshot) instead of being overwritten
    with a fresh, budget-starved "unknown" reading, and every area it would have needed is
    recorded incomplete with `"budget exhausted before this area was started"`.
  - **F10** `modelroom/http.py::Response` gained `requests_made` (default `1`);
    `UrllibTransport` disables `urllib`'s own uncounted redirect following, follows up to 5
    `GET`/`HEAD` hops itself, and reports the hop count; `BudgetedTransport` charges every hop
    against the run's budget.
  - **F11** `BaseModelSpec.parameters_b` is now `float | None` (`None` = not measured this run
    and no previous reading exists); the `1.0` placeholder is gone.
    `modelroom/provenance.py::_decide_ollama` resolves `("unresolved", "parameters_unknown")`
    when it is `None`, before the size-token tolerance check ever runs. A snapshot written
    before this change may still carry the placeholder `1.0` for a base model whose parameter
    count Hugging Face does not expose; the carry-over rule would keep it. Rebuild such a
    snapshot once (delete `modelroom.json`, run `fetch`) -- there is no automatic migration.
  - **F12** `modelroom/llmfit.py::check_llmfit_version`/`fetch_llmfit_system` catch
    `subprocess.TimeoutExpired`/`OSError` from the runner and a non-zero `--version` exit code
    as `LlmfitError`; `hardware_fields_from_llmfit_system` now validates that
    `system.total_ram_gb` is a positive number and `gpu_vram_gb`/`available_ram_gb` are
    non-negative numbers when present, instead of trusting the shape and crashing downstream.
    `FixtureRunner` may map an `args` tuple to an exception instance to raise.
  - **F13** `modelroom/cli.py::fetch_with_config` checks the existing snapshot's
    `schema_version` before calling `acquire_lock`, so an unsupported schema version exits `3`
    without ever creating (or disturbing) a lock file.

  Documented in `CONTRACTS.md` under "Lock file", "Merge rule", "Area semantics", "Request
  budget", "Approval carry-forward across fetch runs", `PathsConfig`, "Path contract",
  "`parameters_b` when Hugging Face has no answer this run" and "Ollama size-token tolerance".
- AP3 acceptance fixes against a live snapshot (13 base models, 79 areas, 358 packages,
  2026-09-22): the packager naming conventions in `modelroom/provenance.py` and
  `modelroom/quantization.py` now recognize per-quant subfolders (unsloth), the mradermacher
  dot separator (`<base>.<QUANT>.gguf`, including its own lowercase `f16`), Qwen's
  `-split-NNNNN-of-NNNNN` shard suffix, and a non-weight marker (`mmproj`/`imatrix`) anywhere
  in a basename rather than only as a prefix; `QUANT_ORDER` gained 22 previously-unparsed
  tokens (the ternary/1-bit family, the missing `IQ`/`Q2_K`/`Q3_K` members, unsloth's
  `UD-IQ4_XS`/`UD-IQ4_NL`, `MXFP4`); the
  Ollama size-token provenance rule is now a 15 % tolerance around measured `parameters_b`
  instead of exact equality (`safetensors.total` counts embeddings, so it can never equal a
  packager's rounded tag); `hf.py`'s duplicate shard-suffix regex was removed in favor of
  `quantization.py`'s single definition; and `modelroom.cli.fetch_with_config` is now a public
  function `_cmd_fetch` delegates to, for a programmatic caller that already holds a
  `Configuration`. Documented in `CONTRACTS.md` under "Ollama size-token tolerance" and
  "Package file-stem naming conventions".
- Fix-round 2, seven confirmed review findings against AP3+AP4:
  - **R1** `modelroom/state.py::acquire_lock`'s stale-lock takeover claims the existing file
    first with an atomic `os.replace(path, <path>.stale.<token>)` rename (only one of two racing
    processes can ever win it) instead of write-then-reread, which had a real window letting two
    processes each see their own token (measured directly on Windows as an intermittent
    `PermissionError`); capped at three attempts before raising `LockHeldError`. `release_lock`
    is symmetrically a claim-then-decide (`os.replace(path, <path>.release.<token>)`) instead of
    a check-then-unlink, which had the same kind of window.
  - **R2** `modelroom/hf.py::fetch_base_model_meta` now accepts a model-info `sha` only when it
    is a real 40-hex commit sha and `safetensors.total` only when it is a real number greater
    than zero, and wraps the architecture build in `try`/`except` -- a malformed `sha` or a
    `safetensors.total` of `0` could previously raise a pydantic `ValidationError` straight out
    of `run_fetch`, contradicting this function's own "never raises" docstring promise.
  - **R3** `modelroom/hf.py`/`modelroom/ollama.py::_carry_forward_approval` now also require
    `previous.base_model_hf_repo == stub.base_model_hf_repo` -- `package_identity_key` alone
    says nothing about which base model a package belongs to, so an approval could otherwise
    follow the bare `(repo, filename)`/`ollama_name` identity to an unrelated base model that
    happens to share a packager repo or tag.
  - **R4** `modelroom/fetch.py::run_fetch` checks `budgeted.remaining <= 0` before each base
    model *and* before each of its areas, not only after a base model finishes -- a budget
    already exhausted at the start of the run (or exhausted between two areas of the same base
    model) previously still let `fetch_base_model_meta`/a fetcher be called once more, silently
    resolving into a fresh budget-starved reading or a bare `"budget exhausted"` area error
    instead of `"budget exhausted before this area was started"`.
  - **R5** Inverts F10's redirect-following composition: `modelroom/http.py::UrllibTransport`
    makes exactly one request per call and returns a 3xx like any other status;
    `RedirectingTransport(inner)` now follows a chain by calling `inner` once per hop (up to
    `MAX_REDIRECTS = 5`), and `run_fetch` composes `RedirectingTransport(BudgetedTransport(...))`
    so every hop is checked and booked against the run's request budget *before* it is made,
    failure paths included, rather than `BudgetedTransport` charging the extra hops only after a
    chain already resolved. `Response.requests_made` and `BudgetedTransport`'s post-call
    accounting are removed.
  - **R6** No code change: a snapshot written before F11 landed may still carry the `1.0`
    placeholder for a base model Hugging Face reports no parameter count for; documented as a
    one-time manual rebuild (delete `modelroom.json`, run `fetch`), not an automatic migration,
    since the package is unreleased and the only existing snapshot is the operator's own.
  - **R7** `modelroom/llmfit.py::hardware_fields_from_llmfit_system` now rejects a non-finite
    number (`math.isfinite`, catching `NaN`/infinity that a naive `<= 0` check lets slip
    through), a `gpu_name`/`backend` that is not a string or `None`, and a `unified_memory` that
    is not a real `bool` (previously silently coerced by `bool(...)`, e.g. `bool("no")` is
    `True`). Second line of defense: `modelroom/cli.py::hardware_with_config` catches a pydantic
    `ValidationError` from `write_hardware_snapshot` and maps it to exit `2`, so an llmfit output
    shape neither validator has anticipated still exits cleanly instead of crashing.

  Documented in `CONTRACTS.md` under "Lock file", "Approval carry-forward across fetch runs",
  "Request budget" and "`parameters_b` when Hugging Face has no answer this run".
- Fix-round 3, four confirmed review findings against AP3+AP4:
  - **1+2** `modelroom/state.py`'s lock is now a kernel lock on a stable file, not R1's
    rename-based stale-lock takeover: that design could not be made exclusive against a third
    process (`os.replace` is not bound to the file generation a process actually read, so process
    A claiming a stale lock, creating a fresh one and returning could be followed by process B
    renaming *A's* fresh lock away and creating its own, with neither `os.replace` call ever
    failing), and giving a foreign lock back on release could hand the file back under a third
    process's now-current lock instead of the one it was actually taken from. `acquire_lock`
    now opens the file with `O_RDWR | O_CREAT` (never `O_EXCL`, never renamed, never deleted) and
    takes an exclusive non-blocking kernel lock (`msvcrt.locking`/`fcntl.flock`) on one reserved
    byte; the JSON content (no `token` field any more) starts one byte later, so it stays
    readable by another process the whole time the lock is held. `LockHeldError` no longer
    carries an age threshold -- a crashed holder's lock is released by the kernel on process
    exit, a live holder keeps it regardless of age -- so `LOCK_MAX_AGE`/`MAX_TAKEOVER_ATTEMPTS`
    and the stale-takeover machinery are gone. `acquire_lock` returns a `LockHandle`
    (`path`, `fd`) instead of a token string; `release_lock(handle)` empties the file (never
    deletes it) and is idempotent. `modelroom/cli.py::fetch_with_config` adapted to the new
    signatures. The three-process race test holds the winner's lock until the test releases it
    and demands exactly one winner per round, no retry (a first draft let the winner exit right
    after acquiring, which releases the lock and let a slower sibling win legitimately -- a test
    flaw that briefly looked like a lock-placement problem).
  - **3** `modelroom/hf.py::_valid_parameters_b` now converts `safetensors.total` inside a
    `try`/`except (OverflowError, ValueError)` and requires `math.isfinite(result) and result >
    0` on the converted value, not just `total > 0` before dividing: a subnormal `total` like
    `1e-320` passed the old check but underflowed to `0.0` after `/ 1e9` (which
    `BaseModelSpec.parameters_b`'s `gt=0` would then reject), and a JSON integer far outside
    `float` range (a raw `10**400`, which `json.loads` parses without complaint) overflowed the
    division itself.
  - **4** `modelroom/llmfit.py::_is_finite_number` now catches `OverflowError` from
    `math.isfinite` itself, which raises for an `int` too large to convert to `float` (e.g. the
    same `10**400` shape, this time in `total_ram_gb`) -- treated as "not finite", the same
    verdict as `NaN`/infinity, instead of crashing `hardware_fields_from_llmfit_system`.

  Documented in `CONTRACTS.md` under "Lock file" (rewritten) and `_valid_parameters_b`'s own
  docstring in `modelroom/hf.py`.
- Fix-round 4, two handle-management findings against the new lock:
  - **P1** `release_lock` is idempotent through `LockHandle.released`, not by catching `EBADF`:
    a closed descriptor's number is reused by the next `os.open`, so a second release of a stale
    handle used to truncate, unlock and close whichever file had inherited that number.
  - **P2** `acquire_lock` unlocks and closes the fd when writing the holder content fails after
    the lock was won; `release_lock` closes the fd in `finally`. Previously such a failure left
    the kernel lock held with no handle to release it until the process exited.
- Fix-round 5, nine confirmed findings (two review passes) plus fourteen P3 items verified
  one by one against `e87a208`:
  - **P2-1** `modelroom/http.py::UrllibTransport` refuses to open any URL whose scheme is not
    `https` or whose host is not on an explicit allow-list (production default: exactly
    `huggingface.co`/`ollama.com`/`registry.ollama.ai` over HTTPS, `127.0.0.1` over HTTP for the
    local Ollama daemon), raising the new `TransportSecurityError` before ever opening a
    connection. Probe: `UrllibTransport()("GET", "file:///…/pyvenv.cfg")` previously returned
    178 bytes of a local file -- `build_opener(_NoAutoRedirect)` still carried urllib's default
    `FileHandler`/`FTPHandler`/`DataHandler` alongside it, since `build_opener` auto-fills every
    default handler class not represented (directly or by subclass) among its arguments. The
    opener is now built by hand (`OpenerDirector` + `add_handler`, no auto-fill) from exactly
    `HTTPHandler`/`HTTPSHandler`/`_NoAutoRedirect`/`HTTPErrorProcessor`/`HTTPDefaultErrorHandler`,
    so `file://`/`ftp://`/`data:` are structurally impossible even if the allow-list check were
    ever bypassed. Every hop of a redirect chain re-enters the same check (both
    `RedirectingTransport` and `hf.py::_fetch_tree` call back into the same transport instance
    per hop), so a poisoned `Location`/`Link: rel="next"` target is refused on the hop that would
    have followed it, with no second request ever made.
  - **P2-2** A real loopback `ThreadingHTTPServer` in `tests/test_http.py` proves
    `_NoAutoRedirect` actually stops `urllib.request`'s own redirect following (`/a` answers 302,
    `UrllibTransport` returns it raw with exactly one request logged), and doubles as the test
    bed for P2-1's redirect-refusal cases -- nothing in `test_http.py` exercised the real
    transport against an actual HTTP response before this.
  - **P2-3** Every persisted datetime is now checked aware-UTC (`contracts.check_aware_utc`,
    already used by AP4's own fields): `Package.observed_at`/`last_seen`, `Area.last_success`
    and `Snapshot.run_at` gained the same validator `InstalledModel.observed_at`/
    `Measurement.*`/`HardwareSnapshot.measured_at` already had. Probe: a naive `Snapshot.run_at`
    validated without complaint, then `state.check_run_is_newer` raised `TypeError: can't compare
    offset-naive and offset-aware datetimes` out of `cli.main`. `cli.py` also gained
    `_read_snapshot`/`_read_hardware_snapshot`, wrapping every load site (`fetch`/`hardware`/
    `render`) so `json.JSONDecodeError` and pydantic `ValidationError` map to exit `3` with the
    file name in the message, the same as an unsupported `schema_version` -- a truncated file or
    one missing required fields previously crashed `main()` uncaught.
  - **F4** `render.parse_header_line` returns `None` (treated as "no header", the document is
    replaced) for a header whose `snapshot_run_at`/`rendered_at` is not an aware UTC datetime,
    not just for one that fails to parse at all -- a naive header timestamp used to compare
    against the new, always-aware snapshot's `run_at` and raise `TypeError`.
    `cli.py::_refusal_against_existing_document` also now treats an existing document that is
    not valid UTF-8 as no header (was: `UnicodeDecodeError` straight out of `render`).
  - **F5** (addendum, same rules as F1-F4) `render._select_for_machine` no longer falls back to
    "the smallest package" when fit v1 cannot judge *any* package of a group on a machine (every
    `fit_class` is `"unknown"` there, or the machine has no hardware profile) -- that machine
    makes no pick at all. Real case (Qwen3.5, measured 2026-09-22): the whole architecture is not
    covered by v1, so every package came back unknown, and the old fallback still picked an
    arbitrary smallest-quant package (e.g. `UD-IQ2_XXS`) with no basis at all. When *no* machine
    picks anything for a group, it renders one row instead of the ordinary per-package rows:
    every package cell `–` except each machine's fit cell (`no recommendation: <reason>`) and
    Variants (`<N> variants, none judged`). The "else the smallest" fallback still applies, but
    only across packages fit v1 *did* judge, when at least one was judged and none reached
    good/perfect.
  - **F6** (second addendum) `modelroom/ollama.py::_validated_layers` now raises when a manifest
    has no `'layers'` key at all (or an explicit `null`), instead of silently treating it as `[]`
    -- that used to build an empty/unresolved stub package under the *same* identity key as a
    previously valid package for the tag, which `merge_snapshot` would then silently overwrite
    the good package with.
  - **F7** `modelroom/hf.py::_files_from_tree`'s `if not name: continue` used to drop a
    `type: "file"` entry with a missing, empty or non-string `path` *before* the `isinstance`
    shape check ever ran; a tree entirely made of such entries looked like a genuinely empty,
    `complete` area, and the merge rule deactivated every one of the old area's packages. Now a
    shape error like any other (F3/F4), ending the area `incomplete` with the old packages left
    untouched.
  - **F8** `hf.py::_tree_entry_size`/`ollama.py::_weight_layer_size` reject a `'size'` at or
    above `2**63` -- `json.loads` parses a JSON integer literal of any width (e.g. `10**400`)
    without complaint, and `fit.py::compute_fit`'s `sum(...) / GIB` later raised `OverflowError`
    out of `render`/`fit` instead of the fetcher ending the area `incomplete`.
  - **F9** `modelroom/llmfit.py::SubprocessRunner`: (a) decodes with a fixed
    `encoding="utf-8", errors="replace"` instead of `text=True` alone, which decodes with the
    Windows system codepage, not UTF-8 -- a non-ASCII `gpu_name` in real `llmfit` output could
    raise `UnicodeDecodeError` uncaught by anything in this module; (b) resolves the bare name
    `"llmfit"` to an absolute path via `which` (`shutil.which` by default, injectable) *before*
    calling `subprocess.run`, rather than passing the bare name straight through -- an
    unqualified name's search order can check the current working directory before `PATH`.
    `check_llmfit_version`/`fetch_llmfit_system` are unchanged (still build `["llmfit", ...]`);
    only `SubprocessRunner`, the one runner never faked in a test, resolves it, so every existing
    `FixtureRunner`-based test is unaffected.
  - **P3-1** CONTRACTS.md's "Run status" section said `run-status.json` is written "at the end of
    every fetch run, including one that ends exit 1" without qualification; the exit-`1` cases
    that stop at the lock or the staleness check write nothing at all (`state.py::
    check_run_is_newer`'s own docstring: "before anything is fetched or written"). Reworded to
    say so explicitly -- no code change, the Exit-codes table (line ~66) was already accurate.
  - **P3-2** CONTRACTS.md's "Area semantics" described a base model with neither `ollama_base`
    nor `ollama_tag` configured as "a trivially complete, empty area" without qualification;
    `run_fetch` only ever calls `fetch_ollama_area` when *both* are set, so that area never
    actually appears in a merged `Snapshot` -- the early return is `fetch_ollama_area`'s own
    defensive default for a direct caller (already unit-tested), not something this section's
    "one area per..." rule describes as `run_fetch`'s output. Reworded; no code change.
  - **P3-3** `hf.py::_fetch_tree`'s `Link: rel="next"` pagination is capped at
    `_MAX_TREE_PAGES = 20` hops; a chain that has not terminated by then raises instead of
    fetching forever, and the page that would exceed the cap is never requested.
  - **P3-4** A packager repo's model-info `sha` is validated as a real 40-hex commit sha
    *before* it is used to build the tree-fetch URL (previously only validated once assembled
    into a `Package.revision`, after the unchecked URL had already been requested); an Ollama
    tag parsed from the (network-controlled) tags page is validated against the same charset
    `Package.ollama_name`'s tag half requires before it is used to build the manifest URL.
  - **P3-5** Verified, not changed: `llmfit.py::check_llmfit_version` already catches
    `FileNotFoundError` from a missing `llmfit` binary and raises `LlmfitError` with an install
    hint (`cli.py` maps it to exit `2`), already covered by `tests/test_llmfit.py`. F9's new
    `SubprocessRunner` tests add end-to-end coverage of the same path through the real runner.
  - **P3-6** (independently raised as P2 by the second review pass, treated as mandatory)
    `config.py::Configuration` gained `_check_repo_names_do_not_collide_across_base_models`: two
    base models sharing a packager repo name (the default `<name>-GGUF` or a `repo_aliases`
    entry) is now a configuration error at load time. Package identity
    (`quantization.package_identity_key`) never carries `base_model_hf_repo`
    (CONTRACTS.md, "Identity"), so two base models fetched into the same repo name would
    otherwise write into the *same* `state.merge_snapshot` dict entry, one silently overwriting
    the other with no error.
  - **P3-7** Two tests that did not test their names: `test_atomic_write_json_leaves_no_tmp_file_
    on_a_serialization_failure` renamed (`json.dumps` fails before any file is ever touched, so
    it never actually exercised the except-clause's cleanup) and a new
    `test_atomic_write_json_leaves_no_tmp_file_when_replace_fails` added that does (a real
    `os.replace` `PermissionError` when the destination is a directory);
    `test_incomplete_package_is_unknown_fit` now asserts the exact `fit.reason` instead of only
    `is not None`.
  - **P3-8** Verified, not removed: `cli.py::hardware_with_config`'s R7 `except ValidationError`
    around `write_hardware_snapshot` is still reachable after P2-3 -- a naive `now` (a legitimate
    input for a programmatic caller, even though the real CLI always passes `now=None`) makes
    `measured_at` naive, which `HardwareSnapshot.measured_at`'s aware-UTC check then rejects.
    Newly tested; was previously reachable but untested too.
  - **P3-9** Decided, not fixed: `state.py`'s atomic writers guarantee atomicity (a reader never
    sees a torn write) but not durability (no `os.fsync` before `os.replace`) -- every file this
    writes is a reconstructible cache of the registries/local daemon, recovered by re-running the
    command, never by restoring a backup. Documented in CONTRACTS.md, "Atomic writes: atomicity,
    not durability".
  - **P3-10** Tightened, not rejected: the three-real-process lock race test now has each child
    signal readiness *before* polling for "go", and the parent waits for all three signals before
    writing it -- closes the gap where a still-starting process might not even be in its polling
    loop yet when "go" appeared.
  - **P3-11** The stale-run CLI test now also asserts the lock file is left empty (`release_lock`
    always empties it, per CONTRACTS.md, "Lock file") -- was previously left unchecked.
  - **P3-12** `hf.py`'s tree-fetch catch clause now distinguishes `BudgetExhaustedError` from
    every other exception, passing its message through bare -- it previously prefixed *every*
    exception there with `f"{repo}: "`, contradicting CONTRACTS.md's "Request budget" promise
    that the area's error is exactly `"budget exhausted"`. The three existing tests asserting
    this by substring were tightened to exact equality.
  - **P3-13** `test_fetch_with_config_defaults_to_the_real_transport_and_clock` renamed to
    `..._accepts_omitted_transport_and_now_without_raising` -- its own comment already conceded
    a non-writer machine short-circuits before either default is ever constructed, so it never
    proved what its old name claimed (and, per AGENTS.md, never safely could: this suite must
    never let a real `UrllibTransport()` touch the live network).
  - **P3-14** `fetch_types.py` gained `error_text(exc)`, a `str(exc)` wrapper that falls back to
    naming the exception's type when that string is empty -- a message-less exception from a
    caller-supplied `Transport` (`raise SomeError()` with no args) used to become an empty
    `AreaOutcome.error`, which `Area`'s own validator rejects (`error` required when `status`
    is `"incomplete"`), raising a pydantic `ValidationError` out of `state.merge_snapshot`
    instead of the ordinary `incomplete` result every other failure produces. Applied at every
    `str(exc)` capture in `hf.py`/`ollama.py` that becomes an `Area.error`.

  Documented in `CONTRACTS.md` under "Exit codes", `check_aware_utc`'s own docstring, "Render
  (AP5)" (header/no-recommendation-row/fit-cell paragraphs), "Area semantics", "Request budget"
  and a new "Atomic writes: atomicity, not durability" subsection.
- Fix-round 6, thirteen findings against `0920f5a` (Codex round 7, verified one by one):
  - **R7-1** `config.py::PathsConfig` gained a model validator: `paths.markdown` must not equal
    and must not lie inside `paths.state`'s resolved directory tree -- probe: `paths.markdown`
    set to the state directory's own snapshot or lock file validated fine, which would have let a
    render silently corrupt state a `fetch`/`hardware` run depends on. `load_config` additionally
    rejects `paths.markdown` equal to the config file itself.
  - **R7-2** `llmfit.py::SubprocessRunner`'s default `which` is now `_default_which`, an explicit
    `PATH` walk, replacing `shutil.which` -- on Python 3.11 on Windows, `shutil.which` prepends
    the current working directory to the search list regardless of the `path` argument
    (`_win_path_needs_curdir` was only added in 3.12), reopening exactly the same-named-CWD-shim
    hole F9b (fix-round 5) was written to close. Only absolute `PATH` directories are searched (a
    relative entry is dropped, not resolved against the current directory); Windows tries each
    `PATHEXT` extension plus the bare name per directory.
  - **R7-3** A stored snapshot or hardware file whose JSON root is not an object (`[]`, `null`, a
    bare string) used to raise an uncaught `AttributeError` out of `contracts.load_snapshot`/
    `load_hardware_snapshot` (`data.get("schema_version")` assumes a `dict`); `state.py` gained
    `StateFileShapeError(ValueError)`, raised by `load_existing_snapshot`/
    `load_existing_hardware_snapshot` before either loader runs. `cli.py`'s
    `_read_snapshot`/`_read_hardware_snapshot` now also catch it and `UnicodeDecodeError` (a file
    that is not valid UTF-8 at all), mapping both to exit `3` with the file path, exactly like
    `JSONDecodeError`/`ValidationError` already did.
  - **R7-4** `http.py`'s local allow-list is now an origin, `(host, port)`
    (`_ALLOWED_HTTP_ORIGINS = {("127.0.0.1", 11434)}`), not a bare host -- probe:
    `http://127.0.0.1:59999/x` was accepted with the production defaults, so a poisoned
    `Location`/`Link` header could reach any local port, not only the real Ollama daemon. A URL
    with no explicit port is refused rather than matching "any port".
    `UrllibTransport`'s constructor parameter is renamed `allowed_http_origins` to match.
  - **R7-5** `http.py::_build_opener` registers `urllib.request.ProxyHandler()` again -- the
    explicit handler list `_build_opener` replaced `build_opener(_NoAutoRedirect)` with (P2-1,
    fix-round 5) dropped the `ProxyHandler` that `build_opener`'s auto-fill used to add for free,
    silently losing `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` support. The allow-list check still runs
    against the request's own target URL before the opener ever consults a proxy, so a proxy can
    change how an already-allowed request is reached but never widen what is reachable.
  - **R7-6** (decided, no code) Two configurations with different `paths.state` but the same
    `paths.markdown` are unsupported by contract and not guarded by a second lock -- documented in
    CONTRACTS.md, "Render (AP5)": one state directory owns exactly one markdown document, the
    newer-document refusal protects only against an older snapshot under the *same* state
    directory's lock.
  - **R7-7** `config.py::Configuration` gained a model validator: two base models sharing an
    `ollama_base` may not claim equal or prefix-overlapping `ollama_tag`s -- probe:
    `ollama_base="nova"` with `ollama_tag="7b"` on two different base models validated fine, but
    `ollama._keep_relevant_tags` keeps `tag == ollama_tag or tag.startswith(ollama_tag + "-")`, so
    both would resolve packages under the identity `nova:7b`, and the later-processed area would
    silently overwrite the other's package (the same P3-6 hazard, for Ollama instead of Hugging
    Face packager names).
  - **R7-8** `config.py::_check_repo_names_do_not_collide_across_base_models` (P3-6, fix-round 5)
    now compares the full `owner/name` candidate, not the bare repo name -- probe: `acme/Nova` and
    `other/Nova` with `packagers = []` were wrongly rejected (`'Nova-GGUF' is claimed by both`),
    even though `hf.py::fetch_hf_area` would probe `acme/Nova-GGUF` and `other/Nova-GGUF`, never
    the same repo. A collision sharing a configured packager still fails, unchanged.
  - **R7-9** `hf.py::_files_from_tree`: a tree entry with no `type` field at all (`{}`) was
    silently treated as "not a file" (a directory) and skipped, exactly the "genuinely empty,
    complete area" hazard F7 (fix-round 5) closed for a bad `path` -- `type` must now be a
    non-empty string before it is compared to `"file"`. `ollama.py::_validated_layer`: a manifest
    layer with no `mediaType`/`digest` at all (`{}`) passed through unchanged, and `_build_package`
    built an empty `format="unknown", complete=False` stub under the tag's real identity, silently
    replacing a previously valid package -- both fields are now required, non-empty strings.
  - **R7-10** `contracts.py::Architecture`'s `num_hidden_layers`/`num_key_value_heads`/`head_dim`/
    `max_context` gained an upper bound, `le=2**31 - 1` -- probe: `num_hidden_layers = 10**400`
    validated fine, then `fit.py::compute_fit` raised `OverflowError: integer division result too
    large for a float`, aborting the whole render. `compute_fit` also now catches `OverflowError`
    directly around its arithmetic, returning `fit_class="unknown",
    reason="architecture values out of range"` -- a second line of defense against an unrelated,
    unbounded field (`Package.default_context`) combined with otherwise in-bound values.
  - **R7-11** `render.py` gained a shared `_cell(value)` helper -- escapes `|` and collapses any
    `\r\n`/`\n`/`\r` to a single space -- applied to every cell carrying external text: area
    error, base model repo, packager, quantization/format labels, hardware `gpu_name`/`backend`/
    `installed_unavailable_reason`, fit/no-recommendation reasons, and a `RatingSource`'s error
    message. Probe: an `Area.error` of `"boom | extra\nsecond line"` produced 4 physical lines and
    an extra column in the areas table instead of the one row it should have been.
  - **R7-12** `cli.py::render_with_config`'s pre-lock snapshot read now also decides "nothing to
    render, run fetch first" (exit `1`) *before* `acquire_lock` is ever called -- previously the
    pre-read result was discarded and the check only ran again inside `_render_locked`, after the
    lock (and therefore the state directory and the lock file, both `acquire_lock` side effects)
    had already been created for a render that had nothing to do.
  - **R7-13** CONTRACTS.md's "Run status" section still said `run-status.json` is written "at the
    end of every fetch run, including one that ends exit 1" -- P3-1 (fix-round 5) had already
    concluded this needed rewording, but the correction was never actually applied to this
    section. Reworded to say explicitly what the code (`cli.py::fetch_with_config`/`_run_locked`)
    already did: written at the end of every run that got past the lock and the stale-run check;
    not written when the lock is held or a newer run is detected. No code change.
- Fix-round 7, three remaining findings against `db1c729` (Codex round 8, verified one by one):
  - **R8-1** `hf.py::_files_from_tree` now accepts exactly the two `type` values the tree endpoint
    sends, `"file"` and `"directory"`; any other value is a shape error that ends the area
    `incomplete` -- probe: `[{"type": "garbage"}]` still returned `[]` after R7-9, so a tree of
    unknown types was the same genuinely-empty, complete area that deactivates every old package.
  - **R8-2** `tests/test_http.py` gained an autouse fixture that removes every `*_proxy`
    environment variable (any spelling) before each test -- `urllib` reads them
    case-insensitively and on Unix a lowercase `http_proxy`/`no_proxy` wins over the uppercase
    value a test sets, so a machine with a proxy configured would have routed the loopback tests
    through it or made the proxy tests pass for the wrong reason. Each proxy test sets exactly the
    variables it needs on the clean slate. Test-only.
  - **R8-3** The two HF repo-name collision tests in `tests/test_config.py` copied the first base
    model's `ollama_base`/`ollama_tag` onto the second, so the R7-7 Ollama collision validator
    would have raised the expected `ValidationError` even with the HF check broken; the copy now
    carries no Ollama mapping and the tests match the HF message (`packager repo`). Test-only.
- Fix-round 8, one remaining finding against `1a7d5b4` (Codex round 9, verified with a probe):
  - **R9-1** The `tests/test_http.py` autouse fixture now also sets `NO_PROXY="*"` and removes
    `REQUEST_METHOD` after clearing every `*_proxy` variable -- probe: with no proxy variable at
    all `urllib.request.getproxies()` is empty and falls back to the Windows registry / macOS
    system proxy on those platforms (`getproxies_environment() or getproxies_registry()`), while
    `NO_PROXY="*"` alone keeps the environment dict non-empty (`{'no': '*'}`) and bypasses every
    host; `REQUEST_METHOD` present makes urllib drop `HTTP_PROXY` (`{}` from `{'http': …}`),
    which would have turned the positive proxy test into a direct connection. The two proxy
    tests override or delete `NO_PROXY` themselves. A regression test pins the clean-slate
    mechanism. Test-only.
