# 0013 -- Another computer: three ways in, neutral names at every machine boundary

- **Status:** accepted, decided 2026-09-29
- **Supersedes:** nothing; ADR 0001 to 0012 stay in force (ADR 0012 already names this one by number)

## Context

The maintainer usually sits at a laptop on the intranet and reaches a server only through remote
administration. Choosing a package for that server meant measuring it there by hand, exporting the
profile, carrying the file over, importing it, and copying an `ollama pull` line typed for the
first row only -- and the README promised more than the tool did: "modelroom itself never connects
to another computer", a fixed count of subcommands that never ask a question, and a promise that
`--config` is required by every one of them. A machine name also reached files in several places at
once -- a measurement's `display_name`, an imported profile's key, a measurement record's package
and daemon-version fields, modelroom's own backups -- with no single place deciding what of a name
or of foreign text was allowed into a file.

## Decision

1. **Three ways to reach another computer, chosen at the laptop.**
   - **Copy the command (way A).** Step 5's package choice for another machine hands over a line to
     paste there -- `modelroom pull <name> --weights-bytes <n>` where modelroom is installed too,
     else the plain `ollama pull <name>`. Nothing is sent anywhere on its own.
   - **A shared folder (way S), for a computer without SSH.** Set up once, in the machine list's own
     menu step, and kept per user account in `~/.modelroom/remote.json` (`profile_share`); an
     environment variable (`MODELROOM_PROFILE_SHARE`) always wins over it, for scripts. A folder is
     checked once by writing and removing a probe file before it is kept.
   - **Over SSH (way B).** A host alias of the user's own choosing, never the address or the domain
     it may resolve to, reaches `modelroom remote <op>` on the server (0.2.0 or newer, with Ollama):
     it reads the machine's profile, its free disk, and pulls a chosen package there, without the
     user ever running modelroom on that machine by hand.
2. **GF guardrails, binding for way B:**
   - only the system `ssh` is ever started, as a separate process with a fixed argument vector --
     never a shell string, never a name or a path where data could sit;
   - **never a password**: `BatchMode=yes` is always on, regardless of the user's own SSH
     configuration, so modelroom's own connection never even asks for one;
   - **never a host key accepted automatically**: `StrictHostKeyChecking=yes` is always on; an
     unknown or changed key stops the connection with a sentence naming the alias, never a prompt;
   - the alias is the only name kept, standing for both the connection and the machine: it becomes
     the profile's key and its `display_name`, and it is what the README and the question text ask
     the user to choose as a neutral name of their own, not the server's real one;
   - `~/.modelroom/remote.json` -- an account-local file next to the existing pointer file, never a
     results folder, never exported or rendered -- is the one deliberate exception to "no file
     carries a machine name": its `profile_share` field is kept exactly as typed and may carry a
     user's or a machine's real name, because that field has exactly one reader
     (`guided_share.share_folder`) and never reaches a results folder, an export, a render or a
     failure sentence;
   - **one active SSH target per run**: with several stored SSH profiles chosen, the run asks which
     one to connect to; every other one is ranked from its last stored profile, without a connection,
     so an identity question (see below) is never ambiguous;
   - **the record goes through the neutral sink twice**: once at the server, matching the package it
     just measured and naming it only from that package's identity, and again at the laptop before it
     is stored, so a server that did not hold the rules never gets its record kept;
   - a package name the server's own Ollama holds locally, or names in its `/api/ps`, are read only
     to decide what the request already asked about -- the installed name itself never reaches the
     laptop, is never shown, and is never stored.
3. **One sink for names (FR-N).** `modelroom/neutral.py` is the only place that decides what of a
   name and of foreign text may reach a file modelroom writes or exports; every writer passes its
   object through it first, and every writing command sweeps its folder with the same rules before it
   writes anything else. Failing to neutralize an object or a folder means nothing is written, not a
   partial write with a name in it.

## Consequences and deliberate limits

- **A POSIX process tree.** The `ssh` process and everything it starts are tracked as one group,
  signaled once and only after its leader has been seen to end, so that a helper such as a
  `ProxyJump` hop cannot keep stderr open past the call's own end. A descendant that explicitly
  breaks away from that group -- or a helper whose PID was reaped behind modelroom's back -- falls
  outside the tree; that is a named, accepted limit, not a bug to chase.
- **The server needs modelroom 0.2.0 or newer.** Without it, way B falls back to way A or way S; an
  older `modelroom remote` on the server is recognized by its argument parser's own exit code and
  reported as "older than 0.2.0", never silently ignored.
- **No anonymity guarantee.** modelroom cannot tell whether an alias is a name the user chose to be
  neutral or the server's real short name; it only refuses the shapes of an address, a domain and a
  neutral key's own pattern. Choosing a neutral alias stays the user's own responsibility (README).
- **What is not checked** stays as documented in CONTRACTS.md: whether the alias truly names the
  intended computer, the host-key rules of a jump host (the user's own `ssh_config`), where Ollama
  keeps its models on the server, and whose daemon actually answers there.

## Alternatives and why not

- **paramiko or asyncssh** instead of the system `ssh`: neither reads `ProxyJump`/`Include`/`Match`
  the way OpenSSH's own config parser does, keys and the agent would need a path of their own, and
  either is a new dependency with its own crypto stack. Rejected.
- **An SSH tunnel or `OLLAMA_HOST`** to reach the server's Ollama over HTTP: a port to manage, a
  background process, and a collision with the local daemon allowlist, which only ever permits
  `127.0.0.1:11434`. Rejected.
- **`ssh host ollama pull <name>` directly**: no disk check, no digest proof of what is already
  there, and the package name sits unchecked in a command line -- the same injection surface A2's
  `PULL_NAME_RE` exists to close. Rejected in favor of a modelroom counterpart on the server.
- **Probe-for-probe measurement over SSH**, reading the server's hardware one probe at a time: a
  Windows memory reading cannot be called this way over a remote shell, and a cross-check against
  llmfit would run on the wrong machine. Rejected in favor of modelroom itself measuring on the
  server.
- **`ssh -G` to verify an alias before use**: proves nothing about anonymity, since a user's real
  short name passes it as easily as a chosen one. Dropped as a false sense of safety.
- **The shared folder only through the environment variable** (question 7 (a)): not discoverable for
  a user who never opens a script or a terminal profile. Rejected in favor of the menu step, with the
  variable kept as the override for scripts.
- **A dedicated `modelroom share <folder>` command** with a mandatory drive letter and a
  neutral-named path (question 7 (b)): one more command to learn, and a drive letter is often not
  there in a non-interactive or an administrator session where a NAS path is. Rejected in favor of
  the menu step and accepting UNC paths outright (only the Windows device and pipe prefixes are
  refused, since a write there could open a pipe instead of a file).
- **Cleaning ssh's own stderr or the server's own text** before it could be shown: a scrubber can
  always miss a name. Rejected in favor of fixed local sentences that never carry the server's own
  text at all.
