"""The K2 rules of the catalog scanner: the top of a group without a `latest` row, and successors that meet in one run (AP9-KA-H).

Pure functions over values: no request, no file, no clock. This module imports `search_release` and nothing else of the
package or the scripts, so `catalog_scan_rules` and `catalog_scan_notes` can import it at module level without a cycle.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from modelroom.search_release import _maximal_members, _ordered_successors, line_key


class NameSuccessor(NamedTuple):
    """A successor the name rule proposes: the target, and in words why (`basis`, for the reader of the proposal)."""

    target: str
    basis: str


def name_successors(rows: list, state: dict, rejected: Callable[[str], bool], refused: frozenset | set = frozenset()) -> tuple[dict[str, NameSuccessor], list[str]]:
    """The successors the name rule of the search proposes for rows of the table: `({row: NameSuccessor}, notes)` (NF,
    decided 2026-10-14; the maintainer approves each one). The rule is `search_release`'s -- `line_key`, `stands_over`,
    the highest members of a group, `_ordered_successors` -- with these conditions on top, every one a test:

    1. both rows are in the table, of one publisher and one group `(publisher, family, line)`, of one `kind` (the
       search has names and no kind, so it groups a text and a vision model of one family; here they stay apart) and
       of one variation (`line_key`'s `variant`, the same set of words in any case: a LoRA, an audio or an Omni build is
       no replacement of the plain model -- a missing successor is allowed, a wrong one is not; the size may differ);
    2. the target stands over the row by a strictly higher version, a later date alone is not enough (U5 rule 4);
    3. both have a `hub_created` and the target's is later;
    4. the row has no successor and the target has none at the end of the run (`state`: `repo -> (latest, successor)`,
       `end_state`); neither is rejected, and the pair `(row, target)` is not in `refused` (the maintainer turned that
       very proposal down; another successor, or another field of the row, is still open);
    5. the target is a highest member of its subgroup (group and variation), the first of those in the order of
       `_ordered_successors`, so no chain arises among proposals and no cycle with a stated link.

    A row that is `latest` at the end of the run gets no proposal where the others would be made: it is a note (the
    maintainer decides that apart). Names the grammar does not read take no part."""
    shown = {row.hf_repo: row for row in rows}
    keys: dict[tuple, dict] = {}
    for row in rows:
        if key := line_key(row.hf_repo, row.parameters_b):
            keys.setdefault((row.kind, key.variant), {})[row.hf_repo] = key
    found: dict[str, NameSuccessor] = {}
    notes: list[str] = []
    for (kind, _variant), members in sorted(keys.items(), key=lambda item: (item[0][0], sorted(item[0][1]))):
        highest = _maximal_members(members)
        for repo, key in sorted(members.items()):
            tops = [top for top in highest[key.group] if members[top].version > key.version]
            if repo in highest[key.group] or not tops or rejected(repo) or state[repo][1] is not None:
                continue
            target = _ordered_successors(key, [(top, members[top]) for top in tops])[0]
            if (repo, target) in refused or not _later(shown[repo], shown[target]) or rejected(target) or state[target][1] is not None or repo.split("/", 1)[0] != target.split("/", 1)[0]:
                continue
            if state[repo][0]:
                notes.append(f"{repo}: latest in the catalog, but {target} stands over it by version; no successor proposed, decide separately")
                continue
            group = f"{key.publisher}/{key.family}" + (f" {key.line}" if key.line else "")
            below = f"version {_dotted(key.version)}" if key.version else "no version"
            found[repo] = NameSuccessor(target, f"name rule: version {_dotted(members[target].version)} stands over {below} in {group} ({kind})")
    return found, notes


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))


def _later(row, target) -> bool:
    """Both rows know the day their repo was created on the Hub, and the target's is later."""
    return row.hub_created is not None and target.hub_created is not None and target.hub_created > row.hub_created


def group_tops(fresh: list[tuple[str, str, float | None]], rows: list, touched=frozenset()) -> tuple[dict[str, str], list[str]]:
    """`({repo: "family top"}, notes)` for the groups `(kind, publisher, family, line)` that get a new member in this run
    (a new repo, or a row in `touched`: one with a field change) and hold no `latest` row. The members are the new repos
    and the rows (once each). The maximal members are those no other member stands over: one maximal new member is the
    top; one maximal row is no change (a note, nothing is promoted); several maximal members stand level (a note naming
    all of them, no top). A tie below the top does not prevent it. A group with no new member gives no note."""
    new = {repo for repo, _, _ in fresh}
    members = {repo: (kind, size) for repo, kind, size in fresh}
    for row in rows:
        members.setdefault(row.hf_repo, (row.kind, row.parameters_b))
    with_latest = {(row.kind, *key.group) for row in rows if row.latest and (key := line_key(row.hf_repo, row.parameters_b))}
    groups: dict[tuple, list] = {}
    for repo, (kind, size) in sorted(members.items()):
        if key := line_key(repo, size):
            groups.setdefault((kind, *key.group), []).append((repo, key))
    tops: dict[str, str] = {}
    notes: list[str] = []
    for group, items in groups.items():
        if group in with_latest or not any(repo in new or repo in touched for repo, _ in items):
            continue
        top = [repo for repo, key in items if not any(other.stands_over(key) for _, other in items)]
        label = f"{group[1]}/{group[2]} {group[0]}"
        if len(top) > 1:
            names = ", ".join(top[:-1]) + f" and {top[-1]}"
            notes.append(f"{label}: no latest, {names} stand level")
        elif top[0] in new:
            tops[top[0]] = "family top"
        else:
            notes.append(f"{label}: no latest row; the top is the shipped row {top[0]}")
    return tops, notes


def successor_pairs(row, target: str, card: str, today: str) -> dict:
    """The `old -> new` pairs of a successor change: the successor with its source and day, and a `latest` row gives up
    `latest` with its source and day."""
    pairs = {"successor": (None, target), "successor_source": (None, card), "successor_checked": (None, today)}
    if row.latest:
        pairs.update(latest=(True, False), latest_source=(row.latest_source, None),
                     latest_checked=(row.latest_checked.isoformat() if row.latest_checked else None, None))
    return pairs


def successor_refusal(row, target: str, kind: str | None, state: dict) -> str | None:
    """The note that says why the `new_version` of a card is no successor of `row`, `None` when it can be one. `kind` is
    the kind of the target in the table or the proposal (`None`: it is neither); `state` is `end_state`: the row must
    have no successor at the end of the run (a carried one counts), and a target counts only as a `latest` row without
    a successor then."""
    if kind is None or target.split("/", 1)[0] != row.publisher or target == row.hf_repo:
        return f"{row.hf_repo}: card names new_version {target}, which is not a model of the table or of this proposal; no successor proposed"
    if kind != row.kind or row.successor is not None or state.get(row.hf_repo, (False, None))[1] is not None:
        return f"{row.hf_repo}: card names new_version {target} of another kind or the row has a successor; no successor proposed"
    if state.get(target) != (True, None):
        return f"{row.hf_repo}: successor {target} from the card is not a latest row at the end of this run, not set"
    return None


def end_state(rows: dict, changes: dict) -> dict[str, tuple[bool, str | None]]:
    """`repo -> (latest, successor)` as the table would stand at the end of a run: the shipped rows, the new rows the
    proposal holds (carried ones and those that passed the last `_takes`) and the `latest` and `successor` fields of
    every field change it holds, carried ones included. `changes` is keyed `("new_row", repo)` / `("field", repo, name)`."""
    state = {repo: (row.latest, row.successor) for repo, row in rows.items()}
    for key, change in sorted(changes.items()):
        fields, whole = change.get("fields", {}), key[0] == "new_row"
        latest, successor = state.get(key[1], (False, None))
        state[key[1]] = (fields.get("latest", {}).get("new") if "latest" in fields or whole else latest,
                         fields.get("successor", {}).get("new") if "successor" in fields or whole else successor)
    return state


def resolve_successors(candidates: dict[str, str], pointed: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """The successors of one run that can stand together: `({row: target}, notes)`.

    `candidates` are the `row -> target` pairs that passed the single checks; `pointed` maps a repo to a row of the end
    state (shipped, or a carried change) whose successor it is. In this order: (b) a candidate whose row is pointed at
    is struck; (c) a cycle among the rest is struck whole, a note per edge; (d) the chains are resolved from their end:
    every candidate whose target is the row of no remaining candidate is set, then every candidate whose target has
    got a successor that way is struck, until none remains. `A -> B -> C` gives `{B -> C}`, `A -> B -> C -> D` gives
    `{A -> B, C -> D}`; the names play no part."""
    notes: list[str] = []
    left: dict[str, str] = {}
    for row, target in sorted(candidates.items()):
        if row in pointed:
            notes.append(f"{row}: successor {target} not set, {pointed[row]} points at {row}")
        else:
            left[row] = target
    cyclic: set[str] = set()
    for start in left:
        walk, node = [], start
        while node in left and node not in walk:
            walk.append(node)
            node = left[node]
        if node in walk:
            cyclic.update(walk[walk.index(node):])
    for row in sorted(cyclic):
        notes.append(f"{row}: successor {left.pop(row)} not set, part of a cycle")
    chosen: dict[str, str] = {}
    while left:
        end = {row: target for row, target in left.items() if target not in left}
        chosen.update(end)
        for row in end:
            del left[row]
        for row, target in sorted(left.items()):
            if target in chosen:
                notes.append(f"{row}: successor {target} not set, {target} has a successor")
                del left[row]
    return chosen, notes
