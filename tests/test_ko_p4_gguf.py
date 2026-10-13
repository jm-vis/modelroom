"""P4, part 1: the pure rules of stage 1 -- the GGUF header, the packages of a file list, the addresses, the scanner's own
hosts (AP9-KO, V1, V5, V6). No request is made here."""

from __future__ import annotations

import struct
from datetime import date

import pytest

import fixture_ko_p4 as F  # noqa: F401  (puts scripts/ on the path)
import catalog_scan_packages as K

from modelroom import http
from modelroom.catalog_findings import PackageCheck, compare_architecture, llamacpp_page_url, parse_arch_names

# --- the header of a GGUF file --------------------------------------------------------------------------------------


def test_the_architecture_is_the_value_of_general_architecture():
    assert K.gguf_architecture(F.gguf_head("nova7")) == "nova7"


def test_pairs_before_the_architecture_are_skipped_by_their_type():
    head = F.gguf_head("nova7", before=(F.kv_uint32("general.quantization_version", 2), F.kv_string_array("tokenizer.ggml.tokens", ["a", "bb", "ccc"])))
    assert K.gguf_architecture(head) == "nova7"


@pytest.mark.parametrize("data", [
    b"", b"GGU", b"GGUF", b"GGUX" + b"\0" * 40,  # no header, or another magic
    F.gguf_head("nova7", version=1),  # a version nobody here has read
    F.gguf_head("nova7")[:-3],  # cut inside the value
    F.gguf_head("Nova 7!"),  # not a GGUF identifier
    F.gguf_head(""),
    F.gguf_head("x" * 70),
    b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + F.kv_uint32("general.alignment", 32),  # the pair is not there
    b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40),  # a count nobody could read in the first bytes
    F.gguf_head("nova7", before=(F.kv_string_array("t", ["a"])[:-2],)),  # a pair cut off before the architecture
])
def test_a_header_that_cannot_be_read_whole_to_the_architecture_is_a_fault(data):
    with pytest.raises(K.HeaderError):
        K.gguf_architecture(data)


def test_a_pair_of_a_type_nobody_knows_is_a_fault_not_a_guess():
    odd = F._string("x.y") + struct.pack("<I", 99) + b"\0" * 8
    with pytest.raises(K.HeaderError):
        K.gguf_architecture(F.gguf_head("nova7", before=(odd,)))


def _nested(depth: int) -> bytes:
    inner = struct.pack("<IQ", 9, 1) * depth + struct.pack("<IQ", 4, 0)
    return F._string("tokenizer.x") + struct.pack("<I", 9) + inner


def test_arrays_nested_without_end_are_a_fault_not_a_crash_of_the_scanner():
    head = F.gguf_head("nova7", before=(_nested(1500),))
    assert len(head) < 64 * 1024
    with pytest.raises(K.HeaderError):
        K.gguf_architecture(head)


def test_an_array_of_arrays_within_the_depth_limit_is_skipped():
    assert K.gguf_architecture(F.gguf_head("nova7", before=(_nested(2),))) == "nova7"


def test_the_architecture_must_be_a_string_pair():
    wrong = F._string("general.architecture") + struct.pack("<I", 4) + struct.pack("<I", 7)
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + wrong
    with pytest.raises(K.HeaderError):
        K.gguf_architecture(data)


# --- the packages of a file list ------------------------------------------------------------------------------------


def test_a_package_is_a_set_of_weight_files_read_at_its_first_shard():
    files = [
        ("Nova-9B-Q8_0.gguf", 10), ("Q5_K_M/Nova-9B-Q5_K_M-00002-of-00002.gguf", 5), ("Q5_K_M/Nova-9B-Q5_K_M-00001-of-00002.gguf", 9),
        ("README.md", 1), ("mmproj-Nova-9B-F16.gguf", 2), ("imatrix-nova.gguf", 3), ("Nova-9B-IQ4_XS.safetensors", 4),
    ]
    found = K.weight_packages(files)
    assert [(p.file, p.identity, p.quantization) for p in found] == [
        ("Nova-9B-Q8_0.gguf", "Nova-9B-Q8_0.gguf", "Q8_0"),
        ("Q5_K_M/Nova-9B-Q5_K_M-00001-of-00002.gguf", "Q5_K_M/Nova-9B-Q5_K_M.gguf", "Q5_K_M"),
    ]


def test_a_set_whose_first_shard_is_missing_is_no_package():
    assert K.weight_packages([("Nova-9B-Q8_0-00002-of-00002.gguf", 5)]) == []


def test_two_sets_of_one_identity_with_different_shard_counts_are_two_packages_the_stage_will_not_tell_apart():
    found = K.weight_packages([("X-Q8_0-00001-of-00002.gguf", 1), ("X-Q8_0-00001-of-00003.gguf", 1)])
    assert [p.identity for p in found] == ["X-Q8_0.gguf", "X-Q8_0.gguf"]


def test_the_count_of_a_quantization_is_over_the_packages_of_the_repository():
    found = K.weight_packages([("Nova-9B-Q8_0.gguf", 1), ("Nova-9B.Q8_0.gguf", 1), ("Nova-9B-Q4_K_M.gguf", 1), ("odd-name.gguf", 1)])
    counts = K.same_quantization(found)
    assert [counts[p.file] for p in found] == [1, 2, 2, None]


# --- the addresses ----------------------------------------------------------------------------------------------------


def test_the_list_query_asks_the_quantization_relation_of_one_row_with_gguf_and_sha():
    url = K.list_url("acme/Atlas-2-9B")
    assert url.startswith("https://huggingface.co/api/models?")
    assert "filter=base_model%3Aquantized%3Aacme%2FAtlas-2-9B" in url and "filter=gguf" in url
    assert "expand=gguf" in url and "expand=sha" in url and "expand=tags" in url and "author=" not in url


def test_the_file_list_and_the_header_are_bound_to_the_revision():
    assert K.tree_url("o/r", F.SHA_A) == f"https://huggingface.co/api/models/o/r/tree/{F.SHA_A}?recursive=true"
    assert K.header_url("o/r", F.SHA_A, "Q8_0/a b.gguf") == f"https://huggingface.co/o/r/resolve/{F.SHA_A}/Q8_0/a%20b.gguf"
    assert K.card_url("o/r", F.SHA_A) == f"https://huggingface.co/o/r/resolve/{F.SHA_A}/README.md"
    assert K.model_url("o/r") == "https://huggingface.co/api/models/o/r"


# --- the hosts of the scanner, and only of the scanner --------------------------------------------------------------


def test_the_scanner_hosts_add_the_source_and_the_hub_cdn_to_the_runtime_hosts_and_nothing_else():
    hosts = K.ScannerHosts()
    for host in ("huggingface.co", "ollama.com", "registry.ollama.ai", "api.github.com", "us.aws.cdn.hf.co", "cas-bridge.xethub.hf.co"):
        assert host in hosts
    for host in ("github.com", "evilhf.co", "hf.co.evil.com", "raw.githubusercontent.com", "example.org", ""):
        assert host not in hosts


def test_the_host_list_of_the_runtime_is_unchanged():
    assert http._ALLOWED_HTTPS_HOSTS == frozenset({"huggingface.co", "ollama.com", "registry.ollama.ai"})
    with pytest.raises(http.TransportSecurityError):
        http.UrllibTransport()("GET", "https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD")


# --- a stored check against a newer source -----------------------------------------------------------------------------


def _check(result: str = "absent", architecture: str = F.UNKNOWN, checked: str = "2026-10-01", commit: str = "c" * 40) -> PackageCheck:
    return PackageCheck.model_validate({
        "repo": "other/Nova-9B-GGUF", "revision": F.SHA_A, "identity": "Nova-9B-Q8_0.gguf", "file": "Nova-9B-Q8_0.gguf", "quantization": "Q8_0",
        "architecture": architecture, "architecture_source": "file_header", "result": result, "llamacpp_commit": commit,
        "source_url": llamacpp_page_url(commit), "checked": checked, "same_quantization": 1,
    })


def test_a_result_that_a_newer_source_changes_is_proposed_again_with_the_source_and_the_day():
    names = parse_arch_names(F.llama_with(F.UNKNOWN).decode())
    new = K.recompare(_check(), names, F.LLAMA_COMMIT, date(2026, 10, 13))
    assert new.result == "present" and new.llamacpp_commit == F.LLAMA_COMMIT and new.checked == date(2026, 10, 13)
    assert new.source_url == llamacpp_page_url(F.LLAMA_COMMIT) and new.revision == F.SHA_A and new.file == "Nova-9B-Q8_0.gguf"


def test_a_result_that_stays_is_not_proposed_again_so_the_catalog_does_not_change_every_day():
    names = parse_arch_names(F.LLAMA_TEXT.decode())
    assert compare_architecture(F.UNKNOWN, names) == "absent"
    assert K.recompare(_check(), names, F.LLAMA_COMMIT, date(2026, 10, 13)) is None
    assert K.recompare(_check("present", F.KNOWN), names, F.LLAMA_COMMIT, date(2026, 10, 13)) is None
