"""The validated example of every contract model, verbatim what `CONTRACTS.md` shows.

Kept apart from `contracts.py` so the contract module stays the models and their rules and
this module stays the data; `tests/test_contracts.py` validates every entry against its model
and checks that `CONTRACTS.md` repeats it unchanged.
"""

from __future__ import annotations

from copy import deepcopy

# Shared by more than one example below (e.g. `ExportObject` carries a profile and a
# measurement); each use is a deep copy, so a test that edits one example never edits another.
_HARDWARE_PROFILE = {
    "schema_version": 2,
    "profile_id": "3f9a0c21d4e6b870",
    "display_name": "workstation",
    "os_fingerprint": "9d2f4b6a8c0e1357",
    "os_fingerprint_source": "windows_machineguid",
    "origin": "measured",
    "recorded_at": "2026-09-23T08:00:00Z",
    "ram_physical_gib": 31.7,
    "ram_physical_source": "os",
    "ram_limit_gib": None,
    "ram_limit_scope": "none",
    "vram_gib": 8.0,
    "vram_source": "nvidia-smi",
    "gpu_state": "measured",
    "gpu_name": "Nova GPU",
    "llmfit_crosscheck": {
        "ram_physical": {"status": "confirmed", "own_gib": 31.7, "llmfit_gib": 31.9},
        "vram": {"status": "confirmed", "own_gib": 8.0, "llmfit_gib": 8.0},
    },
    "llmfit_version": "1.1.16",
}
_SCENARIO = {
    "context_requested": 8192,
    "context_origin": "default",
    "kv_type": "f16",
    "kv_type_assumed": True,
    "requests": 1,
}
_RUN_50_TPS = {
    "done_reason": "length",
    "eval_count": 128,
    "eval_duration": 2560000000,
    "prompt_eval_count": 42,
    "prompt_eval_duration": 90000000,
    "load_duration": 12000000,
}
_RUN_51_2_TPS = {**_RUN_50_TPS, "eval_duration": 2500000000}
_MEASUREMENT_RECORD = {
    "schema_version": 2,
    "measurement_id": "20260923T083000Z-5c1e9a07",
    "profile_id": "3f9a0c21d4e6b870",
    "protocol": "v1",
    "measured_at": "2026-09-23T08:30:00Z",
    "package": {
        "content_source": "ollama",
        "ollama_manifest_digest": "sha256:abababababababababababababababababababababababababababababababab",
        "hf_repo": None,
        "hf_revision": None,
        "hf_file_digest": None,
    },
    "ollama_name": "nova:7b",
    "daemon_version": "0.12.3",
    "scenario": dict(_SCENARIO),
    "runs": [dict(_RUN_50_TPS), dict(_RUN_51_2_TPS), dict(_RUN_50_TPS)],
    "tps_mean": 50.4,
    "tps_min": 50.0,
    "tps_max": 51.2,
    "validity": "valid",
    "validity_reason": None,
    "comparable": True,
    "comparable_reason": None,
    "load_state": {"gpu_utilization_percent": 3.0, "cpu_load": 0.4},
}
_CATALOG_MODEL = {
    "hf_repo": "acme/Nova-7B",
    "latest": False,
    "latest_source": None,
    "latest_checked": None,
    "successor": "acme/Nova-7B-2512",
    "ollama_base": "nova",
    "ollama_tag": "7b",
    "source": "https://huggingface.co/acme/Nova-7B-2512",
}
_CATALOG_FAMILY = {
    "name": "nova",
    "publisher": "acme",
    "models": [
        deepcopy(_CATALOG_MODEL),
        {
            "hf_repo": "acme/Nova-7B-2512",
            "latest": True,
            "latest_source": "https://huggingface.co/collections/acme/nova-2512-0123abcd",
            "latest_checked": "2026-09-23",
            "successor": None,
            "ollama_base": None,
            "ollama_tag": None,
            "source": "https://huggingface.co/acme/Nova-7B-2512",
        },
    ],
}

EXAMPLES: dict[str, dict] = {
    "Architecture": {
        "source_repo": "acme/Nova-7B",
        "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "kind": "dense_classic",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layer_types": None,
        "max_context": 131072,
    },
    "BaseModelSpec": {
        "hf_repo": "acme/Nova-7B",
        "repo_aliases": ["Nova-7B-Instruct-GGUF"],
        "ollama_base": "nova",
        "ollama_tag": "7b",
        "publisher": "acme",
        "parameters_b": 7.0,
        "architecture": {
            "source_repo": "acme/Nova-7B",
            "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
            "kind": "dense_classic",
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": None,
            "max_context": 131072,
        },
    },
    "Family": {
        "name": "nova",
        "base_models": [
            {
                "hf_repo": "acme/Nova-7B",
                "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                "ollama_base": "nova",
                "ollama_tag": "7b",
                "publisher": "acme",
                "parameters_b": 7.0,
                "architecture": {
                    "source_repo": "acme/Nova-7B",
                    "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
                    "kind": "dense_classic",
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "layer_types": None,
                    "max_context": 131072,
                },
            }
        ],
    },
    "PackageFile": {
        "name": "Nova-7B-Q4_K_M.gguf",
        "role": "weights",
        "size_bytes": 4500000000,
        "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
    },
    "Approval": {
        "date": "2026-09-01",
        "content": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "by": "acme-ai-team",
    },
    "Package": {
        "source": "huggingface",
        "repo": "packager/Nova-7B-GGUF",
        "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
        "base_model_hf_repo": "acme/Nova-7B",
        "format": "gguf",
        "files": [
            {
                "name": "Nova-7B-Q4_K_M.gguf",
                "role": "weights",
                "size_bytes": 4500000000,
                "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
            }
        ],
        "complete": True,
        "quantization": "Q4_K_M",
        "default_context": None,
        "provenance": "metadata_ok",
        "unresolved_reason": None,
        "approval": None,
        "observed_at": "2026-09-22T09:00:00Z",
        "last_seen": "2026-09-22T09:00:00Z",
        "active": True,
    },
    "Area": {
        "source": "huggingface",
        "base_model_hf_repo": "acme/Nova-7B",
        "packager": "packager",
        "status": "complete",
        "last_success": "2026-09-22T09:00:00Z",
        "error": None,
    },
    "Snapshot": {
        "schema_version": 1,
        "run_at": "2026-09-22T09:00:00Z",
        "areas": [
            {
                "source": "huggingface",
                "base_model_hf_repo": "acme/Nova-7B",
                "packager": "packager",
                "status": "complete",
                "last_success": "2026-09-22T09:00:00Z",
                "error": None,
            }
        ],
        "base_models": [
            {
                "hf_repo": "acme/Nova-7B",
                "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                "ollama_base": "nova",
                "ollama_tag": "7b",
                "publisher": "acme",
                "parameters_b": 7.0,
                "architecture": {
                    "source_repo": "acme/Nova-7B",
                    "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
                    "kind": "dense_classic",
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "layer_types": None,
                    "max_context": 131072,
                },
            }
        ],
        "packages": [
            {
                "source": "huggingface",
                "repo": "packager/Nova-7B-GGUF",
                "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
                "base_model_hf_repo": "acme/Nova-7B",
                "format": "gguf",
                "files": [
                    {
                        "name": "Nova-7B-Q4_K_M.gguf",
                        "role": "weights",
                        "size_bytes": 4500000000,
                        "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
                    }
                ],
                "complete": True,
                "quantization": "Q4_K_M",
                "default_context": None,
                "provenance": "metadata_ok",
                "unresolved_reason": None,
                "approval": None,
                "observed_at": "2026-09-22T09:00:00Z",
                "last_seen": "2026-09-22T09:00:00Z",
                "active": True,
            }
        ],
    },
    "InstalledModel": {
        "name": "nova:7b",
        "digest": "sha256:abababababababababababababababababababababababababababababababab",
        "size_bytes": 4500000000,
        "observed_at": "2026-09-22T09:00:00Z",
    },
    "Measurement": {
        "content_source": "ollama",
        "ollama_manifest_digest": "sha256:abababababababababababababababababababababababababababababababab",
        "hf_repo": None,
        "hf_revision": None,
        "hf_file_digest": None,
        "context": 8192,
        "runtime": "ollama 0.12.3",
        "profile_measured_at": "2026-09-22T09:00:00Z",
        "measured_at": "2026-09-22T09:05:00Z",
        "tps_mean": 42.5,
        "tps_range": [40.0, 45.0],
    },
    "HardwareSnapshot": {
        "schema_version": 1,
        "machine": "workstation",
        "measured_at": "2026-09-22T09:00:00Z",
        "llmfit_version": "1.1.16",
        "vram_gib": 11.94,
        "ram_gib": 127.46,
        "free_ram_gib_at_measurement": 76.64,
        "gpu_name": "Nova GPU",
        "backend": "CUDA",
        "unified_memory": False,
        "installed": [
            {
                "name": "nova:7b",
                "digest": "sha256:abababababababababababababababababababababababababababababababab",
                "size_bytes": 4500000000,
                "observed_at": "2026-09-22T09:00:00Z",
            }
        ],
        "installed_unavailable_reason": None,
        "measurements": [],
    },
    "Fit": {
        "fit_class": "good",
        "mode": "gpu",
        "need_gib": 7.125,
        "weights_gib": 5.0,
        "kv_gib": 1.125,
        "pool_gib": 10.94,
        "reserve_gib": 1.0,
        "context": 8192,
        "context_assumed": True,
        "reason": None,
    },
    "Rating": {
        "stars": 3.5,
        "source": "market index",
    },
    "CrossCheck": {
        "status": "confirmed",
        "own_gib": 31.7,
        "llmfit_gib": 31.9,
    },
    "LlmfitCrosscheck": {
        "ram_physical": {"status": "confirmed", "own_gib": 31.7, "llmfit_gib": 31.9},
        "vram": {"status": "confirmed", "own_gib": 8.0, "llmfit_gib": 8.0},
    },
    "HardwareProfile": deepcopy(_HARDWARE_PROFILE),
    "Scenario": deepcopy(_SCENARIO),
    "PackageRef": {
        "content_source": "huggingface",
        "ollama_manifest_digest": None,
        "hf_repo": "packager/Nova-7B-GGUF",
        "hf_revision": "0123456789abcdef0123456789abcdef01234567",
        "hf_file_digest": "sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
    },
    "RunCounters": deepcopy(_RUN_50_TPS),
    "LoadState": {
        "gpu_utilization_percent": 3.0,
        "cpu_load": 0.4,
    },
    "MeasurementRecord": deepcopy(_MEASUREMENT_RECORD),
    "ExportObject": {
        "schema_version": 1,
        "profile": deepcopy(_HARDWARE_PROFILE),
        "measurements": [deepcopy(_MEASUREMENT_RECORD)],
    },
    "GuidedPointer": {
        "schema_version": 1,
        "current": "//models/modelroom",
        "bindings": {"//models/modelroom": "3f9a0c21d4e6b870"},
    },
    "CatalogModel": deepcopy(_CATALOG_MODEL),
    "CatalogFamily": deepcopy(_CATALOG_FAMILY),
    "Catalog": {
        "schema_version": 1,
        "families": [deepcopy(_CATALOG_FAMILY)],
    },
    "SearchHit": {
        "repo": "packager/Nova-7B-GGUF",
        "publisher_status": "listed packager",
        "base_model": ["acme/Nova-7B"],
        "base_model_relation": "quantized",
        "resolved": True,
        "unresolved_reason": None,
        "resolved_base_model": "acme/Nova-7B",
        "repo_created_at": "2026-03-02T10:00:00Z",
        "parameters_b": 7.6,
        "license": "apache-2.0",
        "age": "legacy",
        "successor": "acme/Nova-7B-2512",
        "ollama": "nova:7b",
    },
    "Requirement": {
        "origin": "computed",
        "package_identity": ["huggingface", "packager/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf"],
        "scenario": {
            "context_requested": 8192,
            "context_origin": "default",
            "kv_type": "f16",
            "kv_type_assumed": True,
            "requests": 4,
        },
        "mode": "gpu",
        "weights_gib": 4.5,
        "kv_gib_per_request": 1.0,
        "kv_gib_total": 4.0,
        "reserve_gib": 1.0,
        "need_gib": 9.45,
        "perfect_reachable": True,
    },
    "Note": {
        "code": "cpu_caps_at_good",
        "subject": "package",
        "origin": "computed",
        "text": "Runs in system memory only, so the best possible rating on this machine is 'good'.",
        "facts": ["fit.mode", "fit.fit_class"],
    },
}
