"""The validated example of every contract model, verbatim what `CONTRACTS.md` shows.

Kept apart from `contracts.py` so the contract module stays the models and their rules and
this module stays the data; `tests/test_contracts.py` validates every entry against its model
and checks that `CONTRACTS.md` repeats it unchanged.
"""

from __future__ import annotations

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
}
