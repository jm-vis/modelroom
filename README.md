# ModelRoom

Does this model have room on your machine?

ModelRoom (package and command: `modelroom`) answers two questions for a list of model
families you care about:

1. **Which local packages exist?** It reads the Hugging Face API and the Ollama registry for
   the base models and packagers you allow, and records every package with its size, format,
   quantization and the exact revision it was seen at.
2. **Which of them fit a given machine?** It measures a machine with
   [llmfit](https://github.com/AlexsJones/llmfit) and computes, per package and context
   length, whether weights plus KV cache plus a configured reserve fit into the memory that
   machine actually has.

The result is a JSON snapshot you can commit and a rendered Markdown table with one fit
column per configured machine. Nothing in the tool is tied to a specific operator: you bring
a configuration file with your families, packagers and machines, run it on the machine you
want to size, and keep the snapshot wherever you keep your notes.

## Status

Pre-alpha. The repository holds the scaffold, the contract discipline and the hygiene tests.
Contracts, fetchers, the hardware probe and the renderer follow in that order; see
`CHANGELOG.md` for what has landed.

## Planned usage

```bash
uv tool install modelroom      # or: pipx install modelroom
llmfit --version               # 1.1.16 or newer is required for the hardware probe

modelroom hardware --machine laptop --config modelroom.toml
modelroom fetch    --config modelroom.toml
modelroom render   --config modelroom.toml
modelroom check    --config modelroom.toml
```

The configuration file names your model families with their exact base models, the
packagers you trust, your machines with their memory reserves, and where the snapshot and
the rendered table go. Working offline is a first-class case: a snapshot brought from
elsewhere is a pure catalog without any machine state, and `hardware` adds the local
measurement on site.

## Language choice

Python, because the catalog logic, canonicalisation and Pydantic contracts it has to
integrate with are Python, and the run has to be platform neutral (Windows, Linux, macOS).
Everything in this repository, from identifiers to documentation, is English.

## Contributing and conventions

`AGENTS.md` is the tool-neutral rule set for anyone (human or agent) working on this
repository. `CONTRACTS.md` describes the data shapes and their versioning. Issues and pull
requests are welcome.

## About

ModelRoom is built and maintained by [VISCONSULT](https://vis-consult.eu), a consultancy
in Germany that runs AI agents for its own work and for clients under EU data-protection
rules. We use it to decide which models run on our own laptops and servers, and with clients
to size on-premise deployments before anyone buys hardware.

## License

MIT, see [LICENSE](LICENSE).
