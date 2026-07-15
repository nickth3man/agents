# AGENTS.md

This repo contains multiple chat agents, each in its own language directory.

## What this is

A collection of chat clients. Each implementation is standalone and lives in a directory named after its language.

## Where to look

- **`.env`** at the repo root holds shared credentials (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`). Every agent reads it from `../.env` relative to its own directory.
- **Each language directory** has its own `README.md` with language, prerequisites, and the run command.
- **Root `README.md`** lists all implementations and links to each.

## Running

Agents are always run from their own directory. See the `README.md` inside each language folder for prerequisites and the exact command.
