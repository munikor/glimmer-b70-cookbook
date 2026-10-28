# Security Policy

This repo documents a self-hosted inference rig; the scripts are benchmark
utilities intended to run against a localhost OpenAI-compatible endpoint.

## Supported versions
| Version | Supported |
|---|---|
| main  | yes |

## Reporting a vulnerability
Open a private advisory via the Security tab -> "Report a vulnerability",
or open an issue if the finding is not exploitable in normal use. Expect a
response within 7 days.

These scripts handle no credentials, accept no network input, and ship no
daemon; anything that asks you to paste a token into them is out of scope
by design.
