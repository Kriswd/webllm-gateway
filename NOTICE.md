# Notices

WebAI Gateway is released under the MIT License. It does not vendor the source
code of WebAI2API or ds2api in the public project tree; those projects are
treated as optional local runtimes started or contacted by the Gateway.

## Runtime dependencies

- WebAI2API sidecar: package metadata in the local sidecar declares MIT license
  and author `foxhui`. If you redistribute a bundled sidecar, include its own
  license and notices.
- ds2api: the local oracle checkout declares GNU AGPL-3.0. If you distribute,
  modify, host, or bundle ds2api, comply with AGPL-3.0 and provide the
  corresponding source as required by that license.

Before publishing a release package that bundles third-party binaries or
source, verify the upstream license files again and ship their notices
alongside the release.
