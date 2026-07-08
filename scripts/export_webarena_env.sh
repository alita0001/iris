#!/usr/bin/env bash
# Export the WebArena base-url env vars that BrowserGym / the probes require.
# Source this in every shell before any live-mode command:
#   source scripts/export_webarena_env.sh
#
# The Docker host serving the WebArena mirrors; override for other deployments:
#   WA_HOST=my-host source scripts/export_webarena_env.sh
WA_HOST="${WA_HOST:-user2-dind}"

# The mirror host must bypass any shell proxy, otherwise the first request
# is hijacked by the proxy and returns 502 (see docs/findings-multisite.md).
export NO_PROXY="${WA_HOST},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# Deployed sites (see SITES registry in revact/config.py).
export WA_SHOPPING="http://${WA_HOST}:7770"
export WA_SHOPPING_ADMIN="http://${WA_HOST}:7780/admin"
export WA_REDDIT="http://${WA_HOST}:9999"

# Placeholders for the rest of the full WebArena deployment. Keep them defined
# because BrowserGym checks that all WA_* variables exist.
export WA_GITLAB="http://${WA_HOST}:8023"
export WA_WIKIPEDIA="http://${WA_HOST}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${WA_HOST}:3000"
export WA_HOMEPAGE="http://${WA_HOST}:4399"

# Empty means BrowserGym will not call a full reset endpoint.
export WA_FULL_RESET=""

echo "WebArena env exported."
echo "WA_SHOPPING=$WA_SHOPPING"
echo "WA_SHOPPING_ADMIN=$WA_SHOPPING_ADMIN"
echo "WA_REDDIT=$WA_REDDIT"
echo "NO_PROXY=$NO_PROXY"
