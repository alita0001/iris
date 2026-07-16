# IRIS third-environment live feasibility audit

Audit date: 2026-07-16 UTC. This audit was read-only: it performed TCP/HTTP
connectivity checks and listed Docker containers. It did not log in or execute
any state-changing action.

## Measured deployment state

The following command was run after sourcing `scripts/export_webarena_env.sh`:

```bash
for port in 7770 7780 9999 8023 8888 3000 4399 7565; do
  timeout 2 bash -c "</dev/tcp/user2-dind/$port"
done
curl --noproxy '*' -L --max-time 8 <site-url>
DOCKER_HOST=tcp://user2-dind:2375 docker ps -a \
  --format '{{.Names}} {{.Image}} {{.Status}}'
```

Observed output:

```text
tcp user2-dind:7770 OPEN
tcp user2-dind:7780 OPEN
tcp user2-dind:9999 OPEN
tcp user2-dind:8023 REFUSED_OR_UNREACHABLE
tcp user2-dind:8888 REFUSED_OR_UNREACHABLE
tcp user2-dind:3000 REFUSED_OR_UNREACHABLE
tcp user2-dind:4399 REFUSED_OR_UNREACHABLE
tcp user2-dind:7565 REFUSED_OR_UNREACHABLE
http shopping 200
http shopping_admin 200
http reddit 200
http gitlab 000
http wikipedia 000
http map 000
forum postmill-populated-exposed-withimg:latest Up 2 days
shopping shopping_final_0712 Up 2 days
shopping_admin shopping_admin_final_0719 Up 2 days
```

The positive controls on ports 7770/7780/9999 rule out proxy or command-path
failure. There is no deployed third WebArena technology family in the current
Docker host. GitLab, Kiwix/Wikipedia, Map and the reset service are unavailable.

## Code-path audit

- `scripts/export_webarena_env.sh:20-28` explicitly calls GitLab, Wikipedia,
  Map and homepage values placeholders and leaves `WA_FULL_RESET` empty.
- `configs/default.yaml:7-13` configures only shopping, shopping-admin and
  reddit.
- `revact/config.py:182-193` registers only Magento and Postmill sites. Merely
  reading `WA_GITLAB` at `revact/config.py:136` does not register a site.
- `revact/cli.py:1222-1233` constrains point collection to shopping/reddit.
- The probe modules currently cover shopping, shopping-admin and reddit only.

Consequently, changing a frontend/provider setting cannot produce valid
third-environment evidence. A third environment must first be deployed with a
session bootstrap, reset/undo contract, site registry row, collector path,
probe specification, signals and point-level lineage.

## Safe first probes once infrastructure exists

1. GitLab project `star -> unstar`, verified by button state, star count and
   the user's starred-project list.
2. GitLab `follow -> unfollow`, with the same constructive recovery rule.
3. WorkArena isolated incident-field `edit -> restore original value`, verified
   through both UI and Table API, using per-task fixtures and teardown.

Wikipedia/Kiwix and Map can provide navigation/session-state controls, but not
persistent-backend mutation evidence. Branch deletion, issue/comment creation,
catalog orders and other externally visible writes are excluded from the first
batch.

## Readiness consequence

`research:independent_environment_families_ge_3` remains blocked by missing
external infrastructure, not by a configuration flag. No cross-site result may
be claimed from the current two technology families (Magento and Postmill).
