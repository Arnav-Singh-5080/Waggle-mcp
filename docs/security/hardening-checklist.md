# Hardening Checklist

Use this checklist before exposing a self-hosted Waggle deployment to production traffic.

## Network

- [ ] Serve all external traffic over HTTPS.
- [ ] Put Waggle behind a reverse proxy.
- [ ] Restrict Neo4j to private network access.
- [ ] Do not publish database ports to the public internet.
- [ ] Configure firewall rules for `80/443` only.
- [ ] Forward secure proxy headers such as `X-Forwarded-Proto`.

## Secrets

- [ ] Change all default passwords.
- [ ] Store secrets in a secret manager.
- [ ] Rotate Waggle API keys regularly.
- [ ] Revoke unused API keys.
- [ ] Do not commit `.env` files or local secrets.

## Neo4j

- [ ] Enable Neo4j authentication.
- [ ] Use a strong Neo4j password.
- [ ] Restrict Bolt port access.
- [ ] Configure backups.
- [ ] Configure memory limits explicitly.
- [ ] Monitor disk usage and database growth.

## Application

- [ ] Set production environment values explicitly.
- [ ] Disable ad hoc debug-only exposure patterns.
- [ ] Configure request size limits.
- [ ] Configure rate limits.
- [ ] Enable application logging and health checks.
- [ ] Keep the app bound to private networking behind the proxy.

## Data lifecycle

- [ ] Define a retention period before production use.
- [ ] Test data-pruning procedures in staging.
- [ ] Document the deletion process for transcripts and exports.
- [ ] Review `.abhi` export handling for sensitive data.
- [ ] Plan separate retention treatment for future audit logs.

## Identity and access

- [ ] Use least-privilege API keys where possible.
- [ ] Revoke old keys after rotation.
- [ ] Review admin access regularly.

## Monitoring and response

- [ ] Monitor application errors and readiness probes.
- [ ] Monitor failed authentication attempts.
- [ ] Monitor export activity.
- [ ] Monitor backup success and failure.
- [ ] Document the incident-response owner.
- [ ] Test restore procedures before production rollout.
