# Kubernetes deployment guide

## Prerequisites

- A Kubernetes cluster (1.26+) with:
  - [Gateway API CRDs](https://gateway-api.sigs.k8s.io/)
  - A maintained Gateway controller (for example Traefik, Kong, HAProxy, NGINX Gateway Fabric)
  - [cert-manager](https://cert-manager.io/) (for automated TLS)
  - Prometheus scraping enabled (for `/metrics`)
- A reachable Neo4j instance (in-cluster or external)
- `kubectl` connected to the target cluster
- The namespace running your Gateway controller labeled for network policy access:
  - `kubectl label namespace <gateway-namespace> networking.waggle.dev/gateway-access=true --overwrite`

> `ingress-nginx` reached retirement in March 2026 and no longer receives security updates.
> This deployment now uses Gateway API resources instead of `Ingress`.

---

## Apply order

Apply manifests in this exact order so dependencies (ConfigMap, Secret) are
present before the Deployment references them.

```bash
# 1. Network isolation
kubectl apply -f networkpolicy.yaml

# 2. Config and secrets (edit secret.example.yaml → secret.yaml first)
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml        # NOT secret.example.yaml

# 3. Workload
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 4. Autoscaling + disruption budget
kubectl apply -f hpa.yaml
kubectl apply -f pdb.yaml

# 5. TLS certificate (cert-manager must be installed)
kubectl apply -f certificate.yaml

# 6. Gateway API routing (edit gatewayClassName + hostname first)
kubectl apply -f gateway.yaml
kubectl apply -f httproute.yaml
```

---

## Prepare the secret

```bash
# Copy the example, fill real values, apply — never commit secret.yaml to git
cp secret.example.yaml secret.yaml
# Edit WAGGLE_NEO4J_USERNAME and WAGGLE_NEO4J_PASSWORD
kubectl apply -f secret.yaml
```

---

## Replace the image reference

`deployment.yaml` uses `waggle-mcp:latest` as a placeholder.
Push your image to a registry and update the field:

```bash
# Example using Docker Hub
docker build -t yourorg/waggle-mcp:v0.1.0 .
docker push yourorg/waggle-mcp:v0.1.0
# Then update deployment.yaml:  image: yourorg/waggle-mcp:v0.1.0
kubectl apply -f deployment.yaml
```

---

## Verify the deployment

```bash
# Watch pods come up
kubectl rollout status deployment/waggle

# Check health endpoints via port-forward
kubectl port-forward svc/waggle 8080:80
curl http://localhost:8080/health/ready
curl http://localhost:8080/health/live
curl http://localhost:8080/metrics

# Check HPA status after a few minutes
kubectl get hpa waggle
```

---

## Verify Gateway and TLS

```bash
# Confirm Gateway and routes are accepted by your controller
kubectl get gateway waggle-gateway
kubectl get httproute
kubectl describe gateway waggle-gateway

# After DNS is pointed at the Gateway address:
curl -v https://waggle.example.com/health/ready

# Check cert-manager issued the certificate
kubectl describe certificate waggle-tls
kubectl get certificaterequest
```

---

## Rollback

```bash
kubectl rollout undo deployment/waggle
# Or to a specific revision:
kubectl rollout undo deployment/waggle --to-revision=2
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Pods stuck in `Pending` | `kubectl describe pod <name>` → resource limits, node capacity |
| Readiness probe failing | `kubectl logs <pod>` — embedding model may still be downloading |
| TLS certificate not issued | `kubectl describe certificate waggle-tls` → cert-manager events |
| Gateway has no address | `kubectl describe gateway waggle-gateway` and verify `gatewayClassName` in `gateway.yaml` matches your installed controller |
| 401 from `/mcp` | Ensure `X-API-Key` header is set to a valid active key |
| Rate-limit 429 from `/mcp` | Adjust `WAGGLE_RATE_LIMIT_RPM` in configmap.yaml |
