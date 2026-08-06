use std::fs;
use std::path::PathBuf;

use serde_yml::Value;

fn manifest_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../deploy/kubernetes")
        .join(name)
}

fn parse_manifest(name: &str) -> Value {
    let path = manifest_path(name);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_yml::from_str(&contents).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn named<'a>(values: &'a Value, name: &str) -> &'a Value {
    values
        .as_sequence()
        .unwrap_or_else(|| panic!("expected sequence containing {name}"))
        .iter()
        .find(|value| value["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("missing entry named {name}"))
}

fn string_list(value: &Value) -> Vec<&str> {
    value
        .as_sequence()
        .expect("expected string sequence")
        .iter()
        .map(|entry| entry.as_str().expect("expected string entry"))
        .collect()
}

fn assert_string_map_is_subset(selector: &Value, labels: &Value) {
    let selector = selector.as_mapping().expect("selector must be a mapping");
    let labels = labels.as_mapping().expect("labels must be a mapping");
    for (key, value) in selector {
        assert_eq!(labels.get(key), Some(value), "selector entry must match pod labels");
    }
}

#[test]
fn kustomization_includes_only_non_secret_base_resources() {
    let manifest = parse_manifest("kustomization.yaml");
    let resources = string_list(&manifest["resources"]);

    assert_eq!(
        resources,
        [
            "namespace.yaml",
            "service-account.yaml",
            "configmap.yaml",
            "deployment.yaml",
            "service.yaml",
            "network-policy.yaml",
            "pod-disruption-budget.yaml",
        ]
    );
    assert!(!resources.contains(&"secret.example.yaml"));
    assert!(!resources.contains(&"ingress.example.yaml"));
    assert!(!resources.contains(&"network-policy-ingress.example.yaml"));
}

#[test]
fn deployment_is_replicated_hardened_and_probe_driven() {
    let manifest = parse_manifest("deployment.yaml");
    assert_eq!(manifest["apiVersion"].as_str(), Some("apps/v1"));
    assert_eq!(manifest["kind"].as_str(), Some("Deployment"));
    assert_eq!(manifest["spec"]["replicas"].as_u64(), Some(2));
    assert_eq!(
        manifest["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"].as_u64(),
        Some(0)
    );

    let pod_spec = &manifest["spec"]["template"]["spec"];
    assert_eq!(pod_spec["serviceAccountName"].as_str(), Some("agentic-api"));
    assert_eq!(pod_spec["automountServiceAccountToken"].as_bool(), Some(false));
    assert!(
        pod_spec["terminationGracePeriodSeconds"]
            .as_u64()
            .is_some_and(|seconds| seconds >= 10)
    );
    assert_eq!(
        pod_spec["securityContext"]["seccompProfile"]["type"].as_str(),
        Some("RuntimeDefault")
    );
    assert_eq!(pod_spec["securityContext"]["runAsNonRoot"].as_bool(), Some(true));
    assert!(pod_spec["securityContext"]["runAsUser"].is_null());
    assert_eq!(
        pod_spec["affinity"]["podAntiAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0]
            ["podAffinityTerm"]["topologyKey"]
            .as_str(),
        Some("kubernetes.io/hostname")
    );

    let container = named(&pod_spec["containers"], "agentic-api");
    assert_eq!(string_list(&container["args"]), ["--llm-ready-timeout-s", "300"]);
    assert_eq!(container["ports"][0]["name"].as_str(), Some("http"));
    assert_eq!(container["ports"][0]["containerPort"].as_u64(), Some(9000));
    let startup_budget_seconds = container["startupProbe"]["failureThreshold"]
        .as_u64()
        .zip(container["startupProbe"]["periodSeconds"].as_u64())
        .map(|(failures, period)| failures * period)
        .expect("startup probe must define a numeric failure threshold and period");
    assert!(startup_budget_seconds > 600);
    assert!(
        manifest["spec"]["progressDeadlineSeconds"]
            .as_u64()
            .is_some_and(|deadline| deadline >= startup_budget_seconds + 300),
        "deployment progress deadline must leave five minutes beyond the startup probe budget"
    );
    assert_eq!(
        container["envFrom"][0]["configMapRef"]["name"].as_str(),
        Some("agentic-api")
    );

    let database_url = named(&container["env"], "DATABASE_URL");
    assert_eq!(
        database_url["valueFrom"]["secretKeyRef"]["name"].as_str(),
        Some("agentic-api")
    );
    assert_eq!(
        database_url["valueFrom"]["secretKeyRef"]["key"].as_str(),
        Some("DATABASE_URL")
    );
    assert_ne!(
        database_url["valueFrom"]["secretKeyRef"]["optional"].as_bool(),
        Some(true)
    );

    let openai_api_key = named(&container["env"], "OPENAI_API_KEY");
    assert_eq!(
        openai_api_key["valueFrom"]["secretKeyRef"]["optional"].as_bool(),
        Some(true)
    );

    for (probe, path) in [
        ("startupProbe", "/health"),
        ("livenessProbe", "/health"),
        ("readinessProbe", "/ready"),
    ] {
        assert_eq!(container[probe]["httpGet"]["path"].as_str(), Some(path));
        assert_eq!(container[probe]["httpGet"]["port"].as_str(), Some("http"));
        assert!(
            container[probe]["timeoutSeconds"]
                .as_u64()
                .is_some_and(|seconds| seconds > 0)
        );
    }

    assert_eq!(
        container["securityContext"]["allowPrivilegeEscalation"].as_bool(),
        Some(false)
    );
    assert_eq!(
        container["securityContext"]["readOnlyRootFilesystem"].as_bool(),
        Some(true)
    );
    assert_eq!(
        container["securityContext"]["capabilities"]["drop"][0].as_str(),
        Some("ALL")
    );
    assert_eq!(
        string_list(&container["lifecycle"]["preStop"]["exec"]["command"]),
        ["/bin/sh", "-c", "sleep 5"]
    );

    for class in ["requests", "limits"] {
        assert!(container["resources"][class]["cpu"].is_string());
        assert!(container["resources"][class]["memory"].is_string());
    }
}

#[test]
fn service_config_and_disruption_budget_match_the_workload() {
    let deployment = parse_manifest("deployment.yaml");
    let workload_selector = &deployment["spec"]["selector"]["matchLabels"];
    let pod_labels = &deployment["spec"]["template"]["metadata"]["labels"];
    let service = parse_manifest("service.yaml");
    assert_eq!(service["spec"]["type"].as_str(), Some("ClusterIP"));
    assert_eq!(&service["spec"]["selector"], workload_selector);
    assert_string_map_is_subset(&service["spec"]["selector"], pod_labels);
    assert_eq!(service["spec"]["ports"][0]["targetPort"].as_str(), Some("http"));

    let config = parse_manifest("configmap.yaml");
    assert_eq!(config["data"]["GATEWAY_HOST"].as_str(), Some("0.0.0.0"));
    assert_eq!(config["data"]["GATEWAY_PORT"].as_str(), Some("9000"));
    assert_eq!(config["data"]["SKIP_LLM_READY_CHECK"].as_str(), Some("false"));
    assert!(
        config["data"]["CORS_ALLOWED_ORIGINS"]
            .as_str()
            .is_some_and(|origins| origins.starts_with("https://"))
    );
    assert!(
        config["data"]["LLM_API_BASE"]
            .as_str()
            .is_some_and(|url| url.starts_with("https://"))
    );
    assert!(config["data"]["DATABASE_URL"].is_null());
    assert!(config["data"]["OPENAI_API_KEY"].is_null());

    let budget = parse_manifest("pod-disruption-budget.yaml");
    assert_eq!(budget["apiVersion"].as_str(), Some("policy/v1"));
    assert_eq!(budget["spec"]["minAvailable"].as_u64(), Some(1));
    assert_eq!(&budget["spec"]["selector"]["matchLabels"], workload_selector);
    assert_string_map_is_subset(&budget["spec"]["selector"]["matchLabels"], pod_labels);
}

#[test]
fn secret_example_documents_required_credentials() {
    let secret = parse_manifest("secret.example.yaml");
    assert_eq!(secret["kind"].as_str(), Some("Secret"));
    assert_eq!(secret["metadata"]["name"].as_str(), Some("agentic-api"));
    assert!(
        secret["stringData"]["DATABASE_URL"]
            .as_str()
            .is_some_and(|url| url.starts_with("postgresql://") && url.contains("sslmode=verify-full"))
    );
    assert!(secret["stringData"]["OPENAI_API_KEY"].as_str().is_some());
}

#[test]
fn network_access_is_denied_by_default_and_ingress_is_opt_in() {
    let default_deny = parse_manifest("network-policy.yaml");
    assert_eq!(default_deny["apiVersion"].as_str(), Some("networking.k8s.io/v1"));
    assert_eq!(default_deny["kind"].as_str(), Some("NetworkPolicy"));
    let deployment = parse_manifest("deployment.yaml");
    let workload_selector = &deployment["spec"]["selector"]["matchLabels"];
    let pod_labels = &deployment["spec"]["template"]["metadata"]["labels"];
    assert_eq!(&default_deny["spec"]["podSelector"]["matchLabels"], workload_selector);
    assert_string_map_is_subset(&default_deny["spec"]["podSelector"]["matchLabels"], pod_labels);
    assert_eq!(string_list(&default_deny["spec"]["policyTypes"]), ["Ingress"]);
    assert!(default_deny["spec"]["ingress"].as_sequence().is_some_and(Vec::is_empty));

    let ingress_access = parse_manifest("network-policy-ingress.example.yaml");
    assert_eq!(&ingress_access["spec"]["podSelector"]["matchLabels"], workload_selector);
    assert_string_map_is_subset(&ingress_access["spec"]["podSelector"]["matchLabels"], pod_labels);
    let ingress_rule = &ingress_access["spec"]["ingress"][0];
    assert_eq!(
        ingress_rule["from"][0]["namespaceSelector"]["matchLabels"]["kubernetes.io/metadata.name"].as_str(),
        Some("ingress-nginx")
    );
    assert_eq!(
        ingress_rule["from"][0]["podSelector"]["matchLabels"]["app.kubernetes.io/name"].as_str(),
        Some("ingress-nginx")
    );
    assert_eq!(ingress_rule["ports"][0]["port"].as_u64(), Some(9000));
}
