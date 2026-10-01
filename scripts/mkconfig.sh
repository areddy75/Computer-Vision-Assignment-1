# Helper used to derive a config from another with overrides: mk <new> <base> "<comment>" key=val ...
mk () {
	name=$1 
	base=$2 
	comment=$3 
	shift 3
	.venv/bin/python - "$name" "$base" "$comment" "$@" <<'EOF'
import sys, yaml
name, base, comment, *ov = sys.argv[1:]
cfg = yaml.safe_load(open(f"configs/{base}.yaml")); cfg["name"] = name
for item in ov:
    k, v = item.split("=", 1); d = cfg; ps = k.split(".")
    for p in ps[:-1]: d = d.setdefault(p, {})
    d[ps[-1]] = yaml.safe_load(v)
with open(f"configs/{name}.yaml", "w") as f:
    f.write(f"# {comment}\n"); yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
EOF
}
