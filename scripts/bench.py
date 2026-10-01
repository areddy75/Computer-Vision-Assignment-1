"""Quick throughput check: data loading vs. GPU step time (used to pick num_workers / memory format)."""
import sys, time, torch, torch.nn.functional as F, yaml
from cvproj.data import build_datasets, make_loader
from cvproj.models import build_model
cfg = yaml.safe_load(open(sys.argv[1])); dev = torch.device("mps")
tr, _, _, cls = build_datasets(cfg["data"], "data")
for nw in [0]:
    dl = make_loader(tr, 32, True, nw, drop_last=True); it = iter(dl); next(it)
    t = time.time(); n = sum(1 for _ in it); print(f"loader nw={nw}: {(time.time()-t)/n*1000:.0f} ms/batch")
x = torch.randn(32, 3, cfg["data"]["img_size"], cfg["data"]["img_size"], device=dev); y = torch.randint(0, 16, (32,), device=dev)
for cl in [False, True]:
    m = build_model(cfg["model"], 16).to(dev); xx = x
    if cl: m = m.to(memory_format=torch.channels_last); xx = x.contiguous(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters())
    for i in range(25):
        if i == 5: torch.mps.synchronize(); t = time.time()
        loss = F.cross_entropy(m(xx), y); opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize(); print(f"channels_last={cl}: {(time.time()-t)/20*1000:.0f} ms/step")
