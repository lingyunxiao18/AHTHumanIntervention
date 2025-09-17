import os, json, math, random, collections
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

MACROS = ["GO_TO_ONION","TAKE_ONION","GO_TO_POT","PUT_IN_POT",
          "GO_TO_DISH","TAKE_DISH","GO_TO_SERVE","SERVE","WAIT_COOK","TAKE_SOUP"]
MID = {m:i for i,m in enumerate(MACROS)}
GROUP = {
    "GO_TO_ONION":"NAV","GO_TO_POT":"NAV","GO_TO_DISH":"NAV","GO_TO_SERVE":"NAV",
    "TAKE_ONION":"ACT","TAKE_DISH":"ACT","PUT_IN_POT":"ACT","SERVE":"ACT","TAKE_SOUP":"ACT",
    "WAIT_COOK":"WAIT"
}

ARG_MACRO_POT = set(["GO_TO_POT"])  # expects pot_id
ARG_MACRO_ONION = set(["GO_TO_ONION"])  # expects onion_id
ARG_MACRO_SERVE = set(["GO_TO_SERVE"])  # expects serve_id

class MacroJsonl(Dataset):
    def __init__(self, paths, split="train"):
        self.rows = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    ex = json.loads(line)
                    if ("seed" not in ex): continue
                    is_val = (os.path.basename(p).endswith("_01.jsonl"))
                    if (split=="train" and is_val) or (split=="val" and not is_val):
                        continue
                    macro = ex.get("macro_id")
                    if macro not in MID:
                        continue
                    lm = ex.get("legal_macro_mask", [])
                    if len(lm) < len(MACROS):
                        lm = lm + [0]*(len(MACROS)-len(lm))
                    if lm[MID[macro]] == 0 or not ex.get("oracle_success", True):
                        continue
                    ex["legal_macro_mask"] = lm
                    args = ex.get("macro_args", {})
                    ex["pot_id"] = int(args.get("pot_id", -1))
                    ex["pot_mask"] = args.get("pot_mask", [1])
                    ex["onion_id"] = int(args.get("onion_id", -1))
                    ex["onion_mask"] = args.get("onion_mask", [1])
                    ex["serve_id"] = int(args.get("serve_id", -1))
                    ex["serve_mask"] = args.get("serve_mask", [1])
                    self.rows.append(ex)
        # infer arg vocab sizes
        self.K_pot = max([len(r["pot_mask"]) for r in self.rows] + [1])
        self.K_onion = max([len(r["onion_mask"]) for r in self.rows] + [1])
        self.K_serve = max([len(r["serve_mask"]) for r in self.rows] + [1])

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ex = self.rows[i]
        return (
            ex["text"],
            MID[ex["macro_id"]],
            torch.tensor(ex["legal_macro_mask"], dtype=torch.float32),
            torch.tensor(ex["pot_id"], dtype=torch.long),
            torch.tensor(ex["pot_mask"], dtype=torch.float32),
            torch.tensor(ex["onion_id"], dtype=torch.long),
            torch.tensor(ex["onion_mask"], dtype=torch.float32),
            torch.tensor(ex["serve_id"], dtype=torch.long),
            torch.tensor(ex["serve_mask"], dtype=torch.float32),
        )

def make_sampler(ds):
    grp_cnt = collections.Counter(GROUP[MACROS[y]] for _,y,_,_,_,_,_,_,_ in ds)
    weights = []
    for _,y,_,_,_,_,_,_,_ in ds:
        g = GROUP[MACROS[y]]
        weights.append((grp_cnt[g])**-0.5)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.05, class_weights=None):
        super().__init__(); self.eps=eps; self.class_weights=class_weights
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        if self.class_weights is not None:
            w = self.class_weights[target]
            nll = nll*w; smooth = smooth*w
        return ((1-self.eps)*nll + self.eps*smooth).mean()

def class_weights(ds):
    cnt = collections.Counter(y for _,y,_,_,_,_,_,_,_ in ds)
    freq = torch.tensor([cnt[i] for i in range(len(MACROS))], dtype=torch.float32)
    return (freq.sum() / (len(MACROS)*freq)).clamp(max=10.0)

class MacroPolicy(nn.Module):
    def __init__(self, enc="distilbert-base-uncased", proj_dim=256, p_drop=0.1, K_pot=2, K_onion=2, K_serve=1):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc)
        self.enc = AutoModel.from_pretrained(enc)
        for p in self.enc.parameters(): p.requires_grad = False
        h = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, proj_dim), nn.GELU(), nn.Dropout(p_drop))
        self.head_macro = nn.Linear(proj_dim, len(MACROS))
        self.head_pot   = nn.Linear(proj_dim, K_pot)
        self.head_onion = nn.Linear(proj_dim, K_onion)
        self.head_serve = nn.Linear(proj_dim, K_serve) if K_serve > 1 else None
        self.head_legal = nn.Linear(proj_dim, len(MACROS))
        self.K_serve = K_serve
    def forward(self, texts, max_len=256):
        device = self.head_macro.weight.device
        batch = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        for k in batch: batch[k] = batch[k].to(device)
        H = self.enc(**batch).last_hidden_state.mean(1)
        z = self.proj(H)
        logits_serve = self.head_serve(z) if self.head_serve is not None else None
        return self.head_macro(z), self.head_pot(z), self.head_onion(z), logits_serve, self.head_legal(z)

def masked_argmax(logits, legal_mask):
    return logits.masked_fill(legal_mask==0, float("-inf")).argmax(dim=-1)

def summarize(ds, name):
    cnt = collections.Counter()
    for _,y,_,_,_,_,_,_,_ in ds:
        cnt[MACROS[y]] += 1
    print(f"[{name}] size={len(ds)} dist={dict(cnt)}")

def masked_ce(logits, target_idx, mask):
    big_neg = torch.finfo(logits.dtype).min / 2
    masked_logits = logits.masked_fill(mask == 0, big_neg)
    return F.cross_entropy(masked_logits, target_idx)

def eval_epoch(model, loader, device):
    model.eval(); n=0; correct=0; illegal_raw=0
    per_macro = collections.Counter(); per_macro_correct = collections.Counter()
    per_macro_illegal = collections.Counter(); per_macro_count = collections.Counter()
    with torch.no_grad():
        for texts, y, mask, pot_id, pot_mask, onion_id, onion_mask, serve_id, serve_mask in loader:
            logits_m, _, _, _, _ = model(list(texts))
            logits_m = logits_m.to(device)
            mask = mask.to(device); y = y.to(device)
            pred = masked_argmax(logits_m, mask)
            correct += (pred==y).sum().item()
            raw = logits_m.argmax(dim=-1)
            illegal_flags = (mask.gather(1, raw.unsqueeze(1)).squeeze(1)==0)
            illegal_raw += illegal_flags.sum().item()
            for i in range(len(MACROS)):
                sel = (raw==i)
                if sel.any():
                    per_macro_illegal[i] += illegal_flags[sel].float().sum().item()
                    per_macro_count[i] += sel.float().sum().item()
            for yi in y.tolist(): per_macro[yi]+=1
            for pi,yi in zip(pred.tolist(), y.tolist()):
                if pi==yi: per_macro_correct[yi]+=1
            n += y.numel()
    macro_accs = {MACROS[i]: (per_macro_correct[i]/per_macro[i] if per_macro[i] else 0.0) for i in range(len(MACROS))}
    illegal_rate = illegal_raw/n
    # print per-macro illegal@raw summary
    print("illegal@raw per-macro:")
    for i in range(len(MACROS)):
        denom = per_macro_count[i] if per_macro_count[i]>0 else 1
        rate = per_macro_illegal[i]/denom
        print(f"  {MACROS[i]:12s} {rate:.2f}")
    return dict(acc=correct/n, illegal_raw=illegal_rate, per_macro=macro_accs)

def train_one(train_path="../pretrain_shards_balanced/macro_pretrain_00.jsonl", val_path="../pretrain_shards_balanced/macro_pretrain_01.jsonl", outdir="../trained_policies/macro_policy_balanced"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tr = MacroJsonl([train_path], split="train"); va = MacroJsonl([val_path], split="val")
    summarize(tr, "train"); summarize(va, "val")
    print("train/val sizes:", len(tr), len(va))
    tl = DataLoader(tr, batch_size=64, sampler=make_sampler(tr), num_workers=0)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    model = MacroPolicy(K_pot=tr.K_pot, K_onion=tr.K_onion, K_serve=tr.K_serve).to(device)
    cls_w = class_weights(tr).to(device)
    
    # Balanced class weight adjustments - more conservative
    print(f"Original class weights: {cls_w}")
    
    # Moderate adjustments to avoid overcorrection
    cls_w[MID["TAKE_ONION"]] = cls_w[MID["TAKE_ONION"]] * 0.8  # Slightly reduce TAKE_ONION
    cls_w[MID["GO_TO_SERVE"]] = cls_w[MID["GO_TO_SERVE"]] * 0.9  # Slightly reduce GO_TO_SERVE
    cls_w[MID["SERVE"]] = cls_w[MID["SERVE"]] * 0.9  # Slightly reduce SERVE
    
    # Moderate boosts for under-represented classes
    cls_w[MID["GO_TO_ONION"]] = cls_w[MID["GO_TO_ONION"]] * 1.2  # Moderate boost GO_TO_ONION
    cls_w[MID["GO_TO_POT"]] = cls_w[MID["GO_TO_POT"]] * 1.1  # Moderate boost GO_TO_POT
    cls_w[MID["WAIT_COOK"]] = cls_w[MID["WAIT_COOK"]] * 1.1  # SMALL boost WAIT_COOK (was too high)
    cls_w[MID["TAKE_DISH"]] = cls_w[MID["TAKE_DISH"]] * 1.1  # Moderate boost TAKE_DISH
    
    print(f"Adjusted class weights: {cls_w}")
    
    crit_macro = LabelSmoothingCE(0.05, class_weights=cls_w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = 0.0; best_state=None; patience=5  # Increased patience for better convergence
    for epoch in range(15):  # More epochs for better learning
        model.train()
        for texts, y, mask, pot_id, pot_mask, onion_id, onion_mask, serve_id, serve_mask in tl:
            logits_m, logits_pot, logits_onion, logits_serve, logits_legal = model(list(texts))
            y = y.to(device); mask = mask.to(device)
            pot_id = pot_id.to(device); pot_mask = pot_mask.to(device)
            onion_id = onion_id.to(device); onion_mask = onion_mask.to(device)
            serve_id = serve_id.to(device); serve_mask = serve_mask.to(device)
            # soft illegal logit nudge before CE
            logits_m_for_ce = logits_m - 1.0*(1.0 - mask)
            # macro loss + stronger illegal prob penalty + entropy bonus
            p = torch.softmax(logits_m, dim=-1)
            illegal_prob = (p * (1.0 - mask)).sum(dim=-1)
            loss = crit_macro(logits_m_for_ce, y) + 0.35*illegal_prob.mean() - 0.01*(p*torch.log(p+1e-9)).sum(dim=-1).mean()
            # legal-uniform KL
            legal_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            q = mask / legal_counts
            log_p = torch.log_softmax(logits_m, dim=-1)
            kl = (q * (torch.log(q + 1e-9) - log_p)).sum(dim=-1)
            loss = loss + 0.10 * kl.mean()
            # legality head BCE
            bce = F.binary_cross_entropy_with_logits(logits_legal, mask)
            loss = loss + 0.2 * bce
            # Skip argument losses for now - focus on macro classification
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        m = eval_epoch(model, vl, device)
        print(f"epoch {epoch}: acc={m['acc']:.3f} illegal@raw={m['illegal_raw']:.3f}")
        if m['acc'] > best + 1e-3:
            best, best_state, patience = m['acc'], {k:v.cpu() for k,v in model.state_dict().items()}, 3
        else:
            patience -= 1
            if patience < 0: break
    os.makedirs(outdir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(), os.path.join(outdir, "macro_policy.pt"))
    print("best val acc:", best)
    pm = eval_epoch(model, vl, device)['per_macro']
    print("per-macro val acc:")
    for k,v in pm.items(): print(f"  {k:12s} {v:.3f}")

if __name__ == "__main__":
    train_one()
