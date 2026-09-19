"""Assemble the report page: inline thumbnails / figures as data URIs, fill the
atlas image slots and the blind-verification table."""
import base64, os, re, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read()
names = {int(k): v for k, v in json.load(open(os.path.join(HERE, "out", "cluster_names.json"))).items()}


def uri(path):
    ext = path.rsplit(".", 1)[-1].lower(); mime = "image/png" if ext == "png" else "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode())


# figures
for m in re.findall(r"\{\{IMG:([^}]+)\}\}", T):
    T = T.replace("{{IMG:%s}}" % m, uri(os.path.join(HERE, "thumbs", m)))

# atlas thumbnails: two per cluster, different runs where possible
s = pd.read_csv(os.path.join(HERE, "out", "snapshot_samples_v2.csv"))
lab_path = os.path.join(HERE, "out", "snapshot_blind_labels.csv")
blind = pd.read_csv(os.path.join(HERE, "out", "snapshot_blind.csv"))
labels = None
if os.path.exists(lab_path):
    labels = pd.read_csv(lab_path).merge(blind, on="id").merge(s, on="png")
for c in sorted(names):
    g = s[(s.new_cluster == c) & s.png.notna()]
    picks = []; seen = set()
    for _, r in g.iterrows():
        if r.tag in seen: continue
        picks.append(r); seen.add(r.tag)
        if len(picks) == 2: break
    for i, r in enumerate(picks):
        th = os.path.join(HERE, "thumbs", "c%02d_%s_pid%s_ep%s.jpg" % (c, r.tag, r.pid, r.ep))
        cap = "%s · %s steps · %s" % (r.tag, r.steps, "success" if r.success else "failed")
        if labels is not None:
            lr = labels[labels.png == r.png]
            if len(lr): cap += " · seen: %s%s" % (lr.primary.iloc[0], " (convoluted)" if str(lr.convoluted.iloc[0]).strip().lower().startswith("y") else "")
        T = T.replace("{{ATLAS:%d:%d}}" % (c, i), '<figure><img src="%s" alt="snapshot"><figcaption>%s</figcaption></figure>' % (uri(th), cap))
T = re.sub(r"\{\{ATLAS:\d+:\d+\}\}", "", T)

# verification table
if labels is not None:
    rows = []
    for c in sorted(names):
        g = labels[labels.new_cluster == c]
        if not len(g): continue
        vc = g.primary.value_counts()
        conv = g.convoluted.astype(str).str.strip().str.lower().str.startswith("y").mean() * 100
        prog = g.progress.value_counts().index[0] if len(g) else ""
        rows.append("<tr><td class=mono>%s</td><td class=num>%d</td><td>%s</td><td class=num>%.0f%%</td><td>%s</td></tr>" % (
            names[c], len(g), ", ".join("%s ×%d" % (k, v) for k, v in vc.items()), conv, prog))
    table = ('<table><thead><tr><th>cluster</th><th class=num>images</th><th>what the blind labeller saw (primary label ×count)</th>'
             '<th class=num>convoluted</th><th>modal progress</th></tr></thead><tbody>%s</tbody></table>' % "".join(rows))
    T = T.replace("{{VERIFY}}", table)
else:
    T = T.replace("{{VERIFY}}", "<p class=note>Blind labelling still running; table will appear on the next publish.</p>")

out = os.path.join(HERE, "rcca_trajectory_atlas.html")
open(out, "w", encoding="utf-8").write(T)
print(out, "%.2f MB" % (os.path.getsize(out) / 1e6))
