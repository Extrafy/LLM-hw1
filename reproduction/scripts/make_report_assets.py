import json, csv
from pathlib import Path
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]; out=root/'reproduction'
rows=[]
def get(name):
 p=out/'results'/name
 return json.loads(p.read_text())['ppl']
for label,file,ref in [('Baseline','baseline_2048.json',5.47),('Uniform G-LRD','uniform_2048.json',7.36),('M-LRD (gs=1)','g1_r50_2048.json',6.75),('G-LRD (gs=4)','g4_r50_2048.json',6.01),('J-LRD (gs=32)','g32_r50_2048.json',5.62),('Fisher G-LRD 30% compression','g4_r70_2048.json',5.62)]:
 rows.append({'method':label,'reproduced_ppl':get(file),'paper_ppl':ref,'abs_diff':get(file)-ref,'result_file':file})
with (out/'results'/'ppl_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader();w.writerows(rows)
# Main comparison plot
fig,ax=plt.subplots(figsize=(10,5.5)); x=range(len(rows)); width=.36
ax.bar([i-width/2 for i in x],[r['paper_ppl'] for r in rows],width,label='Paper',color='#b8c7e0')
ax.bar([i+width/2 for i in x],[r['reproduced_ppl'] for r in rows],width,label='Reproduced',color='#3478b9')
ax.set_xticks(list(x),[r['method'] for r in rows],rotation=25,ha='right');ax.set_ylabel('WikiText-2 perplexity (lower is better)');ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'figures/ppl_comparison.png',dpi=180);fig.savefig(out/'figures/ppl_comparison.pdf');plt.close(fig)
# Quantization ablation
q=[('FP16',5.6330255,5.63),('3-bit no WHT',5.979022,5.99),('3-bit WHT',5.760711,5.77),('2-bit no WHT',19.573791,10.58),('2-bit WHT',6.994515,6.41)]
fig,ax=plt.subplots(figsize=(8,4.5)); xx=range(len(q));ax.bar([i-.18 for i in xx],[r[1] for r in q],.36,label='Reproduced',color='#3478b9');ax.bar([i+.18 for i in xx],[r[2] for r in q],.36,label='Paper',color='#b8c7e0');ax.set_xticks(list(xx),[r[0] for r in q],rotation=20,ha='right');ax.set_ylabel('WikiText-2 perplexity, seqlen=4096');ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'figures/quantization_ablation.png',dpi=180);fig.savefig(out/'figures/quantization_ablation.pdf');plt.close(fig)
# Fisher rank profile
fs=json.loads((out/'results/fisher_scores.json').read_text())['scores']; layers=range(32); k=[fs[f'model.layers.{i}.self_attn.k_proj'] for i in layers];v=[fs[f'model.layers.{i}.self_attn.v_proj'] for i in layers];fig,ax=plt.subplots(figsize=(9,4));ax.plot(layers,k,'o-',label='K projection');ax.plot(layers,v,'o-',label='V projection');ax.set_xlabel('Transformer layer');ax.set_ylabel('Mean Fisher score');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'figures/fisher_profile.png',dpi=180);plt.close(fig)
print('assets written')
