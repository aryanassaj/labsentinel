#!/usr/bin/env python3
"""
Run this script after retraining to regenerate labsentinel.html with updated models.
Usage: python generate_html.py [--output path/to/labsentinel.html]
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = sys.argv[sys.argv.index('--output')+1] if '--output' in sys.argv else os.path.join(BASE,'labsentinel.html')

print(f"Loading models from {BASE}/models_saved/...")
models = {}
for name in ['model_any_cancer','model_breast','model_colorectal','model_lung',
             'model_prostate','model_leukemia','model_lymphoma']:
    with open(f'{BASE}/models_saved/{name}.json') as f:
        models[name] = json.load(f)
with open(f'{BASE}/models_saved/feature_cols.json') as f:
    feat_cols = json.load(f)
with open(f'{BASE}/models_saved/threshold.json') as f:
    td = json.load(f)

def slim(m):
    raw = m['learner']['gradient_booster']['model']['trees']
    return [{'lc':t['left_children'],'rc':t['right_children'],'sc':t['split_conditions'],
             'si':t['split_indices'],'bw':t['base_weights'],'dl':t['default_left']} for t in raw]

slim_models = {k: slim(v) for k, v in models.items()}
model_block = (f"const FEAT_COLS={json.dumps(feat_cols)};\n"
               f"const THRESHOLD={td['threshold']};\n"
               f"const SENSITIVITY={td['sensitivity']};\n"
               f"const SPECIFICITY={td['specificity']};\n"
               f"const MODELS={json.dumps(slim_models)};")

template_path = os.path.join(BASE, 'frontend', 'labsentinel_template.html')
with open(template_path) as f:
    template = f.read()

html = template.replace('/* __MODEL_DATA_PLACEHOLDER__ */', model_block)
with open(OUT, 'w') as f:
    f.write(html)

size = os.path.getsize(OUT)
print(f"Written: {OUT}")
print(f"Size: {size//1024} KB ({size/1024/1024:.2f} MB)")
tree_counts = {k: len(v['learner']['gradient_booster']['model']['trees']) for k,v in models.items()}
print("Tree counts:", tree_counts)
