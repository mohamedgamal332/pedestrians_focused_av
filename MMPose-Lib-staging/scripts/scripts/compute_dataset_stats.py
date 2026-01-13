from pathlib import Path
import json
import numpy as np

ROOT = Path("hrnet_depth")
files = list(ROOT.rglob("*_states.json")) + list(ROOT.rglob("*_states.json"))
files = [f for f in files if f.is_file()]

all_centers = []
all_deltas = []
count_people = 0

for f in files:
    try:
        data = json.loads(f.read_text())
    except Exception:
        continue
    # group by person
    persons = {}
    for frame in data:
        fi = frame.get('frame_index')
        for p in frame.get('people', []):
            pid = p.get('person_id')
            state = p.get('state')
            if pid is None or state is None:
                continue
            persons.setdefault(pid, []).append((fi, state))
    # compute per-person centers and deltas
    for pid, seq in persons.items():
        seq.sort(key=lambda x: x[0])
        centers = []
        frames = []
        for fi, s in seq:
            Cx, Cy, Cz = s[0], s[1], s[2]
            if Cx is None or Cy is None or Cz is None:
                centers.append(None)
            else:
                centers.append((float(Cx), float(Cy), float(Cz)))
            frames.append(fi)
        # collect centers
        for c in centers:
            if c is not None:
                all_centers.append(c)
                count_people += 1
        # compute deltas where consecutive frames
        for i in range(1, len(centers)):
            if centers[i] is None or centers[i-1] is None:
                continue
            if frames[i] != frames[i-1] + 1:
                continue
            Ca = centers[i-1]
            Cb = centers[i]
            dx = Cb[0] - Ca[0]
            dy = Cb[1] - Ca[1]
            dz = Cb[2] - Ca[2]
            all_deltas.append((dx, dy, dz))

if not all_centers:
    print('No centers found.')
    raise SystemExit(0)

C = np.array(all_centers)
D = np.array(all_deltas) if all_deltas else np.zeros((0,3))

print(f'Files scanned: {len(files)}')
print(f'Center samples: {len(C)}')
print('Center stats (Cx, Cy, Cz):')
print('  mean:', C.mean(axis=0))
print('  std :', C.std(axis=0))
print('  min :', C.min(axis=0))
print('  max :', C.max(axis=0))

print('\nDelta samples (per-frame differences) :', len(D))
if len(D)>0:
    print('Delta stats (dx, dy, dz):')
    print('  mean:', D.mean(axis=0))
    print('  std :', D.std(axis=0))
    print('  min :', D.min(axis=0))
    print('  max :', D.max(axis=0))
else:
    print('No deltas available (not enough consecutive frames).')

# quick percentiles
print('\nCenter percentiles (10,50,90) for Cx, Cy, Cz:')
for i, name in enumerate(['Cx','Cy','Cz']):
    p10 = np.percentile(C[:,i], 10)
    p50 = np.percentile(C[:,i], 50)
    p90 = np.percentile(C[:,i], 90)
    print(f'  {name}: p10={p10:.2f}, p50={p50:.2f}, p90={p90:.2f}')

if len(D)>0:
    print('\nDelta percentiles (10,50,90) for dx,dy,dz:')
    for i, name in enumerate(['dx','dy','dz']):
        p10 = np.percentile(D[:,i], 10)
        p50 = np.percentile(D[:,i], 50)
        p90 = np.percentile(D[:,i], 90)
        print(f'  {name}: p10={p10:.3f}, p50={p50:.3f}, p90={p90:.3f}')

print('\nSuggested normalization:')
print('- Divide x by image width and y by image height to get [0,1] range (if pixel coords).')
print('- Consider converting depths Cz to meters or normalizing by its std.')
print('- Current delta std:', D.std(axis=0) if len(D)>0 else 'N/A')
