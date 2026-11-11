"""Step 1: size-aware operating rule on frozen V10 (no training, VALIDATION only).

Question: V9's autopsy found 17.7% of missed flames are found by the model but scored below .40. Does a lower
threshold for small boxes only (longest side < S px at the 640 input, score >= t) recover small flames without
raising false alarms? Grid and limits are fixed in config_v12.json / common12.best_rule before running.
Output: opcheck_v12/v10.json (every grid cell) and opcheck_v12/v10.md.
"""
from common import write_json
from common12 import V10_FROZEN, V10_FROZEN_SHA, WORK, measure, best_rule, pct
from common import sha256


def table(result):
    L = ['| rule | precision | recall | FP boxes | small-flame recall | nuisance | within limits |', '|---|---|---|---|---|---|---|']
    b = result['base']
    L.append(f'| .40 for all | {pct(b["precision"])} | {pct(b["recall"])} | {b["FP"]} | {pct(b["small_recall"])} | '
             f'{b["nuisance"]["score"]:.4f} | reference |')
    for r in result['grid']:
        L.append(f'| side < {r["side"]} px: {r["threshold"]:.2f} | {pct(r["precision"])} | {pct(r["recall"])} | {r["FP"]} | '
                 f'{pct(r["small_recall"])} | {r["nuisance"]["score"]:.4f} | {"yes" if r["eligible"] else "no"} |')
    return L


def main():
    if sha256(V10_FROZEN) != V10_FROZEN_SHA:
        raise RuntimeError('frozen V10 checksum mismatch')
    v10 = measure(V10_FROZEN)
    result = best_rule(v10['sha256'], v10)
    out = WORK / 'opcheck_v12'
    write_json(dict(model='v10', reference=v10, **result), out / 'v10.json')
    L = ['# Step 1: size-aware threshold on frozen V10 (validation only, measured)', '',
         f'Chosen rule: ' + ('keep .40 for all sizes (no grid cell helped within limits)' if result['side'] == 0 else
                             f'boxes with longest side < {result["side"]} px at 640 input count from score {result["threshold"]:.2f}; '
                             'all others from .40'), ''] + table(result)
    (out / 'v10.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
