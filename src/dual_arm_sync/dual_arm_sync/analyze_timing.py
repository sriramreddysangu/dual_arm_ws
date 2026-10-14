#!/usr/bin/env python3

import csv
import json
import statistics as st
import sys

CATEGORIES = ['NO_COLLISION', 'KURAMOTO', 'RETRACTION', 'UNRESOLVED',
             'IK_FAIL', 'STAGE_ERROR', 'START_COLLIDING']

COMPUTE_FIELDS = [
    ('t_ik_s',             'Stage 1: IK + endpoint CSP'),
    ('t_seed_s',           'Stage 2: B-spline seed'),
    ('t_scan_s',           'Stage 3: interference scan'),
    ('t_kuramoto_s',       'Stage 4: Kuramoto-alone'),
    ('t_escape_s',         'Stage 5: escape + re-sync'),
    ('t_planning_total_s', 'TOTAL planning time'),
]


def load_rows(path):
    if path.endswith('.json'):
        with open(path) as fh:
            data = json.load(fh)
        return data['trials']
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            for k in list(r.keys()):
                v = r[k]
                if v in ('', None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def mean_of(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return None
    return st.mean(vals)


def stdev_of(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    if len(vals) < 2:
        return None
    return st.stdev(vals)


def fmt(x, unit='s', nd=3):
    if x is None:
        return '--'
    return f'{x:.{nd}f}{unit}'


def report_one(path, label):
    rows = load_rows(path)
    n = len(rows)
    print(f'\n{"="*78}')
    print(f'  {label}   ({path}, n={n} trials)')
    print(f'{"="*78}')

    by_cat = {c: [r for r in rows if r.get('category') == c] for c in CATEGORIES}
    by_cat = {c: v for c, v in by_cat.items() if v}

    print(f'\n  {"Category":16s} {"n":>4s} {"%":>6s}   '
          f'{"mean exec time":>16s}   {"mean planning time":>20s}')
    print(f'  {"-"*16} {"-"*4} {"-"*6}   {"-"*16}   {"-"*20}')
    for c, cr in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        pct = 100 * len(cr) / n
        exec_mean = mean_of(cr, 'duration_s')
        plan_mean = mean_of(cr, 't_planning_total_s')
        print(f'  {c:16s} {len(cr):4d} {pct:5.1f}%   '
              f'{fmt(exec_mean):>16s}   {fmt(plan_mean):>20s}')

    print(f'\n  --- computation time breakdown by stage, per category (mean, seconds) ---')
    print(f'  {"Category":16s}' + ''.join(f'{s.split(":")[0]:>13s}'
                                         for _, s in COMPUTE_FIELDS))
    for c, cr in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        line = f'  {c:16s}'
        for field, _ in COMPUTE_FIELDS:
            m = mean_of(cr, field)
            line += f'{fmt(m, "", 2):>13s}'
        print(line)

    print(f'\n  --- execution time / overhead, KURAMOTO and RETRACTION only ---')
    for c in ('KURAMOTO', 'RETRACTION'):
        cr = by_cat.get(c, [])
        if not cr:
            continue
        dm = mean_of(cr, 'duration_s'); ds = stdev_of(cr, 'duration_s')
        om = mean_of(cr, 'overhead')
        print(f'  {c:12s} n={len(cr):3d}  mean duration={fmt(dm)}  '
              f'stdev={fmt(ds)}  mean overhead={fmt(om*100 if om is not None else None, "%", 1)}')

    n_plan_ok = sum(1 for r in rows if r.get('category') in
                    ('NO_COLLISION', 'KURAMOTO', 'RETRACTION'))
    n_e2e_ok = sum(1 for r in rows if r.get('exec_success') in (True, 'True', 1, 1.0))
    print(f'\n  Planning success   : {n_plan_ok}/{n} ({100*n_plan_ok/n:.1f}%)')
    print(f'  End-to-end success : {n_e2e_ok}/{n} ({100*n_e2e_ok/n:.1f}%)')

    return rows, by_cat


def latex_table(all_results):
    """Emit a LaTeX table matching the paper's Table format, ready to paste."""
    print(f'\n{"="*78}')
    print('  LaTeX table (paste into the paper)')
    print(f'{"="*78}\n')
    print(r'\begin{table}[!t]')
    print(r'\caption{Computation and Execution Time by Ensemble Size}')
    print(r'\label{tab:timing}')
    print(r'\centering')
    print(r'\begin{tabular}{lccc}')
    print(r'\hline')
    print('Quantity & ' + ' & '.join(f'{n} arms' for _, n, _ in all_results) + r' \\')
    print(r'\hline')

    rows_by_label = {}
    for label, n, (rows, by_cat) in all_results:
        rows_by_label[n] = (rows, by_cat)

    def line(name, fn):
        vals = []
        for _, n, _ in all_results:
            rows, by_cat = rows_by_label[n]
            vals.append(fn(rows, by_cat))
        print(f'{name} & ' + ' & '.join(vals) + r' \\')

    line('Mean planning time, NO\\_COLLISION',
        lambda rows, bc: fmt(mean_of(bc.get('NO_COLLISION', []), 't_planning_total_s'), '', 2))
    line('Mean planning time, KURAMOTO',
        lambda rows, bc: fmt(mean_of(bc.get('KURAMOTO', []), 't_planning_total_s'), '', 2))
    line('Mean planning time, RETRACTION',
        lambda rows, bc: fmt(mean_of(bc.get('RETRACTION', []), 't_planning_total_s'), '', 2))
    line('Mean planning time, UNRESOLVED',
        lambda rows, bc: fmt(mean_of(bc.get('UNRESOLVED', []), 't_planning_total_s'), '', 2))
    line('Mean execution time, KURAMOTO',
        lambda rows, bc: fmt(mean_of(bc.get('KURAMOTO', []), 'duration_s'), '', 2))
    line('Mean execution time, RETRACTION',
        lambda rows, bc: fmt(mean_of(bc.get('RETRACTION', []), 'duration_s'), '', 2))
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    if args[0] == '--combined':
        all_results = []
        for spec in args[1:]:
            path, n = spec.rsplit(':', 1)
            label = f'{n}-arm cell'
            rows, by_cat = report_one(path, label)
            all_results.append((label, n, (rows, by_cat)))
        latex_table(all_results)
    else:
        path, label = args[0], (args[1] if len(args) > 1 else args[0])
        report_one(path, label)