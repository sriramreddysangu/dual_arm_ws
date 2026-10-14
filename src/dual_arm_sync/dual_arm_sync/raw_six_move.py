#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'six'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, time, argparse, itertools
import numpy as np
from _robot8x import (NDOF, POS_LIM, VEL_LIM, RATE_HZ, ROBOT_BASES,
                      DH_TO_CMD, fk_world, pair_min_dist, resample,
                      CLEAR_M, PREFILTER_M, _caps_surface)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False
    print('[raw_six_arm_move] no ROS2 -- prediction-only mode')

CTRL_TOPIC = '/{arm}/gz/dsr_position_controller/commands'
REORDER_CMD = False          
VEL_USE_FRAC = 0.50          
MIN_DURATION = 4.0
APPROACH_MIN_S = 1.5
APPROACH_FRAC = 0.4
HOLD_AFTER_S = 2.0
GAZEBO_WAIT_S = float(os.environ.get('DUAL_ARM_RAW_WAIT_S', '15'))
SCAN_STRIDE_RAW = int(os.environ.get('DUAL_ARM_RAW_STRIDE', '1'))
PEN_EPS = 1e-6               
OUT_JSON = 'raw_six_arm_move.json'


def _cmd(q):
    return q[DH_TO_CMD] if REORDER_CMD else q


JOINT_UNITS = os.environ.get('DUAL_ARM_RAW_UNITS', 'rad').lower()
IN_RAD = not JOINT_UNITS.startswith('deg')
UNIT = 'rad' if IN_RAD else 'deg'


def fmt(q, deg=None):
    if deg is None:
        deg = not IN_RAD
    if deg:
        return '[' + ', '.join('{:8.2f}'.format(x) for x in np.degrees(q)) + ']'
    return '[' + ', '.join('{:9.4f}'.format(x) for x in q) + ']'

def lerp_duration(starts, targets, arms):
    
    t = MIN_DURATION
    for n in arms:
        disp = np.abs(targets[n] - starts[n])
        t = max(t, float(np.max(disp / (VEL_LIM * VEL_USE_FRAC))))
    return t


def build_lerp(start, target, duration, ease=False):
    
    n = max(2, int(round(duration * RATE_HZ)))
    u = np.linspace(0.0, 1.0, n)
    if ease:
        u = u * u * (3.0 - 2.0 * u)
    pos = start[None, :] + u[:, None] * (target - start)[None, :]
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])


def velocity_check(starts, targets, arms, duration):
    
    worst = 0.0
    rows = []
    for n in arms:
        v = np.abs(targets[n] - starts[n]) / duration
        frac = float(np.max(v / VEL_LIM))
        worst = max(worst, frac)
        rows.append((n, frac, int(np.argmax(v / VEL_LIM)) + 1))
    return worst, rows

def predict_pairs(traj, arms, duration, stride=SCAN_STRIDE_RAW):
    bases = {n: np.array(ROBOT_BASES[n], float) for n in arms}
    out = []
    for a, b in itertools.combinations(arms, 2):
        K = min(len(traj[a]), len(traj[b]))
        ks = list(range(0, K, max(1, stride)))
        if ks[-1] != K - 1:
            ks.append(K - 1)
        dmin = float('inf'); worst_k = -1
        first = -1; n_coll = 0
        pen_first = -1; pen_last = -1; n_pen = 0
        for k in ks:
            if _caps_surface(traj[a][k], bases[a],
                             traj[b][k], bases[b]) > PREFILTER_M:
                continue
            d = pair_min_dist(traj[a][k], bases[a], traj[b][k], bases[b])
            if d < dmin:
                dmin = d; worst_k = k
            if d < CLEAR_M:
                n_coll += 1
                if first < 0:
                    first = k
            if d <= PEN_EPS:
                
                n_pen += 1
                if pen_first < 0:
                    pen_first = k
                pen_last = k
        out.append({
            'pair': (a, b), 'collision': n_coll > 0,
            'samples': len(ks), 'coll_samples': n_coll,
            'min_cm': round(float(min(dmin, 1.0)) * 100, 2),
            'worst_t_s': (round(worst_k / max(K - 1, 1) * duration, 3)
                          if worst_k >= 0 else None),
            'worst_arc': (round(worst_k / max(K - 1, 1), 3)
                          if worst_k >= 0 else None),
            'first_t_s': (round(first / max(K - 1, 1) * duration, 3)
                          if first >= 0 else None),
            'first_arc': (round(first / max(K - 1, 1), 3)
                          if first >= 0 else None),
            'worst_deficit_cm': (round((CLEAR_M - min(dmin, CLEAR_M)) * 100, 2)
                                 if n_coll > 0 else 0.0),
            'pen_samples': n_pen,
            'pen_window_s': (None if pen_first < 0 else
                             [round(pen_first / max(K - 1, 1) * duration, 3),
                              round(pen_last / max(K - 1, 1) * duration, 3)])})
    return out


def print_prediction(recs, duration):
    bar = '=' * 78
    print('\n' + bar)
    print('  RAW LERP PREDICTION  (real mesh, gate {:.0f} cm, no resolution)'
          .format(CLEAR_M * 100))
    print(bar)
    print('  {:<18} {:>9} {:>9} {:>10}  {}'.format(
        'pair', 'min[cm]', 'below[%]', 'first[s]', 'verdict'))
    print('  ' + '-' * 74)
    n_coll = 0
    for r in sorted(recs, key=lambda x: (not x['collision'], x['min_cm'])):
        tag = '{}<->{}'.format(*r['pair'])
        if not r['collision']:
            print('  {:<18} {:>9.2f} {:>9} {:>10}  clear'.format(
                tag, r['min_cm'], '-', '-'))
            continue
        n_coll += 1
        pct = 100.0 * r['coll_samples'] / max(r['samples'], 1)
        if r.get('pen_samples', 0) > 0:
            w = r['pen_window_s']
            print('  {:<18} {:>9.2f} {:>9.1f} {:>10.2f}  INTERPENETRATING '
                  't={:.2f}..{:.2f}s ({} of {} samples)'.format(
                      tag, r['min_cm'], pct, r['first_t_s'],
                      w[0], w[1], r['pen_samples'], r['samples']))
        else:
            print('  {:<18} {:>9.2f} {:>9.1f} {:>10.2f}  COLLISION ({:.1f} cm '
                  'inside the gate at t={:.2f}s)'.format(
                      tag, r['min_cm'], pct, r['first_t_s'],
                      r['worst_deficit_cm'], r['worst_t_s']))
    print('  ' + '-' * 74)
    print('  {} of {} pair(s) collide over {:.2f}s of simultaneous motion.'
          .format(n_coll, len(recs), duration))
    print(bar)
    return n_coll


def print_config_block(arms, starts, targets, duration):
    bar = '=' * 78
    print('\n' + bar)
    print('  START (live Gazebo)  ->  TARGET   [screen-capture block]')
    print(bar)
    for n in arms:
        base = np.array(ROBOT_BASES[n], float)
        dq = targets[n] - starts[n]
        d = dq if IN_RAD else np.degrees(dq)
        wfmt = '{:+9.4f}' if IN_RAD else '{:+8.2f}'
        print('  {}  base=({:+.3f}, {:+.3f})'.format(n, base[0], base[1]))
        print('    START  {} {}'.format(UNIT, fmt(starts[n])))
        print('    TARGET {} {}'.format(UNIT, fmt(targets[n])))
        print('    DELTA  {} [{}]   max {:.4f} {} on J{}'.format(
            UNIT, ', '.join(wfmt.format(v) for v in d),
            float(np.max(np.abs(d))), UNIT, int(np.argmax(np.abs(d))) + 1))
        sw = fk_world(starts[n], base); tw = fk_world(targets[n], base)
        print('    EE world  [{:.3f}, {:.3f}, {:.3f}]  ->  [{:.3f}, {:.3f}, {:.3f}]'
              .format(sw[0], sw[1], sw[2], tw[0], tw[1], tw[2]))
    print('  shared duration {:.2f}s, {} samples/arm @ {:.0f} Hz'.format(
        duration, max(2, int(round(duration * RATE_HZ))), RATE_HZ))
    print(bar)

if _ROS_OK:
    class RawNode(Node):

        def __init__(self, arms):
            super().__init__('raw_six_arm_move')
            self.arms = list(arms)
            self._cur = {n: None for n in self.arms}
            self._pubs = {n: self.create_publisher(
                Float64MultiArray, CTRL_TOPIC.format(arm=n), 10)
                for n in self.arms}
            for name in self.arms:
                for topic in ('/{}/gz/joint_states'.format(name),
                              '/{}/joint_states'.format(name)):
                    self.create_subscription(
                        JointState, topic,
                        lambda m, n=name: self._cb(m, n),
                        qos_profile_sensor_data)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF:
                return
            jmap = {nm: i for i, nm in enumerate(msg.name)}
            keys = ['joint_{}'.format(k) for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._cur[name] = q.astype(float)

        def wait(self, t=GAZEBO_WAIT_S):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._cur[n] is not None for n in self.arms):
                    return True
            return False

        def refresh(self, seconds=0.4):
            for _ in range(max(1, int(seconds * RATE_HZ))):
                rclpy.spin_once(self, timeout_sec=0.02)

        def missing(self):
            return [n for n in self.arms if self._cur[n] is None]

        def current(self, n):
            return None if self._cur[n] is None else self._cur[n].copy()

        def _publish(self, q_by_arm):
            msg = Float64MultiArray()
            for n in self.arms:
                msg.data = [float(v) for v in _cmd(q_by_arm[n])]
                self._pubs[n].publish(msg)

        def approach(self, traj):
            
            dt_ns = int(1e9 / RATE_HZ)
            self.refresh(0.5)
            cur = {n: self.current(n) for n in self.arms}
            if any(cur[n] is None for n in self.arms):
                print('  APPROACH: no live pose -- skipping the ramp.')
                return
            gap = max(float(np.max(np.abs(np.degrees(traj[n][0] - cur[n]))))
                      for n in self.arms)
            if gap < 0.05:
                print('  APPROACH: already on the start pose (gap {:.3f} deg).'
                      .format(gap))
                return
            dur = APPROACH_MIN_S
            for n in self.arms:
                disp = np.abs(traj[n][0] - cur[n])
                dur = max(dur, float(np.max(disp / (VEL_LIM * APPROACH_FRAC))))
            print('  APPROACH: {:.1f} deg gap, ramping over {:.1f}s'.format(gap, dur))
            n_app = max(2, int(dur * RATE_HZ))
            for k in range(n_app):
                t0 = time.monotonic_ns()
                a = k / (n_app - 1)
                self._publish({n: cur[n] + a * (traj[n][0] - cur[n])
                               for n in self.arms})
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

        def execute(self, traj, duration, recs):
            dt_ns = int(1e9 / RATE_HZ)
            n_steps = max(len(traj[n]) for n in self.arms)
            bases = {n: np.array(ROBOT_BASES[n], float) for n in self.arms}
            watch = [r['pair'] for r in recs if r['collision']]
            if not watch and recs:
                watch = [min(recs, key=lambda r: r['min_cm'])['pair']]
            min_seen = {p: float('inf') for p in watch}
            milestone = max(1, n_steps // 10)
            measured = {n: [] for n in self.arms}
            print('  Executing {} steps @ {:.0f} Hz ({:.2f}s) -- watch Gazebo'
                  .format(n_steps, RATE_HZ, duration))
            print('  (the t= lines below are a 10-point DISPLAY sample; the'
                  ' exact minimum is recomputed at full resolution after the'
                  ' move)')

            for k in range(n_steps):
                t0 = time.monotonic_ns()
                self._publish({n: traj[n][min(k, len(traj[n]) - 1)]
                               for n in self.arms})
                rclpy.spin_once(self, timeout_sec=0.)
                for n in self.arms:         
                    q = self._cur[n]
                    measured[n].append(None if q is None else q.copy())
                if watch and (k % milestone == 0 or k == n_steps - 1):
                    cells = []
                    for (a, b) in watch:
                        qa = traj[a][min(k, len(traj[a]) - 1)]
                        qb = traj[b][min(k, len(traj[b]) - 1)]
                        d = pair_min_dist(qa, bases[a], qb, bases[b])
                        min_seen[(a, b)] = min(min_seen[(a, b)], d)
                        cells.append('{}/{} {:5.1f}cm{}'.format(
                            a[-2:], b[-2:], d * 100, '!' if d < CLEAR_M else ' '))
                    print('    t={:5.2f}s  '.format(k / RATE_HZ) + '  '.join(cells))
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

            for _ in range(int(HOLD_AFTER_S * RATE_HZ)):
                t0 = time.monotonic_ns()
                self._publish({n: traj[n][-1] for n in self.arms})
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

            if watch:
                print('\n  Display-sample minimum (10 points, NOT the answer):')
                for p, d in min_seen.items():
                    print('    {}<->{}  {:.2f} cm'.format(p[0], p[1], d * 100))
            print('\n  ARRIVAL CHECK (measured Gazebo joints vs commanded target):')
            arrival = {}
            for n in self.arms:
                meas = self.current(n)
                if meas is None:
                    print('    {}: no joint_states received'.format(n))
                    continue
                err = float(np.max(np.abs(np.degrees(meas - traj[n][-1]))))
                arrival[n] = round(err, 3)
                note = ('   (blocked -- the arms are interpenetrating)'
                        if err > 5.0 else '')
                print('    {}  max joint error {:.3f} deg{}'.format(n, err, note))

           
            meas_traj = {}
            for n in self.arms:
                rows = [q for q in measured[n] if q is not None]
                if len(rows) >= 2:
                    meas_traj[n] = np.asarray(rows)
            if len(meas_traj) == len(self.arms):
                nmin = min(len(meas_traj[n]) for n in self.arms)
                meas_traj = {n: meas_traj[n][:nmin] for n in self.arms}
                mrecs = predict_pairs(meas_traj, self.arms, duration, stride=1)
                print('\n  MEASURED path (what Gazebo executed, every step):')
                for r in sorted(mrecs, key=lambda x: x['min_cm']):
                    if not r['collision'] and r['min_cm'] > 100 * CLEAR_M * 2:
                        continue
                    tag = '{}<->{}'.format(*r['pair'])
                    if r.get('pen_samples', 0) > 0:
                        w = r['pen_window_s']
                        print('    {:<18} INTERPENETRATED t={:.2f}..{:.2f}s '
                              '({} steps)'.format(tag, w[0], w[1],
                                                  r['pen_samples']))
                    else:
                        print('    {:<18} min {:.2f} cm{}'.format(
                            tag, r['min_cm'],
                            '   <-- inside the gate' if r['collision'] else ''))
            else:
                mrecs = []
                print('\n  (no complete measured trajectory -- joint states '
                      'were incomplete)')
            return min_seen, arrival, mrecs

def parse_target(raw):
    vals = [float(v) for v in raw.replace(',', ' ').split()]
    if len(vals) != NDOF:
        raise ValueError('need exactly {} values, got {}'.format(NDOF, len(vals)))
    v = np.array(vals, float)
    q_raw = v if IN_RAD else np.radians(v)
    q = np.clip(q_raw, POS_LIM[:, 0], POS_LIM[:, 1])
    return q, not np.allclose(q, q_raw, atol=1e-9)


def prompt_targets(arms, starts):
    targets = {}
    for n in arms:
        base = np.array(ROBOT_BASES[n], float)
        print('\n  -- {}  base=({:+.3f}, {:+.3f}) ----------------------'
              .format(n.upper(), base[0], base[1]))
        print('  START (live Gazebo, {}): {}'.format(UNIT, fmt(starts[n])))
        if IN_RAD:
            print('  limits (rad): J1=+-6.2832  J2=+-1.6493  J3=+-2.7908  '
                  'J4/5/6=+-6.2832')
        else:
            print('  limits (deg): J1=+-360  J2=+-94.5  J3=+-159.9  J4/5/6=+-360')
        while True:
            raw = input('  TARGET J1..J6 ({}): '.format(UNIT)).strip()
            try:
                q, clamped = parse_target(raw)
            except ValueError as e:
                print('    {}'.format(e)); continue
            if clamped:
                print('    WARNING: outside joint limits -- CLAMPED.')
            print('    confirmed: {} {}'.format(fmt(q), UNIT))
            targets[n] = q
            break
    return targets


def targets_from_file(path, arms):
    with open(path) as fh:
        cfg = json.load(fh)
    missing = [n for n in arms if n not in cfg.get('arms', {})]
    if missing:
        raise SystemExit('  {} has no entry for: {}'.format(
            path, ', '.join(missing)))
    out = {}
    for n in arms:
        e = cfg['arms'][n]
        
        if 'target_joints' in e:
            out[n] = np.array(e['target_joints'], float)
        else:
            out[n] = np.radians(np.array(e['target_deg'], float))
    print('  Targets loaded from {} (START still comes from Gazebo).'.format(path))
    return out


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--arms', default=None, help='comma list (default: all six)')
    ap.add_argument('--targets-from', default=None,
                    help='read target_deg from an s201_config.json')
    ap.add_argument('--from-seed', nargs='?', const='s202_trajectories.json',
                    default=None,
                    help='replay the step_202 B-spline seed instead of a lerp')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--ease', action='store_true',
                    help='smoothstep timing instead of pure lerp (same path)')
    ap.add_argument('--no-exec', action='store_true', help='predict only')
    ap.add_argument('--yes', action='store_true', help='skip the confirmation')
    args, _ = ap.parse_known_args()

    arms = (list(_R8X.ARM_NAMES) if args.arms is None
            else [a.strip() for a in args.arms.split(',')])
    bad = [a for a in arms if a not in _R8X.ROBOT_BASES]
    if bad:
        raise SystemExit('  unknown arm(s): {}'.format(', '.join(bad)))
    if len(arms) < 2:
        raise SystemExit('  need at least 2 arms')

    bar = '=' * 78
    print('\n' + bar)
    print('  RAW MULTI-ARM MOVE  --  simultaneous joint-space LERP, NO avoidance')
    print(bar)
    print('  Active arms ({}): {}'.format(len(arms), ', '.join(arms)))
    for n in arms:
        b = np.array(ROBOT_BASES[n], float)
        print('    {}  base ({:+.3f}, {:+.3f})'.format(n, b[0], b[1]))

    node = None
    if _ROS_OK:
        rclpy.init()
        node = RawNode(arms)
        print('\n  Reading Gazebo joint states ({:.0f}s timeout)...'
              .format(GAZEBO_WAIT_S))
        if node.wait():
            print('  All {} arm(s) read -- using the live pose as START.'
                  .format(len(arms)))
        else:
            print('  TIMEOUT -- nothing from: {}'.format(', '.join(node.missing())))

    traj = {}; starts = {}; targets = {}
    duration = args.duration
    source = 'lerp_from_live_gazebo'

    if args.from_seed:
        with open(args.from_seed) as fh:
            t = json.load(fh)
        missing = [n for n in arms if n not in t]
        if missing:
            raise SystemExit('  {} has no entry for: {}'.format(
                args.from_seed, ', '.join(missing)))
        duration = duration or float(t['duration'])
        for n in arms:
            traj[n] = resample(np.array(t[n]['trajectory']['positions'], float),
                               duration)
            starts[n] = traj[n][0].copy()
            targets[n] = traj[n][-1].copy()
        source = args.from_seed
        print('\n  Source: {} -- B-spline seed replay, NOT a lerp.'.format(source))
        print('  Sample 0 is the saved start, so the arms are ramped onto it first.')

    else:
        for n in arms:
            q = node.current(n) if node is not None else None
            if q is None:
                print('\n  No live START pose for {}.'.format(n))
                print('  Refusing to assume HOME -- zeros as start would snap the')
                print('  arm across the cell. Fix the joint-state topic and re-run.')
                if node is not None:
                    try: node.destroy_node()
                    except Exception: pass
                    try: rclpy.shutdown()
                    except Exception: pass
                sys.exit(1)
            starts[n] = q
        targets = (targets_from_file(args.targets_from, arms)
                   if args.targets_from else prompt_targets(arms, starts))
        auto_dur = lerp_duration(starts, targets, arms)
        if duration is None:
            duration = auto_dur
        for n in arms:
            traj[n] = build_lerp(starts[n], targets[n], duration, ease=args.ease)
        worst, rows = velocity_check(starts, targets, arms, duration)
        print('\n  Motion: {} over {:.2f}s'.format(
            'smoothstep-timed straight line (--ease)' if args.ease
            else 'PURE LERP', duration))
        print('  peak joint velocity = {:.0f}% of VEL_LIM{}'.format(
            100 * worst, '' if args.ease else '  (constant through the move)'))
        for n, frac, j in rows:
            print('    {}  {:.0f}% of limit on J{}'.format(n, 100 * frac, j))
        if worst > 1.0:
            print('  WARNING: --duration {:.2f}s exceeds a joint velocity limit.'
                  .format(duration))
            print('  The limit-respecting duration is {:.2f}s.'.format(auto_dur))
        if args.ease and worst * 1.5 > 1.0:
            print('  WARNING: smoothstep peaks at 1.5x the average -- {:.0f}% '
                  'of limit.'.format(150 * worst))

    print_config_block(arms, starts, targets, duration)
    recs = predict_pairs(traj, arms, duration)
    n_coll = print_prediction(recs, duration)

    record = {'source': source,
              'motion': ('bspline_seed_replay' if args.from_seed
                         else ('smoothstep_line' if args.ease else 'pure_lerp')),
              'arm_names': arms, 'duration_s': duration, 'clear_m': CLEAR_M,
              'resolution': 'none',
              'prediction': [{**r, 'pair': '{}<->{}'.format(*r['pair'])}
                             for r in recs],
              'start_joints': {n: starts[n].tolist() for n in arms},
              'target_joints': {n: targets[n].tolist() for n in arms},
              'start_deg': {n: np.degrees(starts[n]).tolist() for n in arms},
              'target_deg': {n: np.degrees(targets[n]).tolist() for n in arms}}

    if args.no_exec:
        print('\n  --no-exec set -- nothing commanded.')
    elif not _ROS_OK:
        print('\n  No ROS2 -- the prediction above is the whole result.')
    elif any(node.current(n) is None for n in arms):
        print('\n  Joint states missing for {} -- refusing to execute.'
              .format(', '.join(node.missing())))
    else:
        if n_coll:
            print('\n  This WILL drive the arms into each other in Gazebo.')
            print('  Simulation only -- never run this against hardware.')
        go = 'y' if args.yes else input('\n  Execute? [y/N]: ').strip().lower()
        if go == 'y':
            node.approach(traj)
            min_seen, arrival, mrecs = node.execute(traj, duration, recs)
            record['measured'] = {
                'display_sample_min_cm': {'{}<->{}'.format(a, b): round(d * 100, 2)
                                          for (a, b), d in min_seen.items()},
                'arrival_max_joint_err_deg': arrival,
                'scan': [{**r, 'pair': '{}<->{}'.format(*r['pair'])}
                         for r in mrecs]}
        else:
            print('  Skipped.')

    with open(OUT_JSON, 'w') as fh:
        json.dump(record, fh, indent=2)
    print('\n  Saved: {}'.format(OUT_JSON))

    if node is not None:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()