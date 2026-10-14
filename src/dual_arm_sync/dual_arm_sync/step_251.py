#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'dual'
os.environ.setdefault('DUAL_ARM_REQUIRE_MESH', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, time, argparse
import numpy as np
from _robot8x import (NDOF, POS_LIM, fk_world, pair_collides, CLEAR_M)
import step_101 as S101

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

GAZEBO_WAIT_S = float(os.environ.get('DUAL_ARM_S251_WAIT_S', '20'))

if _ROS_OK:
    class JointReader(Node):

        def __init__(self, arms):
            super().__init__('step_251_reader')
            self._arms = list(arms)
            self._q = {n: None for n in self._arms}
            for name in self._arms:
                for topic in ('/{}/gz/joint_states'.format(name),
                              '/{}/joint_states'.format(name)):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n),
                        qos_profile_sensor_data)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF:
                return
            jmap = {nm: i for i, nm in enumerate(msg.name)}
            keys = ['joint_{}'.format(k) for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._q[name] = q.astype(float)

        def wait(self, t=GAZEBO_WAIT_S):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._q[n] is not None for n in self._arms):
                    return True
            return False

        def missing(self):
            return [n for n in self._arms if self._q[n] is None]

        def joints(self, n):
            return None if self._q[n] is None else self._q[n].copy()
        
        
def parse_deg_joints(vals):
    q_raw = np.radians(np.array(vals, float))
    q = np.clip(q_raw, POS_LIM[:, 0], POS_LIM[:, 1])
    return q, not np.allclose(q, q_raw, atol=1e-9)


def show(label, q, base):
    deg = ', '.join('{:8.2f}'.format(np.degrees(v)) for v in q)
    ee = fk_world(q, base)
    print('    {}'.format(label))
    print('      [{}] deg'.format(deg))
    print('      EE world: [{:.3f}, {:.3f}, {:.3f}] m'.format(ee[0], ee[1], ee[2]))


def prompt_arm(name, base, start_q):
    """6 numbers = joints (deg, used as typed). 3 numbers = X Y Z world (IK)."""
    print('\n  ' + '-' * 62)
    print('  ARM: {}  base={}'.format(name.upper(), np.asarray(base).tolist()))
    show('Current Gazebo (START):', start_q, base)
    print('  limits (deg): J1=+-360  J2=+-94.5  J3=+-159.9  J4/5/6=+-360')
    while True:
        raw = input('  [{}] TARGET  6 joints j1..j6 (deg)  OR  3 numbers X Y Z (m): '
                    .format(name)).strip()
        try:
            vals = [float(v) for v in raw.replace(',', ' ').split()]
        except ValueError:
            print('    numbers only'); continue
        if len(vals) == 6:
            q, clamped = parse_deg_joints(vals)
            if clamped:
                print('    WARNING: outside joint limits -- CLAMPED.')
            show('Target (joints, no IK):', q, base)
            return {'mode': 'joint', 'forced_q': q, 'target_world': None}
        if len(vals) == 3:
            tw = np.array(vals, float)
            print('    target world: [{:.3f}, {:.3f}, {:.3f}] m  -> IK'.format(*tw))
            return {'mode': 'cartesian', 'forced_q': None, 'target_world': tw}
        print('    enter 6 numbers (joints, deg) or 3 numbers (X Y Z, m)')


def env_arm(name):
    key = name.upper()
    goal = os.environ.get('DUAL_ARM_{}_GOAL'.format(key))
    if goal is None:
        return None
    mode = os.environ.get('DUAL_ARM_S251_MODE', 'joint').strip().lower()
    vals = [float(x) for x in goal.replace(',', ' ').split()]
    if mode == 'joint':
        if len(vals) != NDOF:
            raise SystemExit('  DUAL_ARM_{}_GOAL needs {} degree values'
                             .format(key, NDOF))
        q, _ = parse_deg_joints(vals)
        return {'mode': 'joint', 'forced_q': q, 'target_world': None}
    if len(vals) != 3:
        raise SystemExit('  DUAL_ARM_{}_GOAL needs x,y,z metres'.format(key))
    return {'mode': 'cartesian', 'forced_q': None,
            'target_world': np.array(vals, float)}


def env_start(name):
    raw = os.environ.get('DUAL_ARM_{}_START'.format(name.upper()))
    if raw is None:
        return None
    vals = [float(x) for x in raw.replace(',', ' ').split()]
    if len(vals) != NDOF:
        raise SystemExit('  DUAL_ARM_{}_START needs {} degree values'
                         .format(name.upper(), NDOF))
    return parse_deg_joints(vals)[0]

def endpoint_report(best, bases, arms):
    issues = []
    for i, ni in enumerate(arms):
        for nj in arms[i + 1:]:
            if pair_collides(best[ni], bases[ni], best[nj], bases[nj],
                             margin=CLEAR_M):
                issues.append('{}<->{}'.format(ni, nj))
    return issues


def print_config_block(arms, bases, starts, targets):
    bar = '=' * 78
    print('\n' + bar)
    print('  CONFIGURATION  (screen-capture block)')
    print(bar)
    for n in arms:
        print('  {}  base=({:+.3f}, {:+.3f})'.format(
            n, bases[n][0], bases[n][1]))
        print('    START  deg [{}]'.format(
            ', '.join('{:8.2f}'.format(np.degrees(v)) for v in starts[n])))
        print('    TARGET deg [{}]'.format(
            ', '.join('{:8.2f}'.format(np.degrees(v)) for v in targets[n])))
        sw = fk_world(starts[n], bases[n]); tw = fk_world(targets[n], bases[n])
        print('    EE world  [{:.3f}, {:.3f}, {:.3f}]  ->  [{:.3f}, {:.3f}, {:.3f}]'
              .format(sw[0], sw[1], sw[2], tw[0], tw[1], tw[2]))
    print(bar)

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--arms', default=None,
                    help='comma list, e.g. dsr01,dsr02 (default: both)')
    ap.add_argument('--from-config', default=None,
                    help='reuse an s251_config.json (exact replay)')
    ap.add_argument('--start-live', action='store_true',
                    help='with --from-config, take START from Gazebo instead '
                         'of the saved starts')
    ap.add_argument('--start-home', action='store_true',
                    help='deliberately use HOME (zeros) as START')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--force', action='store_true',
                    help='write the file even if target poses collide')
    args, _ = ap.parse_known_args()

    all_arms = list(_R8X.ARM_NAMES)
    arms = all_arms if args.arms is None else [a.strip() for a in args.arms.split(',')]
    bad = [a for a in arms if a not in _R8X.ROBOT_BASES]
    if bad:
        raise SystemExit('  unknown arm(s): {}'.format(', '.join(bad)))
    if len(arms) < 2:
        raise SystemExit('  need at least 2 arms')
    bases = {n: np.array(_R8X.ROBOT_BASES[n], float) for n in arms}

    print('\n' + '=' * 78)
    print('  STEP 251  --  manual configuration entry  [6-arm cell, 1.39 m grid]')
    print('=' * 78)
    print('  Active arms ({}): {}'.format(len(arms), ', '.join(arms)))

    cfg = None
    if args.from_config:
        with open(args.from_config) as fh:
            cfg = json.load(fh)
        missing = [n for n in arms if n not in cfg.get('arms', {})]
        if missing:
            raise SystemExit('  {} has no entry for: {}'.format(
                args.from_config, ', '.join(missing)))
        print('  Replaying configuration from {}'.format(args.from_config))

    node = None
    live = {n: None for n in arms}
    want_live = args.start_live or (cfg is None)
    if _ROS_OK and want_live and not args.start_home:
        rclpy.init()
        node = JointReader(arms)
        print('\n  Reading Gazebo joint states ({:.0f}s timeout)...'.format(GAZEBO_WAIT_S))
        if node.wait():
            for n in arms:
                live[n] = node.joints(n)
            print('  All {} arms read.'.format(len(arms)))
        else:
            print('  TIMEOUT -- nothing received from: {}'.format(
                ', '.join(node.missing())))

    starts = {}
    for n in arms:
        q = env_start(n)
        if q is None and cfg is not None and not args.start_live:
            q = np.radians(np.array(cfg['arms'][n]['start_deg'], float))
        if q is None:
            q = live[n]
        if q is None:
            if args.start_home:
                q = np.zeros(NDOF)
            else:
                print('\n  No START pose for {}.'.format(n))
                print('  Refusing to assume HOME: zeros-as-start makes the arm')
                print('  snap across the cell on the first command. Fix the')
                print('  joint-state topic, or pass --start-home on purpose.')
                raise SystemExit(1)
        starts[n] = np.asarray(q, float)

    spec = {}
    for n in arms:
        s = env_arm(n)
        if s is None and cfg is not None:
            c = cfg['arms'][n]
            if c.get('mode') == 'cartesian' and c.get('target_world') is not None:
                s = {'mode': 'cartesian', 'forced_q': None,
                     'target_world': np.array(c['target_world'], float)}
            else:
                s = {'mode': 'joint',
                     'forced_q': np.radians(np.array(c['target_deg'], float)),
                     'target_world': None}
        if s is None:
            s = prompt_arm(n, bases[n], starts[n])
        spec[n] = s

    duration = args.duration
    if duration is None and cfg is not None:
        duration = float(cfg.get('duration', 10.0))
    if duration is None:
        raw = input('\n  Duration [s] (Enter=10.0): ').strip()
        duration = float(raw) if raw else 10.0

    arms_data = {}
    for n in arms:
        s = spec[n]
        tl = (None if s['target_world'] is None
              else np.asarray(s['target_world'], float) - bases[n])
        arms_data[n] = {'base': bases[n], 'start_q': starts[n],
                        'target_local': tl, 'target_rot': None,
                        'constrain_orient': False, 'forced_q': s['forced_q']}

    print('\n  Planning (step_101 IK + endpoint CSP, real-mesh gate)...')
    t0 = time.time()
    best = S101.greedy(arms_data)
    if best is None:
        print('  STEP 251: no IK solution for at least one arm -- aborting.')
        sys.exit(1)
    endpoint_ok = bool(best.pop('_endpoint_ok', True))
    t_ik = time.time() - t0

    targets = {n: np.asarray(best[n], float) for n in arms}
    issues = endpoint_report(targets, bases, arms)
    if issues:
        print('\n  TARGET POSES COLLIDE ON THE REAL MESH: {}'.format(', '.join(issues)))
        print('  This is NOT resolvable downstream -- both arms must occupy those')
        print('  poses at the end, so no phase lag and no deformation can fix it.')
        print('  Retype one of the targets (or pass --force to write anyway).')
        if not args.force:
            sys.exit(1)

    for n in arms:
        if spec[n]['mode'] != 'joint':
            continue
        d = np.max(np.abs(targets[n] - spec[n]['forced_q']))
        if d > 1e-6:
            print('\n  NOTE {}: target re-wound to the nearest 2pi representation'
                  ' from the start pose (same pose, less travel).'.format(n))
            print('    typed  deg [{}]'.format(', '.join(
                '{:8.2f}'.format(np.degrees(v)) for v in spec[n]['forced_q'])))
            print('    using  deg [{}]'.format(', '.join(
                '{:8.2f}'.format(np.degrees(v)) for v in targets[n])))

    print_config_block(arms, bases, starts, targets)

    ik = {'duration': float(duration), 'arm_names': arms,
          'endpoint_ok': bool(endpoint_ok and not issues),
          'ik_time_s': round(t_ik, 3), 'source': 'step_251_manual'}
    cart = {}
    for n in arms:
        ik[n] = {'base': bases[n].tolist(),
                 'start_joints': starts[n].tolist(),
                 'target_joints': targets[n].tolist(),
                 'end_joints': targets[n].tolist(),
                 'target_joints_deg': np.degrees(targets[n]).tolist(),
                 'input_mode': spec[n]['mode']}
        cart[n] = {'start_world': fk_world(starts[n], bases[n]).tolist(),
                   'target_world': fk_world(targets[n], bases[n]).tolist()}
    ik['cartesian'] = cart
    with open('s251_ik.json', 'w') as fh:
        json.dump(ik, fh)

    out_cfg = {'arm_names': arms, 'duration': float(duration), 'arms': {}}
    for n in arms:
        out_cfg['arms'][n] = {
            'mode': spec[n]['mode'],
            'start_deg': np.degrees(starts[n]).tolist(),
            'target_deg': np.degrees(targets[n]).tolist(),
            'target_world': (None if spec[n]['target_world'] is None
                             else np.asarray(spec[n]['target_world']).tolist())}
    with open('s251_config.json', 'w') as fh:
        json.dump(out_cfg, fh, indent=2)

    print('\n  Saved: s251_ik.json  +  s251_config.json   (IK {:.2f}s)'.format(t_ik))
    print('  Replay this exact configuration later with:')
    print('    python3 step_251.py --from-config s251_config.json')
    print('  Isolate one pair for the dual-arm clip with:')
    print('    python3 step_251.py --arms <a>,<b> --from-config s251_config.json')
    print('  Next: step_252')

    if node is not None:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()