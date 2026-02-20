from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Memory:
    is_active: bool
    strength: int
    useful_count: int
    useful_score: float
    created_day: int
    last_used_day: int
    next_decay_day: int
    use_prob: float
    deleted_day: int | None = None
    profile: str = "random"


def _memory_tier(useful_score: float) -> int:
    if useful_score >= 10:
        return 2
    if useful_score >= 3:
        return 1
    return 0


def forget_cycle_days(
    useful_score: float,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> int:
    tier = _memory_tier(useful_score)
    if tier == 2:
        base = cycle_30d
        tier_speed = tier2_speed
    elif tier == 1:
        base = cycle_7d
        tier_speed = tier1_speed
    else:
        base = cycle_1d
        tier_speed = tier0_speed

    speed = max(0.01, float(forget_speed)) * max(0.01, float(tier_speed))
    # speed 越大，遗忘越快（周期越短）
    adjusted = int(round(base / speed))
    return max(1, adjusted)


def _apply_natural_decay_if_needed(
    m: Memory,
    day: int,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> None:
    """仅 0 档执行时间衰减。"""
    tier = _memory_tier(m.useful_score)
    if tier != 0:
        return
    while day >= m.next_decay_day and m.strength > 0:
        m.strength -= 1
        cycle = forget_cycle_days(
            m.useful_score,
            cycle_1d,
            cycle_7d,
            cycle_30d,
            forget_speed,
            tier0_speed,
            tier1_speed,
            tier2_speed,
        )
        m.next_decay_day += cycle


def _apply_recall_result(
    m: Memory,
    day: int,
    useful: bool,
    useful_boost: int,
    consolidate_speed: float,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> None:
    """
    三档规则：
    - 0档：时间遗忘；有用时强化
    - 1档：仅在“召回且没用”时衰减；有用时强化
    - 2档：永不遗忘；有用时可继续强化（不衰减）
    """
    tier = _memory_tier(m.useful_score)
    if useful:
        m.strength += useful_boost
        m.useful_count += 1
        m.useful_score += max(0.01, float(consolidate_speed))
        m.last_used_day = day
        if _memory_tier(m.useful_score) == 0:
            cycle = forget_cycle_days(
                m.useful_score,
                cycle_1d,
                cycle_7d,
                cycle_30d,
                forget_speed,
                tier0_speed,
                tier1_speed,
                tier2_speed,
            )
            m.next_decay_day = day + cycle
        return

    # useless recall
    if tier == 1:
        m.strength -= 1


def make_memory(
    rng: random.Random,
    day: int,
    initial_strength: int,
    active_ratio: float,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> Memory:
    is_active = rng.random() < active_ratio
    # 用三档概率近似“热/温/冷”记忆
    p = rng.random()
    if p < 0.15:
        use_prob = rng.uniform(0.08, 0.18)  # 热
    elif p < 0.50:
        use_prob = rng.uniform(0.02, 0.08)  # 温
    else:
        use_prob = rng.uniform(0.001, 0.02)  # 冷

    useful_count = 0
    useful_score = 0.0
    cycle = forget_cycle_days(
        useful_score, cycle_1d, cycle_7d, cycle_30d, forget_speed, tier0_speed, tier1_speed, tier2_speed
    )
    return Memory(
        is_active=is_active,
        strength=initial_strength,
        useful_count=useful_count,
        useful_score=useful_score,
        created_day=day,
        last_used_day=day,
        next_decay_day=day + cycle,
        use_prob=use_prob,
        profile="random",
    )


@dataclass
class MemoryProfile:
    name: str
    count: int
    initial_strength: int
    # 线性使用曲线：用于概率模式（默认）
    p_start: float = 0.0
    p_mid: float = 0.0
    p_end: float = 0.0
    # 特殊策略
    mode: str = "linear"  # linear | yearly_once
    hot_days: int = 0     # 阶段性高频时长
    # 每日新增（随机刷新）配置
    new_prob: float = 0.2
    new_min: int = 0
    new_max: int = 2


def simulate_once(
    years: int,
    seed: int,
    initial_strength: int,
    useful_boost: int,
    initial_memories: int,
    daily_new_memories: int,
    active_ratio: float,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    consolidate_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> Dict[str, float]:
    rng = random.Random(seed)
    total_days = years * 365

    memories: List[Memory] = [
        make_memory(
            rng=rng,
            day=0,
            initial_strength=initial_strength,
            active_ratio=active_ratio,
            cycle_1d=cycle_1d,
            cycle_7d=cycle_7d,
            cycle_30d=cycle_30d,
            forget_speed=forget_speed,
            tier0_speed=tier0_speed,
            tier1_speed=tier1_speed,
            tier2_speed=tier2_speed,
        )
        for _ in range(initial_memories)
    ]
    all_memories: List[Memory] = list(memories)

    for day in range(1, total_days + 1):
        for _ in range(daily_new_memories):
            m = make_memory(
                rng=rng,
                day=day,
                initial_strength=initial_strength,
                active_ratio=active_ratio,
                cycle_1d=cycle_1d,
                cycle_7d=cycle_7d,
                cycle_30d=cycle_30d,
                forget_speed=forget_speed,
                tier0_speed=tier0_speed,
                tier1_speed=tier1_speed,
                tier2_speed=tier2_speed,
            )
            memories.append(m)
            all_memories.append(m)

        alive: List[Memory] = []
        for m in memories:
            if m.deleted_day is not None:
                continue
            if m.is_active:
                alive.append(m)
                continue

            if rng.random() < m.use_prob:
                _apply_recall_result(
                    m=m,
                    day=day,
                    useful=True,
                    useful_boost=useful_boost,
                    consolidate_speed=consolidate_speed,
                    cycle_1d=cycle_1d,
                    cycle_7d=cycle_7d,
                    cycle_30d=cycle_30d,
                    forget_speed=forget_speed,
                    tier0_speed=tier0_speed,
                    tier1_speed=tier1_speed,
                    tier2_speed=tier2_speed,
                )
                alive.append(m)
                continue

            _apply_natural_decay_if_needed(
                m=m,
                day=day,
                cycle_1d=cycle_1d,
                cycle_7d=cycle_7d,
                cycle_30d=cycle_30d,
                forget_speed=forget_speed,
                tier0_speed=tier0_speed,
                tier1_speed=tier1_speed,
                tier2_speed=tier2_speed,
            )

            if m.strength <= 0:
                m.deleted_day = day
            else:
                alive.append(m)

        memories = alive

    passive = [m for m in all_memories if not m.is_active]
    deleted = [m for m in passive if m.deleted_day is not None]
    survivors = [m for m in passive if m.deleted_day is None]

    lifetimes = [
        (m.deleted_day if m.deleted_day is not None else total_days) - m.created_day
        for m in passive
    ]

    avg_life = statistics.mean(lifetimes) if lifetimes else 0.0
    p90_life = statistics.quantiles(lifetimes, n=10)[8] if len(lifetimes) >= 10 else avg_life
    deleted_ratio = (len(deleted) / len(passive)) if passive else 0.0
    survive_10y_ratio = (len(survivors) / len(passive)) if passive else 0.0

    return {
        "passive_total": float(len(passive)),
        "deleted_ratio": deleted_ratio,
        "survive_10y_ratio": survive_10y_ratio,
        "avg_lifetime_days": avg_life,
        "p90_lifetime_days": p90_life,
    }


def _profile_use_happens(m: Memory, day: int, total_days: int, rng: random.Random, profile: MemoryProfile) -> bool:
    if profile.mode == "yearly_once":
        # 每年固定温习一次（创建日对齐）
        return (day - m.created_day) > 0 and (day - m.created_day) % 365 == 0

    t = day - m.created_day
    if t < 0:
        return False

    # 两段线性：start -> mid -> end
    half = max(1, total_days // 2)
    if t <= profile.hot_days and profile.hot_days > 0:
        p = profile.p_start
    elif t < half:
        ratio = t / half
        p = profile.p_start + (profile.p_mid - profile.p_start) * ratio
    else:
        ratio = (t - half) / max(1, total_days - half)
        p = profile.p_mid + (profile.p_end - profile.p_mid) * ratio
    p = max(0.0, min(1.0, p))
    return rng.random() < p


def _profile_day_weight(day: int, total_days: int, profile: MemoryProfile) -> float:
    """按时间给每个画像一个当天被召回权重（用于分配召回槽位）。"""
    if profile.mode == "yearly_once":
        # 年度复习型也要有少量机会被召回
        return 0.05

    half = max(1, total_days // 2)
    if day <= profile.hot_days and profile.hot_days > 0:
        p = profile.p_start
    elif day < half:
        ratio = day / half
        p = profile.p_start + (profile.p_mid - profile.p_start) * ratio
    else:
        ratio = (day - half) / max(1, total_days - half)
        p = profile.p_mid + (profile.p_end - profile.p_mid) * ratio
    return max(0.0001, float(p))


def _new_memories_today(rng: random.Random, profile: MemoryProfile) -> int:
    if rng.random() > max(0.0, min(1.0, profile.new_prob)):
        return 0
    lo = max(0, int(profile.new_min))
    hi = max(lo, int(profile.new_max))
    return rng.randint(lo, hi)


def _pick_profile_for_new_memory(rng: random.Random, profiles: List[MemoryProfile]) -> MemoryProfile:
    """按画像权重选择新增记忆来源。"""
    if not profiles:
        raise ValueError("profiles 不能为空")
    weights = [max(0.0001, float(p.new_prob)) for p in profiles]
    total_w = sum(weights)
    r = rng.random() * total_w
    acc = 0.0
    for i, p in enumerate(profiles):
        acc += weights[i]
        if r <= acc:
            return p
    return profiles[-1]


def simulate_profiles_once(
    *,
    years: int,
    seed: int,
    useful_boost: int,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    profiles: List[MemoryProfile],
    recall_topk: int,
    useful_prob: float,
    daily_calls_min: int,
    daily_calls_max: int,
    forget_speed: float,
    consolidate_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> Tuple[Dict[str, Dict[str, float]], List[float]]:
    rng = random.Random(seed)
    total_days = years * 365

    memories: List[Memory] = []
    for profile in profiles:
        for _ in range(profile.count):
            useful_count = 0
            useful_score = 0.0
            cycle = forget_cycle_days(
                useful_score, cycle_1d, cycle_7d, cycle_30d, forget_speed, tier0_speed, tier1_speed, tier2_speed
            )
            memories.append(
                Memory(
                    is_active=False,
                    strength=profile.initial_strength,
                    useful_count=useful_count,
                    useful_score=useful_score,
                    created_day=0,
                    last_used_day=0,
                    next_decay_day=cycle,
                    use_prob=0.0,
                    profile=profile.name,
                )
            )

    profile_map = {p.name: p for p in profiles}
    peak_alive_total = len(memories)
    initial_total = len(memories)
    monthly_alive: List[float] = []

    for day in range(1, total_days + 1):
        # 1) 限额召回：每天 10~30 次，每次最多 10 条
        calls = rng.randint(max(1, daily_calls_min), max(daily_calls_min, daily_calls_max))
        recall_slots = calls * max(1, recall_topk)

        # 2) 每次召回调用都刷新 1 条随机样本记忆
        for _ in range(calls):
            profile = _pick_profile_for_new_memory(rng, profiles)
            useful_count = 0
            useful_score = 0.0
            cycle = forget_cycle_days(
                useful_score, cycle_1d, cycle_7d, cycle_30d, forget_speed, tier0_speed, tier1_speed, tier2_speed
            )
            memories.append(
                Memory(
                    is_active=False,
                    strength=profile.initial_strength,
                    useful_count=useful_count,
                    useful_score=useful_score,
                    created_day=day,
                    last_used_day=day,
                    next_decay_day=day + cycle,
                    use_prob=0.0,
                    profile=profile.name,
                )
            )

        alive_by_profile: Dict[str, List[Memory]] = {}
        for m in memories:
            if m.deleted_day is None and not m.is_active:
                alive_by_profile.setdefault(m.profile, []).append(m)

        if recall_slots > 0 and alive_by_profile:
            weights = [
                _profile_day_weight(day=day, total_days=total_days, profile=p)
                for p in profiles
            ]
            total_w = sum(weights)
            if total_w <= 0:
                weights = [1.0 for _ in profiles]
                total_w = float(len(weights))
            normalized = [w / total_w for w in weights]

            # 将槽位按画像分配
            slots_by_profile = {p.name: 0 for p in profiles}
            for _ in range(recall_slots):
                r = rng.random()
                acc = 0.0
                picked = profiles[-1].name
                for i, p in enumerate(profiles):
                    acc += normalized[i]
                    if r <= acc:
                        picked = p.name
                        break
                slots_by_profile[picked] += 1

            # 被召回后，单条仅 5% 概率“有用”
            p_useful = max(0.0, min(1.0, useful_prob))
            for pname, slots in slots_by_profile.items():
                bucket = alive_by_profile.get(pname, [])
                if not bucket:
                    continue
                for _ in range(slots):
                    m = bucket[rng.randrange(0, len(bucket))]
                    useful = rng.random() < p_useful
                    _apply_recall_result(
                        m=m,
                        day=day,
                        useful=useful,
                        useful_boost=useful_boost,
                        consolidate_speed=consolidate_speed,
                        cycle_1d=cycle_1d,
                        cycle_7d=cycle_7d,
                        cycle_30d=cycle_30d,
                        forget_speed=forget_speed,
                        tier0_speed=tier0_speed,
                        tier1_speed=tier1_speed,
                        tier2_speed=tier2_speed,
                    )

        # 3) 自然遗忘（仅按周期衰减）
        for m in memories:
            if m.deleted_day is not None:
                continue
            profile = profile_map[m.profile]
            if profile.mode == "yearly_once":
                # 年度复习型额外自然触发一次“温习”
                if _profile_use_happens(m, day, total_days, rng, profile):
                    _apply_recall_result(
                        m=m,
                        day=day,
                        useful=True,
                        useful_boost=useful_boost,
                        consolidate_speed=consolidate_speed,
                        cycle_1d=cycle_1d,
                        cycle_7d=cycle_7d,
                        cycle_30d=cycle_30d,
                        forget_speed=forget_speed,
                        tier0_speed=tier0_speed,
                        tier1_speed=tier1_speed,
                        tier2_speed=tier2_speed,
                    )

            _apply_natural_decay_if_needed(
                m=m,
                day=day,
                cycle_1d=cycle_1d,
                cycle_7d=cycle_7d,
                cycle_30d=cycle_30d,
                forget_speed=forget_speed,
                tier0_speed=tier0_speed,
                tier1_speed=tier1_speed,
                tier2_speed=tier2_speed,
            )
            if m.strength <= 0:
                m.deleted_day = day

        alive_total = sum(1 for m in memories if m.deleted_day is None)
        if alive_total > peak_alive_total:
            peak_alive_total = alive_total
        if day % 30 == 0 or day == total_days:
            monthly_alive.append(float(alive_total))

    by_name: Dict[str, List[Memory]] = {}
    for m in memories:
        by_name.setdefault(m.profile, []).append(m)

    out: Dict[str, Dict[str, float]] = {}
    for name, arr in by_name.items():
        lifetimes = [
            (m.deleted_day if m.deleted_day is not None else total_days) - m.created_day
            for m in arr
        ]
        deleted = [m for m in arr if m.deleted_day is not None]
        survive = [m for m in arr if m.deleted_day is None]
        out[name] = {
            "count": float(len(arr)),
            "alive_end": float(len(survive)),
            "deleted_ratio": len(deleted) / len(arr) if arr else 0.0,
            "survive_10y_ratio": len(survive) / len(arr) if arr else 0.0,
            "avg_lifetime_days": statistics.mean(lifetimes) if lifetimes else 0.0,
            "p90_lifetime_days": statistics.quantiles(lifetimes, n=10)[8] if len(lifetimes) >= 10 else (statistics.mean(lifetimes) if lifetimes else 0.0),
        }
    out["_overall"] = {
        "initial_total": float(initial_total),
        "peak_alive_total": float(peak_alive_total),
        "alive_end_total": float(sum(1 for m in memories if m.deleted_day is None)),
        "created_total": float(len(memories)),
        "forgotten_total": float(sum(1 for m in memories if m.deleted_day is not None)),
        "tier0_alive": float(sum(1 for m in memories if m.deleted_day is None and _memory_tier(m.useful_score) == 0)),
        "tier1_alive": float(sum(1 for m in memories if m.deleted_day is None and _memory_tier(m.useful_score) == 1)),
        "tier2_alive": float(sum(1 for m in memories if m.deleted_day is None and _memory_tier(m.useful_score) == 2)),
    }
    return out, monthly_alive


def run_grid(
    years: int,
    repeats: int,
    seed: int,
    strengths: List[int],
    boosts: List[int],
    initial_memories: int,
    daily_new_memories: int,
    active_ratio: float,
    cycle_1d: int,
    cycle_7d: int,
    cycle_30d: int,
    forget_speed: float,
    consolidate_speed: float,
    tier0_speed: float,
    tier1_speed: float,
    tier2_speed: float,
) -> List[Tuple[Tuple[int, int], Dict[str, float]]]:
    rows: List[Tuple[Tuple[int, int], Dict[str, float]]] = []
    for s in strengths:
        for b in boosts:
            agg: Dict[str, float] = {
                "passive_total": 0.0,
                "deleted_ratio": 0.0,
                "survive_10y_ratio": 0.0,
                "avg_lifetime_days": 0.0,
                "p90_lifetime_days": 0.0,
            }
            for i in range(repeats):
                result = simulate_once(
                    years=years,
                    seed=seed + i,
                    initial_strength=s,
                    useful_boost=b,
                    initial_memories=initial_memories,
                    daily_new_memories=daily_new_memories,
                    active_ratio=active_ratio,
                    cycle_1d=cycle_1d,
                    cycle_7d=cycle_7d,
                    cycle_30d=cycle_30d,
                    forget_speed=forget_speed,
                    consolidate_speed=consolidate_speed,
                    tier0_speed=tier0_speed,
                    tier1_speed=tier1_speed,
                    tier2_speed=tier2_speed,
                )
                for k, v in result.items():
                    agg[k] += v
            for k in agg:
                agg[k] /= repeats
            rows.append(((s, b), agg))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆衰减10年模拟器")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", type=str, default="profiles", choices=["profiles", "grid"])
    parser.add_argument("--initial-memories", type=int, default=2000)
    parser.add_argument("--daily-new-memories", type=int, default=15)
    parser.add_argument("--active-ratio", type=float, default=0.05)
    parser.add_argument("--cycle-1d", type=int, default=1)
    parser.add_argument("--cycle-7d", type=int, default=7)
    parser.add_argument("--cycle-30d", type=int, default=30)
    parser.add_argument("--forget-speed", type=float, default=1.0, help="全局遗忘速度倍率；越大忘得越快")
    parser.add_argument("--consolidate-speed", type=float, default=1.0, help="巩固速度倍率；越大升档越快")
    parser.add_argument("--tier0-forget-speed", type=float, default=1.0, help="0档遗忘速度倍率")
    parser.add_argument("--tier1-forget-speed", type=float, default=1.0, help="1档遗忘速度倍率")
    parser.add_argument("--tier2-forget-speed", type=float, default=1.0, help="2档遗忘速度倍率")
    parser.add_argument("--recall-topk", type=int, default=10)
    parser.add_argument("--useful-prob", type=float, default=0.05)
    parser.add_argument("--daily-calls-min", type=int, default=10)
    parser.add_argument("--daily-calls-max", type=int, default=30)
    parser.add_argument(
        "--strengths",
        type=str,
        default="4,6,8,10",
        help="初始强度候选，如 4,6,8,10",
    )
    parser.add_argument(
        "--boosts",
        type=str,
        default="1,2",
        help="有用时强度增量候选，如 1,2",
    )
    args = parser.parse_args()

    strengths = [int(x.strip()) for x in args.strengths.split(",") if x.strip()]
    boosts = [int(x.strip()) for x in args.boosts.split(",") if x.strip()]

    if args.mode == "grid":
        rows = run_grid(
            years=args.years,
            repeats=args.repeats,
            seed=args.seed,
            strengths=strengths,
            boosts=boosts,
            initial_memories=args.initial_memories,
            daily_new_memories=args.daily_new_memories,
            active_ratio=args.active_ratio,
            cycle_1d=args.cycle_1d,
            cycle_7d=args.cycle_7d,
            cycle_30d=args.cycle_30d,
            forget_speed=args.forget_speed,
            consolidate_speed=args.consolidate_speed,
            tier0_speed=args.tier0_forget_speed,
            tier1_speed=args.tier1_forget_speed,
            tier2_speed=args.tier2_forget_speed,
        )

        print("=== 10年记忆衰减模拟（网格模式） ===")
        print(
            "strength boost | deleted_ratio survive_10y avg_life_days p90_life_days passive_total"
        )
        for (s, b), m in rows:
            print(
                f"{s:>8} {b:>5} | "
                f"{m['deleted_ratio']:.3f} "
                f"{m['survive_10y_ratio']:.3f} "
                f"{m['avg_lifetime_days']:.1f} "
                f"{m['p90_lifetime_days']:.1f} "
                f"{m['passive_total']:.0f}"
            )
        return

    profiles = [
        # 短期背书：前60天高频，之后几乎不用
        MemoryProfile(name="短期背书后弃用", count=300, initial_strength=8, p_start=0.80, p_mid=0.05, p_end=0.00, hot_days=60, new_prob=0.30, new_min=0, new_max=2),
        # 天天可见：长期稳定高频
        MemoryProfile(name="日常天天可见", count=300, initial_strength=6, p_start=0.35, p_mid=0.35, p_end=0.35, new_prob=0.80, new_min=1, new_max=4),
        # 近期频繁，后续消失
        MemoryProfile(name="近期高频后消失", count=300, initial_strength=7, p_start=0.45, p_mid=0.10, p_end=0.00, hot_days=180, new_prob=0.45, new_min=0, new_max=3),
        # 每年温习一次
        MemoryProfile(name="每年温习一次", count=300, initial_strength=6, mode="yearly_once", new_prob=0.15, new_min=0, new_max=1),
        # 低频偶遇
        MemoryProfile(name="偶发低频记忆", count=300, initial_strength=5, p_start=0.03, p_mid=0.02, p_end=0.01, new_prob=0.70, new_min=1, new_max=6),
        # 几乎不用
        MemoryProfile(name="一次性噪声", count=300, initial_strength=4, p_start=0.005, p_mid=0.002, p_end=0.0, new_prob=0.95, new_min=3, new_max=10),
    ]
    # 按你的真实口径：记忆从 0 条开始，仅靠每日召回调用产生新记忆
    for p in profiles:
        p.count = 0

    print("=== 10年记忆衰减模拟（人群画像模式） ===")
    print(
        f"参数: cycle={args.cycle_1d}/{args.cycle_7d}/{args.cycle_30d}, "
        f"forget_speed={args.forget_speed}, consolidate_speed={args.consolidate_speed}, "
        f"tier_speeds={args.tier0_forget_speed}/{args.tier1_forget_speed}/{args.tier2_forget_speed}, "
        f"calls/day={args.daily_calls_min}~{args.daily_calls_max}, topk={args.recall_topk}, useful_prob={args.useful_prob}"
    )
    print("profile | deleted_ratio survive_10y avg_life_days p90_life_days count")
    for boost in boosts:
        agg: Dict[str, Dict[str, float]] = {}
        monthly_agg: List[float] = []
        for i in range(args.repeats):
            res, monthly = simulate_profiles_once(
                years=args.years,
                seed=args.seed + i,
                useful_boost=boost,
                cycle_1d=args.cycle_1d,
                cycle_7d=args.cycle_7d,
                cycle_30d=args.cycle_30d,
                profiles=profiles,
                recall_topk=args.recall_topk,
                useful_prob=args.useful_prob,
                daily_calls_min=args.daily_calls_min,
                daily_calls_max=args.daily_calls_max,
                forget_speed=args.forget_speed,
                consolidate_speed=args.consolidate_speed,
                tier0_speed=args.tier0_forget_speed,
                tier1_speed=args.tier1_forget_speed,
                tier2_speed=args.tier2_forget_speed,
            )
            for k, v in res.items():
                if k not in agg:
                    agg[k] = {kk: 0.0 for kk in v}
                for kk, vv in v.items():
                    agg[k][kk] += vv
            if not monthly_agg:
                monthly_agg = [0.0 for _ in monthly]
            for idx, val in enumerate(monthly):
                if idx < len(monthly_agg):
                    monthly_agg[idx] += val
        for k in agg:
            for kk in agg[k]:
                agg[k][kk] /= args.repeats
        if monthly_agg:
            monthly_agg = [x / args.repeats for x in monthly_agg]

        print(f"\n-- useful_boost={boost} --")
        ov = agg.get("_overall", {})
        if ov:
            init_total = ov.get("initial_total", 0.0)
            peak_total = ov.get("peak_alive_total", 0.0)
            end_total = ov.get("alive_end_total", 0.0)
            created_total = ov.get("created_total", 0.0)
            forgotten_total = ov.get("forgotten_total", 0.0)
            tier0_alive = ov.get("tier0_alive", 0.0)
            tier1_alive = ov.get("tier1_alive", 0.0)
            tier2_alive = ov.get("tier2_alive", 0.0)
            growth = (end_total / init_total) if init_total > 0 else 0.0
            print(
                f"总体膨胀: 初始={init_total:.0f} 峰值存活={peak_total:.0f} "
                f"10年末存活={end_total:.0f} 净增长倍数={growth:.2f}x "
                f"累计创建={created_total:.0f} 被遗忘={forgotten_total:.0f}"
            )
            print(
                f"分档存活: tier0={tier0_alive:.0f} tier1={tier1_alive:.0f} tier2={tier2_alive:.0f}"
            )
        for name in [p.name for p in profiles]:
            m = agg.get(
                name,
                {
                    "deleted_ratio": 0.0,
                    "survive_10y_ratio": 0.0,
                    "avg_lifetime_days": 0.0,
                    "p90_lifetime_days": 0.0,
                    "count": 0.0,
                },
            )
            print(
                f"{name} | "
                f"{m['deleted_ratio']:.3f} "
                f"{m['survive_10y_ratio']:.3f} "
                f"{m['avg_lifetime_days']:.1f} "
                f"{m['p90_lifetime_days']:.1f} "
                f"{m['count']:.0f}"
            )
        if monthly_agg:
            print("月度存活总量: " + ", ".join(f"M{idx+1}={int(v)}" for idx, v in enumerate(monthly_agg)))


if __name__ == "__main__":
    main()
