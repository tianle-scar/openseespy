"""
钢筋混凝土框架结构分析程序 (修正版 v2)
使用OpenSeesPy进行静力分析、时程分析、静力推覆分析

修正内容:
1. 内力图直接在框架上绘制（弯矩图、剪力图）
2. Pushover等效屈服点标注面积相等区域
3. 优化刚度计算说明

"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    import openseespy.opensees as ops
except Exception as exc:
    raise SystemExit(
        "OpenSeesPy导入失败。请先安装/检查环境：\n"
        "  pip install openseespy numpy matplotlib scipy\n\n"
        f"原始错误: {exc}"
    )

K_OVER_M2_PER_MPA = 1_000.0


def configure_chinese_font() -> None:
    """配置中文字体用于Matplotlib绘图"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
        "STHeiti", "PingFang SC", "WenQuanYi Micro Hei", "Droid Sans Fallback",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_found = False
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            font_found = True
            print(f"  使用字体: {name}")
            break
    
    if not font_found:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        print("  警告: 未找到中文字体，使用默认配置")
    
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.max_open_warning"] = 50
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["font.size"] = 10


@dataclass(frozen=True)
class RCFrameConfig:
    """钢筋混凝土框架配置参数"""
    
    n_bays: int = 3
    bay_width: float = 6.0
    story_heights: Tuple[float, ...] = (3.9, 3.6, 3.6, 3.6, 3.6)
    tributary_width: float = 6.0

    slab_dead: float = 5.5
    slab_live: float = 2.0
    beam_self: float = 5.0
    concrete_unit_weight: float = 25.0
    gravity_live_factor: float = 1.0
    seismic_live_factor: float = 0.5
    g: float = 9.81

    beam_b: float = 0.30
    beam_h: float = 0.65
    col_b: float = 0.55
    col_h: float = 0.55
    cover: float = 0.040
    
    beam_bar_dia: float = 0.022
    beam_bars_top: int = 4
    beam_bars_bottom: int = 4
    beam_stirrup_dia: float = 0.010
    beam_stirrup_spacing: float = 0.100
    col_bar_dia: float = 0.022
    col_bars_total: int = 12
    col_stirrup_dia: float = 0.010
    col_stirrup_spacing: float = 0.100

    concrete_grade: str = "C35"
    steel_grade: str = "HRB400"
    fc: float = 35.0 * K_OVER_M2_PER_MPA
    fc_design: float = 16.7 * K_OVER_M2_PER_MPA
    fy: float = 400.0 * K_OVER_M2_PER_MPA
    ec: float = 31_500.0 * K_OVER_M2_PER_MPA
    es: float = 200_000.0 * K_OVER_M2_PER_MPA

    damping_ratio: float = 0.05
    modal_count: int = 6
    pushover_target_drift: float = 0.04
    pushover_step: float = 0.0005
    
    max_iterations: int = 100
    tolerance: float = 1.0e-5
    min_step: float = 0.0001
    
    th_dt: float = 0.02
    th_duration: float = 30.0
    th_pga_g: float = 0.10

    @property
    def n_stories(self) -> int:
        return len(self.story_heights)

    @property
    def total_width(self) -> float:
        return self.n_bays * self.bay_width

    @property
    def roof_height(self) -> float:
        return sum(self.story_heights)


def node_tag(story: int, bay: int, cfg: RCFrameConfig) -> int:
    return 1000 + story * 100 + bay


def bar_area(diameter: float) -> float:
    return math.pi * diameter**2 / 4.0


def story_elevations(cfg: RCFrameConfig) -> List[float]:
    elevations = [0.0]
    for height in cfg.story_heights:
        elevations.append(elevations[-1] + height)
    return elevations


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def define_materials_and_sections(cfg: RCFrameConfig) -> None:
    """定义材料和截面"""
    cover_conc = 1
    core_conc = 2
    steel = 3

    ops.uniaxialMaterial("Concrete01", cover_conc, -0.80 * cfg.fc, -0.0020, -0.20 * cfg.fc, -0.0040)
    ops.uniaxialMaterial("Concrete01", core_conc, -1.15 * cfg.fc, -0.0025, -0.30 * cfg.fc, -0.020)
    ops.uniaxialMaterial("Steel02", steel, cfg.fy, cfg.es, 0.01, 15.0, 0.925, 0.15)

    create_rc_rect_section(
        sec_tag=1, b=cfg.beam_b, h=cfg.beam_h, cover=cfg.cover,
        bar_d=cfg.beam_bar_dia, bars_top=cfg.beam_bars_top, bars_bottom=cfg.beam_bars_bottom,
        side_bars_each=0, cover_mat=cover_conc, core_mat=core_conc, steel_mat=steel,
    )

    create_rc_rect_section(
        sec_tag=2, b=cfg.col_b, h=cfg.col_h, cover=cfg.cover,
        bar_d=cfg.col_bar_dia, bars_top=4, bars_bottom=4,
        side_bars_each=2, cover_mat=cover_conc, core_mat=core_conc, steel_mat=steel,
    )


def create_rc_rect_section(
    sec_tag: int, b: float, h: float, cover: float, bar_d: float,
    bars_top: int, bars_bottom: int, side_bars_each: int,
    cover_mat: int, core_mat: int, steel_mat: int,
) -> None:
    """创建矩形钢筋混凝土纤维截面"""
    y_top = h / 2.0
    y_bot = -h / 2.0
    z_left = -b / 2.0
    z_right = b / 2.0
    y_core_top = y_top - cover
    y_core_bot = y_bot + cover
    z_core_left = z_left + cover
    z_core_right = z_right - cover

    ops.section("Fiber", sec_tag)
    ops.patch("rect", core_mat, 14, 8, y_core_bot, z_core_left, y_core_top, z_core_right)
    ops.patch("rect", cover_mat, 4, 8, y_core_top, z_left, y_top, z_right)
    ops.patch("rect", cover_mat, 4, 8, y_bot, z_left, y_core_bot, z_right)
    ops.patch("rect", cover_mat, 14, 2, y_core_bot, z_left, y_core_top, z_core_left)
    ops.patch("rect", cover_mat, 14, 2, y_core_bot, z_core_right, y_core_top, z_right)

    area = bar_area(bar_d)
    y_top_bar = y_top - cover
    y_bot_bar = y_bot + cover
    ops.layer("straight", steel_mat, bars_top, area, y_top_bar, z_core_left, y_top_bar, z_core_right)
    ops.layer("straight", steel_mat, bars_bottom, area, y_bot_bar, z_core_left, y_bot_bar, z_core_right)

    if side_bars_each > 0:
        ops.layer("straight", steel_mat, side_bars_each, area, y_core_bot, z_core_left, y_core_top, z_core_left)
        ops.layer("straight", steel_mat, side_bars_each, area, y_core_bot, z_core_right, y_core_top, z_core_right)


def build_frame_model(cfg: RCFrameConfig) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, int, int, str]]]:
    """建立框架模型"""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    coords: Dict[int, Tuple[float, float]] = {}
    elevations = story_elevations(cfg)
    
    for story, y in enumerate(elevations):
        for bay in range(cfg.n_bays + 1):
            tag = node_tag(story, bay, cfg)
            x = bay * cfg.bay_width
            ops.node(tag, x, y)
            coords[tag] = (x, y)
            if story == 0:
                ops.fix(tag, 1, 1, 1)

    define_materials_and_sections(cfg)
    ops.geomTransf("PDelta", 1)
    ops.beamIntegration("Lobatto", 1, 1, 5)
    ops.beamIntegration("Lobatto", 2, 2, 5)

    elements: List[Tuple[int, int, int, str]] = []
    
    for story in range(1, cfg.n_stories + 1):
        for bay in range(cfg.n_bays + 1):
            ele = 10_000 + story * 100 + bay
            i_node = node_tag(story - 1, bay, cfg)
            j_node = node_tag(story, bay, cfg)
            ops.element("forceBeamColumn", ele, i_node, j_node, 1, 2)
            elements.append((ele, i_node, j_node, "column"))

    for story in range(1, cfg.n_stories + 1):
        for bay in range(cfg.n_bays):
            ele = 20_000 + story * 100 + bay
            i_node = node_tag(story, bay, cfg)
            j_node = node_tag(story, bay + 1, cfg)
            ops.element("forceBeamColumn", ele, i_node, j_node, 1, 1)
            elements.append((ele, i_node, j_node, "beam"))

    assign_floor_masses(cfg)
    return coords, elements


def assign_floor_masses(cfg: RCFrameConfig) -> None:
    """分配楼层质量"""
    for story in range(1, cfg.n_stories + 1):
        for bay in range(cfg.n_bays + 1):
            trib_len = cfg.bay_width / 2.0 if bay in (0, cfg.n_bays) else cfg.bay_width
            floor_weight = (
                (cfg.slab_dead + cfg.seismic_live_factor * cfg.slab_live) * cfg.tributary_width
                + cfg.beam_self
            ) * trib_len
            column_weight = cfg.col_b * cfg.col_h * cfg.concrete_unit_weight * cfg.story_heights[story - 1]
            nodal_weight = floor_weight + column_weight
            mass = nodal_weight / cfg.g
            ops.mass(node_tag(story, bay, cfg), mass, mass, 0.0)


def apply_gravity_loads(cfg: RCFrameConfig, elements: List[Tuple[int, int, int, str]]) -> None:
    """施加重力荷载"""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    beam_line_load = -(
        (cfg.slab_dead + cfg.gravity_live_factor * cfg.slab_live) * cfg.tributary_width + cfg.beam_self
    )
    for ele, _, _, ele_type in elements:
        if ele_type == "beam":
            ops.eleLoad("-ele", ele, "-type", "-beamUniform", beam_line_load)

    col_line_weight = cfg.col_b * cfg.col_h * cfg.concrete_unit_weight
    for story in range(1, cfg.n_stories + 1):
        column_weight = col_line_weight * cfg.story_heights[story - 1]
        for bay in range(cfg.n_bays + 1):
            ops.load(node_tag(story, bay, cfg), 0.0, -column_weight, 0.0)


def setup_static_analysis(integrator_args: Tuple[object, ...], cfg: RCFrameConfig) -> None:
    """设置静力分析"""
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", cfg.tolerance, cfg.max_iterations)
    ops.algorithm("Newton")
    ops.integrator(*integrator_args)
    ops.analysis("Static")


def setup_transient_analysis(dt: float, cfg: RCFrameConfig) -> None:
    """设置动力分析"""
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", cfg.tolerance, cfg.max_iterations)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")


def analyze_one_step_improved(dt: float | None = None, cfg: RCFrameConfig | None = None) -> int:
    """执行一步分析，支持算法切换"""
    ok = ops.analyze(1) if dt is None else ops.analyze(1, dt)
    if ok == 0:
        return ok

    algorithms = [
        ("ModifiedNewton", "-initial"),
        ("ModifiedNewton",),
        ("KrylovNewton",),
        ("NewtonLineSearch", 0.8),
        ("NewtonLineSearch", 0.6),
        ("BFGS",),
        ("Broyden", 8),
        ("Newton", "-initial"),
    ]
    
    for alg in algorithms:
        try:
            ops.algorithm(*alg)
            ok = ops.analyze(1) if dt is None else ops.analyze(1, dt)
            if ok == 0:
                ops.algorithm("Newton")
                return 0
        except:
            continue
    
    if cfg is not None:
        try:
            ops.test("NormDispIncr", cfg.tolerance * 10, cfg.max_iterations * 2)
            ops.algorithm("Newton")
            ok = ops.analyze(1) if dt is None else ops.analyze(1, dt)
            ops.test("NormDispIncr", cfg.tolerance, cfg.max_iterations)
            if ok == 0:
                return 0
        except:
            pass
    
    ops.algorithm("Newton")
    return ok


def run_gravity_analysis(cfg: RCFrameConfig, steps: int = 10) -> None:
    """运行重力分析"""
    setup_static_analysis(("LoadControl", 1.0 / steps), cfg)
    for step in range(steps):
        ok = analyze_one_step_improved(cfg=cfg)
        if ok != 0:
            raise RuntimeError(f"重力分析在第 {step + 1}/{steps} 步失败")
    ops.loadConst("-time", 0.0)


def floor_average_displacements(cfg: RCFrameConfig) -> List[float]:
    """计算各层平均位移"""
    floor_u = [0.0]
    for story in range(1, cfg.n_stories + 1):
        vals = [ops.nodeDisp(node_tag(story, bay, cfg), 1) for bay in range(cfg.n_bays + 1)]
        floor_u.append(float(np.mean(vals)))
    return floor_u


def floor_average_velocities(cfg: RCFrameConfig) -> List[float]:
    """计算各层平均速度"""
    floor_v = [0.0]
    for story in range(1, cfg.n_stories + 1):
        vals = [ops.nodeVel(node_tag(story, bay, cfg), 1) for bay in range(cfg.n_bays + 1)]
        floor_v.append(float(np.mean(vals)))
    return floor_v


def floor_average_accelerations(cfg: RCFrameConfig) -> List[float]:
    """计算各层平均加速度"""
    floor_a = [0.0]
    for story in range(1, cfg.n_stories + 1):
        vals = [ops.nodeAccel(node_tag(story, bay, cfg), 1) for bay in range(cfg.n_bays + 1)]
        floor_a.append(float(np.mean(vals)))
    return floor_a


def story_drifts(cfg: RCFrameConfig) -> List[float]:
    """计算层间位移角"""
    floor_u = floor_average_displacements(cfg)
    return [
        (floor_u[story] - floor_u[story - 1]) / cfg.story_heights[story - 1]
        for story in range(1, cfg.n_stories + 1)
    ]


def collect_node_displacements(coords: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    """收集节点位移"""
    return {tag: (ops.nodeDisp(tag, 1), ops.nodeDisp(tag, 2)) for tag in coords}


def base_shear_x(cfg: RCFrameConfig) -> float:
    """计算基底剪力"""
    ops.reactions()
    return -sum(ops.nodeReaction(node_tag(0, bay, cfg), 1) for bay in range(cfg.n_bays + 1))


def base_reaction_y(cfg: RCFrameConfig) -> float:
    """计算竖向反力"""
    ops.reactions()
    return sum(ops.nodeReaction(node_tag(0, bay, cfg), 2) for bay in range(cfg.n_bays + 1))


def estimate_periods(cfg: RCFrameConfig) -> List[float]:
    """特征值分析估算周期"""
    try:
        lambdas = ops.eigen(cfg.modal_count)
    except Exception:
        lambdas = ops.eigen("-fullGenLapack", cfg.modal_count)
    periods = []
    for lam in lambdas:
        if lam > 0.0:
            periods.append(2.0 * math.pi / math.sqrt(lam))
    return periods


def get_mode_shapes(cfg: RCFrameConfig, num_modes: int = 3) -> List[List[float]]:
    """获取振型"""
    mode_shapes = []
    for mode in range(1, num_modes + 1):
        shape = [0.0]
        for story in range(1, cfg.n_stories + 1):
            eigenvector = ops.nodeEigenvector(node_tag(story, 0, cfg), mode, 1)
            shape.append(eigenvector)
        max_val = max(abs(v) for v in shape)
        if max_val > 1e-10:
            shape = [v / max_val for v in shape]
        mode_shapes.append(shape)
    return mode_shapes


def set_rayleigh_damping(cfg: RCFrameConfig, periods: List[float]) -> Tuple[float, float]:
    """设置Rayleigh阻尼"""
    if len(periods) < 2:
        return (0.0, 0.0)
    w1 = 2.0 * math.pi / periods[0]
    w2 = 2.0 * math.pi / periods[min(2, len(periods) - 1)]
    alpha_m = 2.0 * cfg.damping_ratio * w1 * w2 / (w1 + w2)
    beta_k = 2.0 * cfg.damping_ratio / (w1 + w2)
    ops.rayleigh(alpha_m, 0.0, 0.0, beta_k)
    return alpha_m, beta_k


def synthetic_ground_motion(cfg: RCFrameConfig) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成地震波"""
    dt = cfg.th_dt
    duration = cfg.th_duration
    pga = cfg.th_pga_g * cfg.g
    
    time = np.arange(0.0, duration + dt, dt)
    n_points = len(time)
    
    t_rise = duration * 0.15
    t_strong = duration * 0.35
    
    envelope = np.zeros(n_points)
    for i, t in enumerate(time):
        if t < t_rise:
            envelope[i] = (t / t_rise) ** 2
        elif t < t_strong:
            envelope[i] = 1.0 - 0.1 * ((t - t_rise) / (t_strong - t_rise))
        else:
            decay_rate = 2.5 / (duration - t_strong)
            envelope[i] = 0.9 * np.exp(-decay_rate * (t - t_strong))
    
    np.random.seed(42)
    fg = 2.5
    
    frequencies = [
        (0.8, 0.15), (1.5, 0.25), (fg, 0.35), (3.5, 0.20),
        (5.0, 0.12), (8.0, 0.08), (12.0, 0.05),
    ]
    
    signal = np.zeros(n_points)
    for freq, amp in frequencies:
        phase = np.random.uniform(0, 2 * np.pi)
        freq_mod = freq * (1 + 0.1 * np.sin(2 * np.pi * 0.3 * time))
        signal += amp * np.sin(2 * np.pi * freq_mod * time + phase)
    
    accel = envelope * signal
    accel -= np.mean(accel)
    
    accel_corrected = np.zeros_like(accel)
    accel_corrected[0] = accel[0]
    alpha = 0.995
    for i in range(1, len(accel)):
        accel_corrected[i] = alpha * accel_corrected[i-1] + alpha * (accel[i] - accel[i-1])
    
    max_abs = np.max(np.abs(accel_corrected))
    if max_abs > 1e-12:
        accel_corrected *= pga / max_abs
    
    taper_len = int(0.5 / dt)
    if taper_len > 0:
        taper = np.linspace(0, 1, taper_len)
        accel_corrected[:taper_len] *= taper
        accel_corrected[-taper_len:] *= taper[::-1]
    
    return time, accel_corrected


def get_element_forces(elements: List[Tuple[int, int, int, str]], 
                       coords: Dict[int, Tuple[float, float]]) -> Dict[int, Dict]:
    """
    获取各单元的内力
    返回: {ele_tag: {'i_node': tag, 'j_node': tag, 'type': str, 
                     'N_i': float, 'V_i': float, 'M_i': float,
                     'N_j': float, 'V_j': float, 'M_j': float}}
    """
    forces = {}
    for ele, i_node, j_node, ele_type in elements:
        try:
            # forceBeamColumn单元的局部力: [N_i, V_i, M_i, N_j, V_j, M_j]
            local_forces = ops.eleForce(ele)
            if len(local_forces) >= 6:
                forces[ele] = {
                    'i_node': i_node,
                    'j_node': j_node,
                    'type': ele_type,
                    'xi': coords[i_node][0],
                    'yi': coords[i_node][1],
                    'xj': coords[j_node][0],
                    'yj': coords[j_node][1],
                    'N_i': local_forces[0],  # 轴力 i端
                    'V_i': local_forces[1],  # 剪力 i端
                    'M_i': local_forces[2],  # 弯矩 i端
                    'N_j': local_forces[3],  # 轴力 j端
                    'V_j': local_forces[4],  # 剪力 j端
                    'M_j': local_forces[5],  # 弯矩 j端
                }
        except:
            pass
    return forces


def internal_force_table_rows(element_forces: Dict[int, Dict]) -> List[Dict[str, object]]:
    """整理构件内力表, 端部内力来自OpenSees, 跨中弯矩为端弯矩线性插值近似值。"""
    rows: List[Dict[str, object]] = []
    for ele, force in sorted(element_forces.items()):
        m_i = float(force.get("M_i", 0.0))
        m_j = float(force.get("M_j", 0.0))
        v_i = float(force.get("V_i", 0.0))
        v_j = float(force.get("V_j", 0.0))
        n_i = float(force.get("N_i", 0.0))
        n_j = float(force.get("N_j", 0.0))
        rows.append({
            "element": ele,
            "member_type": force["type"],
            "i_node": force["i_node"],
            "j_node": force["j_node"],
            "N_i_kN": n_i,
            "V_i_kN": v_i,
            "M_i_kN_m": m_i,
            "N_j_kN": n_j,
            "V_j_kN": v_j,
            "M_j_kN_m": m_j,
            "M_mid_kN_m_approx": 0.5 * (m_i + m_j),
            "V_mid_kN_approx": 0.5 * (v_i + v_j),
            "N_mid_kN_approx": 0.5 * (n_i + n_j),
        })
    return rows


def write_internal_force_outputs(
    element_forces: Dict[int, Dict],
    out_dir: Path,
    prefix: str,
) -> List[Dict[str, object]]:
    rows = internal_force_table_rows(element_forces)
    write_csv(out_dir / f"{prefix}_internal_forces.csv", rows)
    return rows


def estimate_member_yield_moment(cfg: RCFrameConfig, member_type: str) -> float:
    """估算梁、柱截面屈服弯矩(kN*m), 用于塑性铰可视化判别。"""
    if member_type == "beam":
        b, h = cfg.beam_b, cfg.beam_h
        bar_d = cfg.beam_bar_dia
        as_tension = cfg.beam_bars_bottom * bar_area(cfg.beam_bar_dia)
    else:
        b, h = cfg.col_b, cfg.col_h
        bar_d = cfg.col_bar_dia
        as_tension = (cfg.col_bars_total / 2.0) * bar_area(cfg.col_bar_dia)

    d = h - cfg.cover - bar_d / 2.0
    fc_mpa = cfg.fc_design / K_OVER_M2_PER_MPA
    fy_mpa = cfg.fy / K_OVER_M2_PER_MPA
    as_mm2 = as_tension * 1.0e6
    b_mm = b * 1000.0
    d_mm = d * 1000.0
    x_mm = as_mm2 * fy_mpa / max(fc_mpa * b_mm, 1.0e-9)
    x_mm = min(x_mm, 0.55 * d_mm)
    lever_arm_mm = d_mm - 0.5 * x_mm
    my_kn_m = as_mm2 * fy_mpa * lever_arm_mm / 1.0e6
    return float(max(my_kn_m, 1.0e-6))


def plastic_hinge_definition(cfg: RCFrameConfig) -> Dict[str, object]:
    """返回塑性铰判别定义，便于报告和程序结果保持一致。"""
    return {
        "method": "member-end moment demand-to-yield ratio",
        "beam_yield_moment_kN_m": estimate_member_yield_moment(cfg, "beam"),
        "column_yield_moment_kN_m": estimate_member_yield_moment(cfg, "column"),
        "thresholds": {
            "elastic": "DCR < 0.80",
            "near_yield": "0.80 <= DCR < 1.00",
            "plastic_hinge": "1.00 <= DCR < 1.30",
            "severe_plastic": "DCR >= 1.30",
        },
        "note": (
            "DCR = |member end moment| / My. My is estimated by rectangular RC "
            "section force equilibrium. This is a visualization-oriented hinge "
            "indicator; detailed hinge rotation capacity should be calibrated "
            "with section analysis or code acceptance criteria."
        ),
    }


def classify_hinge_state(dcr: float) -> str:
    if dcr >= 1.30:
        return "severe_plastic"
    if dcr >= 1.00:
        return "plastic_hinge"
    if dcr >= 0.80:
        return "near_yield"
    return "elastic"


def collect_plastic_hinges(
    element_forces: Dict[int, Dict],
    cfg: RCFrameConfig,
) -> List[Dict[str, object]]:
    """根据单元端弯矩需求比收集塑性铰判别结果。"""
    hinge_rows: List[Dict[str, object]] = []
    my_by_type = {
        "beam": estimate_member_yield_moment(cfg, "beam"),
        "column": estimate_member_yield_moment(cfg, "column"),
    }
    for ele, force in sorted(element_forces.items()):
        member_type = force["type"]
        my = my_by_type.get(member_type, my_by_type["beam"])
        for end_key, node_key, moment_key, x_key, y_key in (
            ("i", "i_node", "M_i", "xi", "yi"),
            ("j", "j_node", "M_j", "xj", "yj"),
        ):
            moment = float(force.get(moment_key, 0.0))
            dcr = abs(moment) / max(my, 1.0e-9)
            hinge_rows.append({
                "element": ele,
                "member_type": member_type,
                "end": end_key,
                "node": force[node_key],
                "x_m": float(force[x_key]),
                "y_m": float(force[y_key]),
                "moment_kN_m": moment,
                "yield_moment_kN_m": my,
                "dcr": dcr,
                "state": classify_hinge_state(dcr),
            })
    return hinge_rows


def summarize_plastic_hinges(hinges: List[Dict[str, object]]) -> Dict[str, object]:
    counts = {"elastic": 0, "near_yield": 0, "plastic_hinge": 0, "severe_plastic": 0}
    for hinge in hinges:
        counts[str(hinge["state"])] = counts.get(str(hinge["state"]), 0) + 1
    active = [h for h in hinges if str(h["state"]) in ("plastic_hinge", "severe_plastic")]
    return {
        "counts": counts,
        "active_hinge_count": len(active),
        "max_dcr": max((float(h["dcr"]) for h in hinges), default=0.0),
        "active_hinges": active,
    }


def write_plastic_hinge_outputs(
    hinges: List[Dict[str, object]],
    cfg: RCFrameConfig,
    out_dir: Path,
    prefix: str,
) -> Dict[str, object]:
    write_csv(out_dir / f"{prefix}_plastic_hinges.csv", hinges)
    summary = {
        "definition": plastic_hinge_definition(cfg),
        **summarize_plastic_hinges(hinges),
    }
    write_json(out_dir / f"{prefix}_plastic_hinges.json", summary)
    return summary


def plot_plastic_hinge_distribution(
    coords: Dict[int, Tuple[float, float]],
    elements: List[Tuple[int, int, int, str]],
    hinges: List[Dict[str, object]],
    cfg: RCFrameConfig,
    path: Path,
    title: str,
) -> None:
    """绘制塑性铰需求比分布图。"""
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [3.0, 1.05]}
    )
    ax.set_facecolor("#f8f9fa")

    for _, i_node, j_node, ele_type in elements:
        xi, yi = coords[i_node]
        xj, yj = coords[j_node]
        color = "#95a5a6" if ele_type == "column" else "#7f8c8d"
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=2.2, alpha=0.70, zorder=1)

    style = {
        "elastic": ("#bdc3c7", 32, "o", "弹性 DCR<0.80"),
        "near_yield": ("#f1c40f", 74, "o", "接近屈服 0.80-1.00"),
        "plastic_hinge": ("#e67e22", 126, "o", "塑性铰 1.00-1.30"),
        "severe_plastic": ("#c0392b", 180, "*", "严重塑性 DCR>=1.30"),
    }

    for state in ("elastic", "near_yield", "plastic_hinge", "severe_plastic"):
        group = [h for h in hinges if h["state"] == state]
        if not group:
            continue
        color, size, marker, label = style[state]
        ax.scatter(
            [float(h["x_m"]) for h in group],
            [float(h["y_m"]) for h in group],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=label,
        )

    critical = sorted(hinges, key=lambda h: float(h["dcr"]), reverse=True)[:8]
    for hinge in critical:
        if float(hinge["dcr"]) < 0.8:
            continue
        ax.annotate(
            f"{hinge['member_type'][0].upper()}{hinge['element']}-{hinge['end']}\nDCR={float(hinge['dcr']):.2f}",
            xy=(float(hinge["x_m"]), float(hinge["y_m"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, edgecolor="#bdc3c7"),
        )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("水平坐标 (m)")
    ax.set_ylabel("高度 (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, cfg.total_width + 1.0)
    ax.set_ylim(-0.8, cfg.roof_height + 1.0)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)

    counts = summarize_plastic_hinges(hinges)["counts"]
    labels = ["弹性", "近屈服", "塑性铰", "严重塑性"]
    values = [counts["elastic"], counts["near_yield"], counts["plastic_hinge"], counts["severe_plastic"]]
    colors = [style[k][0] for k in ("elastic", "near_yield", "plastic_hinge", "severe_plastic")]
    ax_bar.bar(labels, values, color=colors, edgecolor="white")
    ax_bar.set_title("铰状态统计", fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("端部数量")
    ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--")
    for tick in ax_bar.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    for idx, val in enumerate(values):
        ax_bar.text(idx, val + 0.2, str(val), ha="center", va="bottom", fontsize=10)

    info = plastic_hinge_definition(cfg)
    text = (
        f"梁 My={info['beam_yield_moment_kN_m']:.1f} kN·m\n"
        f"柱 My={info['column_yield_moment_kN_m']:.1f} kN·m\n"
        "DCR=|M端|/My"
    )
    ax_bar.text(
        0.02, 0.98, text, transform=ax_bar.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fbff", edgecolor="#3498db", alpha=0.92),
    )

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def find_equivalent_yield_point(disp: np.ndarray, force: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    计算等效屈服点 (能量等效法).
    
    返回: (dy, fy, k0, k1, area_actual)
    
    说明：
    - 采用割线刚度而非切线刚度的原因：
      1. 切线刚度是曲线在某点的导数，对局部噪声敏感
      2. 割线刚度代表从原点到某点的平均刚度，更稳定
      3. 工程规范(ATC-40, FEMA-356)推荐用弹性阶段的割线刚度
      4. 取75%峰值荷载前的割线刚度，避免将塑性段纳入
    """
    disp = np.asarray(disp, dtype=float)
    force = np.asarray(force, dtype=float)
    valid = np.isfinite(disp) & np.isfinite(force)
    disp = disp[valid]
    force = np.abs(force[valid])

    if len(disp) < 3:
        du = float(disp[-1]) if len(disp) else 1.0
        vmax = float(force.max()) if len(force) else 1.0
        dy = max(du * 0.3, 1.0e-9)
        return dy, vmax * 0.7, vmax * 0.7 / dy, 0.0, 0.0

    order = np.argsort(disp)
    disp = disp[order]
    force = force[order]
    unique_disp, unique_idx = np.unique(disp, return_index=True)
    disp = unique_disp
    force = force[unique_idx]
    positive = disp > 0.0
    disp = disp[positive]
    force = force[positive]

    if len(disp) < 3:
        du = float(disp[-1])
        vmax = float(force.max())
        dy = max(du * 0.3, 1.0e-9)
        return dy, vmax * 0.7, vmax * 0.7 / dy, 0.0, 0.0

    try:
        spline = UnivariateSpline(disp, force, s=len(disp) * 0.2)
        disp_smooth = np.linspace(float(disp.min()), float(disp.max()), 800)
        force_smooth = np.maximum(spline(disp_smooth), 0.0)
    except Exception:
        disp_smooth = disp
        force_smooth = force

    max_force_idx = int(np.argmax(force_smooth))
    max_force = float(force_smooth[max_force_idx])
    disp_u = float(disp_smooth[-1])
    area_actual = float(np.trapezoid(force_smooth, disp_smooth))

    # ========== 割线刚度计算 ==========
    # 取达到75%峰值荷载时的割线刚度作为初始刚度
    # 这样可以避免把明显的塑性段纳入初始刚度计算
    rising_disp = disp_smooth[:max_force_idx + 1]
    rising_force = force_smooth[:max_force_idx + 1]
    target_force = 0.75 * max_force
    try:
        disp_at_target = float(np.interp(target_force, rising_force, rising_disp))
    except Exception:
        disp_at_target = float(disp_smooth[max(1, int(0.15 * len(disp_smooth)))])

    # 割线刚度 = V / d (从原点到目标点)
    k0 = target_force / max(disp_at_target, 1.0e-9)
    if not np.isfinite(k0) or k0 <= 0.0:
        early_count = max(3, min(12, len(disp) // 10))
        k0 = float(np.polyfit(disp[:early_count], force[:early_count], 1)[0])
    if not np.isfinite(k0) or k0 <= 0.0:
        k0 = max_force / max(disp_u * 0.3, 1.0e-9)

    alpha = 0.03  # 屈服后刚度比

    def area_bilinear(dy: float) -> float:
        fy = k0 * dy
        fu = fy + alpha * k0 * (disp_u - dy)
        return 0.5 * fy * dy + 0.5 * (fy + fu) * (disp_u - dy)

    lower = max(float(disp_smooth[0]), 1.0e-9)
    upper = min(0.90 * disp_u, 1.20 * max_force / k0)
    if upper <= lower:
        upper = 0.90 * disp_u

    try:
        dy_solution = float(brentq(lambda dy: area_bilinear(dy) - area_actual, lower, upper))
    except Exception:
        fy_fallback = 0.85 * max_force
        dy_solution = fy_fallback / k0

    fy_solution = k0 * dy_solution

    max_fy = 1.05 * max_force
    if fy_solution > max_fy:
        fy_solution = max_fy
        dy_solution = fy_solution / k0

    return dy_solution, fy_solution, k0, alpha * k0, area_actual


def concrete_constitutive_info(cfg: RCFrameConfig) -> Dict[str, Dict[str, object]]:
    """混凝土本构信息"""
    return {
        "cover_unconfined_concrete": {
            "material": "OpenSees Concrete01",
            "description": "非约束保护层混凝土",
            "fpc_MPa": -0.80 * cfg.fc / K_OVER_M2_PER_MPA,
            "epsc0": -0.0020,
            "fpcu_MPa": -0.20 * cfg.fc / K_OVER_M2_PER_MPA,
            "epsU": -0.0040,
        },
        "core_confined_concrete": {
            "material": "OpenSees Concrete01",
            "description": "箍筋约束核心混凝土",
            "fpc_MPa": -1.15 * cfg.fc / K_OVER_M2_PER_MPA,
            "epsc0": -0.0025,
            "fpcu_MPa": -0.30 * cfg.fc / K_OVER_M2_PER_MPA,
            "epsU": -0.020,
        },
    }


def steel_constitutive_info(cfg: RCFrameConfig) -> Dict[str, object]:
    """钢筋本构信息"""
    return {
        "material": "OpenSees Steel02",
        "description": "钢筋 (Giuffre-Menegotto-Pinto模型)",
        "fy_MPa": cfg.fy / K_OVER_M2_PER_MPA,
        "E0_MPa": cfg.es / K_OVER_M2_PER_MPA,
        "b": 0.01,
        "R0": 15.0,
        "cR1": 0.925,
        "cR2": 0.15,
    }


# ==================== 绘图函数 ====================

def plot_structure_info(cfg: RCFrameConfig, path: Path) -> None:
    """绘制结构信息汇总图"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_facecolor('#f8f9fa')
    elevations = story_elevations(cfg)
    
    for i in range(cfg.n_stories):
        y = elevations[i]
        h = cfg.story_heights[i]
        for bay in range(cfg.n_bays + 1):
            x = bay * cfg.bay_width
            ax1.plot([x, x], [y, y + h], color='#2c3e50', linewidth=4)
        ax1.plot([0, cfg.total_width], [y + h, y + h], color='#3498db', linewidth=3)
    
    ax1.plot([0, cfg.total_width], [0, 0], color='#7f8c8d', linewidth=5)
    for x in np.linspace(0, cfg.total_width, 8):
        ax1.plot([x, x - 0.3], [0, -0.5], color='#7f8c8d', linewidth=2)
    
    for i, elev in enumerate(elevations[1:], 1):
        ax1.text(-1.5, elev, f'F{i}', fontsize=12, fontweight='bold', va='center')
        ax1.text(cfg.total_width + 0.5, elev, f'{elev:.1f}m', fontsize=10, va='center')
    
    ax1.set_xlim(-3, cfg.total_width + 3)
    ax1.set_ylim(-2, cfg.roof_height + 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('结构简图', fontsize=14, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    
    info_data = [
        ['项目', '数值', '单位'],
        ['结构层数', f'{cfg.n_stories}', '层'],
        ['结构跨数', f'{cfg.n_bays}', '跨'],
        ['总高度', f'{cfg.roof_height:.1f}', 'm'],
        ['总宽度', f'{cfg.total_width:.1f}', 'm'],
        ['标准层高', f'{cfg.story_heights[1]:.1f}', 'm'],
        ['跨度', f'{cfg.bay_width:.1f}', 'm'],
        ['梁截面', f'{cfg.beam_b*1000:.0f}x{cfg.beam_h*1000:.0f}', 'mm'],
        ['柱截面', f'{cfg.col_b*1000:.0f}x{cfg.col_h*1000:.0f}', 'mm'],
    ]
    
    table = ax2.table(cellText=info_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax2.set_title('结构基本信息', fontsize=14, fontweight='bold', pad=20)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    mat_data = [
        ['材料参数', '数值'],
        ['混凝土等级', cfg.concrete_grade],
        ['钢筋等级', f'{cfg.steel_grade}(三级钢)'],
        ['fc (MPa)', f'{cfg.fc/K_OVER_M2_PER_MPA:.1f}'],
        ['fy (MPa)', f'{cfg.fy/K_OVER_M2_PER_MPA:.0f}'],
        ['Ec (GPa)', f'{cfg.ec/K_OVER_M2_PER_MPA/1000:.1f}'],
        ['Es (GPa)', f'{cfg.es/K_OVER_M2_PER_MPA/1000:.0f}'],
    ]
    
    table2 = ax3.table(cellText=mat_data, loc='center', cellLoc='center',
                       colWidths=[0.5, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 1.8)
    for i in range(2):
        table2[(0, i)].set_facecolor('#27ae60')
        table2[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax3.set_title('材料参数', fontsize=14, fontweight='bold', pad=20)
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    load_data = [
        ['荷载参数', '数值'],
        ['楼面恒载 (kN/m²)', f'{cfg.slab_dead:.1f}'],
        ['楼面活载 (kN/m²)', f'{cfg.slab_live:.1f}'],
        ['梁自重 (kN/m)', f'{cfg.beam_self:.1f}'],
        ['地震PGA (g)', f'{cfg.th_pga_g:.2f}'],
        ['阻尼比', f'{cfg.damping_ratio:.0%}'],
        ['地震持时 (s)', f'{cfg.th_duration:.0f}'],
    ]
    
    table3 = ax4.table(cellText=load_data, loc='center', cellLoc='center',
                       colWidths=[0.5, 0.4])
    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1.2, 1.8)
    for i in range(2):
        table3[(0, i)].set_facecolor('#e74c3c')
        table3[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('荷载参数', fontsize=14, fontweight='bold', pad=20)
    
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor('#f8f9fa')
    
    stories = np.arange(1, cfg.n_stories + 1)
    heights = list(cfg.story_heights)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(stories)))
    bars = ax5.bar(stories, heights, color=colors, edgecolor='white', linewidth=2)
    
    for bar, h in zip(bars, heights):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{h:.1f}m', ha='center', fontsize=11, fontweight='bold')
    
    ax5.set_xlabel('楼层', fontsize=12)
    ax5.set_ylabel('层高 (m)', fontsize=12)
    ax5.set_title('各层层高分布', fontsize=14, fontweight='bold')
    ax5.set_xticks(stories)
    ax5.set_xticklabels([f'F{i}' for i in stories])
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, max(heights) * 1.2)
    
    plt.suptitle('钢筋混凝土框架结构信息汇总', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_model(
    coords: Dict[int, Tuple[float, float]],
    elements: List[Tuple[int, int, int, str]],
    cfg: RCFrameConfig,
    path: Path,
    title: str,
) -> None:
    """绘制框架模型"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#f8f9fa')
    
    elevations = story_elevations(cfg)
    colors_floor = plt.cm.Blues(np.linspace(0.08, 0.2, cfg.n_stories))
    for i in range(cfg.n_stories):
        rect = Rectangle((0, elevations[i]), cfg.total_width, cfg.story_heights[i],
                         facecolor=colors_floor[i], alpha=0.3, edgecolor='none')
        ax.add_patch(rect)
        ax.text(-0.8, elevations[i] + cfg.story_heights[i]/2, f'F{i+1}', 
               fontsize=11, fontweight='bold', va='center', ha='right', color='#2c3e50')
    
    for _, i_node, j_node, ele_type in elements:
        xi, yi = coords[i_node]
        xj, yj = coords[j_node]
        if ele_type == "column":
            color = "#2c3e50"
            linewidth = 5.0
        else:
            color = "#3498db"
            linewidth = 4.0
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=linewidth, solid_capstyle='round')
    
    xs = [v[0] for v in coords.values()]
    ys = [v[1] for v in coords.values()]
    ax.scatter(xs, ys, s=50, color="#e74c3c", zorder=5, edgecolor='white', linewidth=1.5)
    
    for tag, (x, y) in coords.items():
        if y == 0:
            triangle = Polygon([(x-0.3, y), (x+0.3, y), (x, y-0.35)], 
                              closed=True, facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=2)
            ax.add_patch(triangle)
            ax.plot([x-0.4, x+0.4], [y-0.35, y-0.35], color='#2c3e50', linewidth=2)
            for dx in np.linspace(-0.35, 0.35, 5):
                ax.plot([x+dx, x+dx-0.12], [-0.35, -0.55], color='#2c3e50', linewidth=1.5)
    
    y_dim = -1.2
    for bay in range(cfg.n_bays):
        x1, x2 = bay * cfg.bay_width, (bay + 1) * cfg.bay_width
        ax.annotate('', xy=(x1, y_dim), xytext=(x2, y_dim),
                   arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=1.5))
        ax.text((x1+x2)/2, y_dim-0.25, f'{cfg.bay_width}m', fontsize=10, ha='center', color='#e74c3c')
    
    x_dim = cfg.total_width + 0.8
    for i, (h, elev) in enumerate(zip(cfg.story_heights, elevations[:-1])):
        ax.annotate('', xy=(x_dim, elev), xytext=(x_dim, elev + h),
                   arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
        ax.text(x_dim + 0.25, elev + h/2, f'{h}m', fontsize=9, va='center', color='#27ae60')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(-1.5, cfg.total_width + 2)
    ax.set_ylim(-1.8, cfg.roof_height + 1)
    
    legend_elements = [
        Line2D([0], [0], color='#2c3e50', linewidth=5, label=f'柱 {cfg.col_b*1000:.0f}x{cfg.col_h*1000:.0f}'),
        Line2D([0], [0], color='#3498db', linewidth=4, label=f'梁 {cfg.beam_b*1000:.0f}x{cfg.beam_h*1000:.0f}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    info_text = f"结构信息\n"
    info_text += f"----------\n"
    info_text += f"层数: {cfg.n_stories}层\n"
    info_text += f"跨数: {cfg.n_bays}跨\n"
    info_text += f"总高: {cfg.roof_height:.1f}m\n"
    info_text += f"总宽: {cfg.total_width:.1f}m\n"
    info_text += f"混凝土: {cfg.concrete_grade}\n"
    info_text += f"钢筋: {cfg.steel_grade}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#bdc3c7', alpha=0.95))
    
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_internal_forces_on_frame(
    coords: Dict[int, Tuple[float, float]],
    elements: List[Tuple[int, int, int, str]],
    element_forces: Dict[int, Dict],
    cfg: RCFrameConfig,
    path: Path,
    force_type: str = "moment",  # "moment", "shear", "axial"
    title: str = "内力图",
) -> None:
    """
    在框架上直接绘制内力图（弯矩图、剪力图、轴力图）
    
    内力图绘制规则:
    - 弯矩图: 垂直于杆件方向绘制，受拉侧为正
    - 剪力图: 垂直于杆件方向绘制
    - 轴力图: 沿杆件方向绘制
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f8f9fa')
    
    # 1. 绘制原始框架（灰色虚线）
    for _, i_node, j_node, ele_type in elements:
        xi, yi = coords[i_node]
        xj, yj = coords[j_node]
        ax.plot([xi, xj], [yi, yj], color='#bdc3c7', linewidth=2, linestyle='-', zorder=1)
    
    # 2. 绘制节点
    for tag, (x, y) in coords.items():
        ax.scatter([x], [y], s=30, color='#2c3e50', zorder=5)
        if y == 0:
            triangle = Polygon([(x-0.2, y), (x+0.2, y), (x, y-0.25)], 
                              closed=True, facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=1)
            ax.add_patch(triangle)
    
    # 3. 确定绘图比例
    if force_type == "moment":
        key_i, key_j = 'M_i', 'M_j'
        color_pos, color_neg = '#e74c3c', '#3498db'
        unit = 'kN·m'
    elif force_type == "shear":
        key_i, key_j = 'V_i', 'V_j'
        color_pos, color_neg = '#27ae60', '#9b59b6'
        unit = 'kN'
    else:  # axial
        key_i, key_j = 'N_i', 'N_j'
        color_pos, color_neg = '#f39c12', '#1abc9c'
        unit = 'kN'
    
    # 找最大值用于缩放
    max_force = 0.0
    for ele, data in element_forces.items():
        max_force = max(max_force, abs(data.get(key_i, 0)), abs(data.get(key_j, 0)))
    
    if max_force < 1e-6:
        max_force = 1.0
    
    # 内力图绘制比例尺
    scale = 0.8 / max_force  # 最大偏移0.8m
    
    # 4. 绘制内力图
    for ele, data in element_forces.items():
        xi, yi = data['xi'], data['yi']
        xj, yj = data['xj'], data['yj']
        fi = data.get(key_i, 0)
        fj = data.get(key_j, 0)
        ele_type = data['type']
        
        # 杆件长度和方向
        dx = xj - xi
        dy = yj - yi
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            continue
        
        # 单位向量
        ex = dx / length  # 沿杆件方向
        ey = dy / length
        # 垂直方向（逆时针90度）
        nx = -ey
        ny = ex
        
        # 分段绘制内力图
        n_seg = 20
        x_points = []
        y_points = []
        force_values = []
        
        for i in range(n_seg + 1):
            t = i / n_seg
            # 杆件上的位置
            px = xi + t * dx
            py = yi + t * dy
            
            # 线性插值内力（简化处理）
            # 对于梁，弯矩可能是抛物线分布，这里简化为线性
            f = fi * (1 - t) + (-fj) * t  # 注意j端符号
            
            # 内力图偏移
            offset = f * scale
            
            if force_type == "moment":
                # 弯矩图：受拉侧绘制
                # 对于梁：下部受拉为正弯矩，向下绘制
                # 对于柱：外侧受拉为正弯矩
                if ele_type == "beam":
                    offset_x = px + nx * offset
                    offset_y = py + ny * offset
                else:  # column
                    offset_x = px + nx * offset
                    offset_y = py + ny * offset
            else:
                # 剪力和轴力
                offset_x = px + nx * offset
                offset_y = py + ny * offset
            
            x_points.append(offset_x)
            y_points.append(offset_y)
            force_values.append(f)
        
        # 闭合图形
        x_closed = [xi] + x_points + [xj, xi]
        y_closed = [yi] + y_points + [yj, yi]
        
        # 根据正负填充颜色
        avg_force = (fi - fj) / 2
        fill_color = color_pos if avg_force >= 0 else color_neg
        
        # 绘制填充区域
        ax.fill(x_closed, y_closed, color=fill_color, alpha=0.3, zorder=2)
        ax.plot(x_points, y_points, color=fill_color, linewidth=1.5, zorder=3)
        
        # 标注端部数值
        if abs(fi) > max_force * 0.1:  # 只标注较大的值
            label_x = xi + nx * fi * scale * 1.2
            label_y = yi + ny * fi * scale * 1.2
            ax.annotate(f'{abs(fi):.1f}', xy=(label_x, label_y), fontsize=7,
                       ha='center', va='center', color=fill_color, fontweight='bold')
    
    # 5. 添加图例和标题
    legend_elements = [
        Line2D([0], [0], color=color_pos, linewidth=8, alpha=0.5, label='正值'),
        Line2D([0], [0], color=color_neg, linewidth=8, alpha=0.5, label='负值'),
        Line2D([0], [0], color='#bdc3c7', linewidth=2, label='框架'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    force_name = {"moment": "弯矩图", "shear": "剪力图", "axial": "轴力图"}
    ax.set_title(f'{title} - {force_name.get(force_type, "内力图")} (单位: {unit})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, cfg.total_width + 2)
    ax.set_ylim(-1.5, cfg.roof_height + 1.5)
    
    # 添加比例尺说明
    scale_text = f"比例尺: 最大值 {max_force:.1f} {unit}"
    ax.text(0.02, 0.02, scale_text, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_all_internal_forces(
    coords: Dict[int, Tuple[float, float]],
    elements: List[Tuple[int, int, int, str]],
    element_forces: Dict[int, Dict],
    cfg: RCFrameConfig,
    out_dir: Path,
    title_prefix: str = "重力荷载"
) -> None:
    """绘制所有内力图（弯矩、剪力、轴力）"""
    
    # 弯矩图
    plot_internal_forces_on_frame(
        coords, elements, element_forces, cfg,
        out_dir / f"{title_prefix}_moment.png",
        force_type="moment",
        title=f"{title_prefix}作用下"
    )
    
    # 剪力图
    plot_internal_forces_on_frame(
        coords, elements, element_forces, cfg,
        out_dir / f"{title_prefix}_shear.png",
        force_type="shear",
        title=f"{title_prefix}作用下"
    )
    
    # 轴力图
    plot_internal_forces_on_frame(
        coords, elements, element_forces, cfg,
        out_dir / f"{title_prefix}_axial.png",
        force_type="axial",
        title=f"{title_prefix}作用下"
    )


def plot_mode_shapes(cfg: RCFrameConfig, periods: List[float], mode_shapes: List[List[float]], path: Path) -> None:
    """绘制振型图"""
    n_modes = min(3, len(mode_shapes))
    fig, axes = plt.subplots(1, n_modes + 1, figsize=(16, 10))
    
    elevations = story_elevations(cfg)
    colors = ['#3498db', '#e74c3c', '#27ae60']
    
    for i in range(n_modes):
        ax = axes[i]
        ax.set_facecolor('#f8f9fa')
        
        shape = mode_shapes[i]
        
        for elev in elevations:
            ax.axhline(elev, color='#bdc3c7', linewidth=0.5, linestyle='--')
        
        scale = 2.0
        x_orig = [0] * len(elevations)
        x_deform = [s * scale for s in shape]
        
        ax.plot(x_orig, elevations, 'k--', linewidth=1, alpha=0.5, label='原始位置')
        ax.plot(x_deform, elevations, 'o-', color=colors[i], linewidth=3, markersize=10, 
               markerfacecolor='white', markeredgewidth=2.5, label=f'第{i+1}振型')
        ax.fill_betweenx(elevations, x_orig, x_deform, alpha=0.2, color=colors[i])
        
        for j, (x, y) in enumerate(zip(x_deform, elevations)):
            if j > 0:
                ax.annotate(f'{shape[j]:.3f}', xy=(x, y), xytext=(x + 0.3, y),
                           fontsize=9, va='center')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, cfg.roof_height + 1)
        ax.set_xlabel('归一化位移', fontsize=11)
        ax.set_ylabel('高度 (m)', fontsize=11)
        ax.set_title(f'第{i+1}振型\nT{i+1} = {periods[i]:.4f}s\nf{i+1} = {1/periods[i]:.3f}Hz', 
                    fontsize=12, fontweight='bold')
        ax.axvline(0, color='#2c3e50', linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    ax = axes[n_modes]
    ax.set_facecolor('#f8f9fa')
    
    for elev in elevations:
        ax.axhline(elev, color='#bdc3c7', linewidth=0.5, linestyle='--')
    
    for i in range(n_modes):
        shape = mode_shapes[i]
        x_deform = [s * 1.5 for s in shape]
        ax.plot(x_deform, elevations, 'o-', color=colors[i], linewidth=2.5, markersize=8,
               markerfacecolor='white', markeredgewidth=2, label=f'T{i+1}={periods[i]:.3f}s')
    
    ax.axvline(0, color='#2c3e50', linewidth=1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, cfg.roof_height + 1)
    ax.set_xlabel('归一化位移', fontsize=11)
    ax.set_ylabel('高度 (m)', fontsize=11)
    ax.set_title('振型对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('结构振型分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_modal_analysis(periods: List[float], cfg: RCFrameConfig, path: Path) -> None:
    """绘制模态分析结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_modes = len(periods)
    modes = np.arange(1, n_modes + 1)
    freqs = [1/p for p in periods]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_modes))
    
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    bars1 = ax1.bar(modes, periods, color=colors, edgecolor='white', linewidth=2)
    for bar, p in zip(bars1, periods):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{p:.4f}s', ha='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('振型阶数', fontsize=12)
    ax1.set_ylabel('周期 T (s)', fontsize=12)
    ax1.set_title('自振周期', fontsize=14, fontweight='bold')
    ax1.set_xticks(modes)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, n_modes))
    bars2 = ax2.bar(modes, freqs, color=colors2, edgecolor='white', linewidth=2)
    for bar, f in zip(bars2, freqs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{f:.3f}Hz', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xlabel('振型阶数', fontsize=12)
    ax2.set_ylabel('频率 f (Hz)', fontsize=12)
    ax2.set_title('自振频率', fontsize=14, fontweight='bold')
    ax2.set_xticks(modes)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('模态分析结果', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_deformed_shape(
    coords: Dict[int, Tuple[float, float]],
    elements: List[Tuple[int, int, int, str]],
    displacements: Dict[int, Tuple[float, float]],
    path: Path,
    title: str,
) -> None:
    """绘制变形图"""
    width = max(x for x, _ in coords.values()) - min(x for x, _ in coords.values())
    height = max(y for _, y in coords.values()) - min(y for _, y in coords.values())
    max_disp = max((math.hypot(*u) for u in displacements.values()), default=0.0)
    scale = 1.0 if max_disp <= 1.0e-12 else 0.08 * max(width, height) / max_disp

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#f8f9fa')
    
    for _, i_node, j_node, _ in elements:
        xi, yi = coords[i_node]
        xj, yj = coords[j_node]
        ax.plot([xi, xj], [yi, yj], color="#bdc3c7", linewidth=2, linestyle="--", alpha=0.7)
    
    for _, i_node, j_node, _ in elements:
        xi, yi = coords[i_node]
        xj, yj = coords[j_node]
        uxi, uyi = displacements[i_node]
        uxj, uyj = displacements[j_node]
        ax.plot(
            [xi + scale * uxi, xj + scale * uxj],
            [yi + scale * uyi, yj + scale * uyj],
            color="#e74c3c", linewidth=3,
        )
    
    max_disp_node = max(displacements.keys(), key=lambda k: abs(displacements[k][0]))
    max_u = displacements[max_disp_node]
    x0, y0 = coords[max_disp_node]
    
    ax.scatter([x0 + scale * max_u[0]], [y0 + scale * max_u[1]], s=150, 
              marker='*', color='#e74c3c', edgecolor='white', linewidth=1.5, zorder=10)
    ax.annotate(f'最大位移: {max_disp*1000:.3f} mm', 
                xy=(x0 + scale * max_u[0], y0 + scale * max_u[1]),
                xytext=(x0 + scale * max_u[0] + 2, y0 + scale * max_u[1] + 1),
                fontsize=11, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9))
    
    ax.set_title(f"{title}\n(放大系数 = {scale:.1f})", fontsize=16, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        Line2D([0], [0], color='#bdc3c7', linewidth=2, linestyle='--', label='原始结构'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='变形后结构'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_story_drifts_line(drifts: List[float], cfg: RCFrameConfig, path: Path, title: str, limit: float | None = None) -> None:
    """绘制层间位移角"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    stories = np.arange(1, len(drifts) + 1)
    
    abs_drifts = [abs(d) for d in drifts]
    max_drift = max(abs_drifts) if abs_drifts else 1e-6
    max_story = stories[abs_drifts.index(max_drift)]
    
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    
    ax1.plot(abs_drifts, stories, 'o-', color='#3498db', linewidth=3, markersize=12, 
            markerfacecolor='white', markeredgewidth=2.5, label='层间位移角')
    ax1.fill_betweenx(stories, 0, abs_drifts, alpha=0.2, color='#3498db')
    
    for i, (drift, story) in enumerate(zip(abs_drifts, stories)):
        ratio = int(1/drift) if drift > 1e-10 else 0
        ax1.annotate(f'{drift:.5f}\n(1/{ratio})', xy=(drift, story), 
                   xytext=(drift + max_drift * 0.12, story),
                   fontsize=10, va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bdc3c7', alpha=0.8))
    
    ax1.scatter([max_drift], [max_story], s=200, color='#e74c3c', zorder=10, marker='*',
               edgecolor='white', linewidth=1.5)
    ax1.annotate(f'最大值: {max_drift:.5f}\n(1/{int(1/max_drift)})\n第{max_story}层', 
                xy=(max_drift, max_story),
                xytext=(max_drift + max_drift * 0.2, max_story + 0.4),
                fontsize=11, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5f5', edgecolor='#e74c3c', alpha=0.95))
    
    if limit is not None:
        ax1.axvline(limit, color="#e74c3c", linestyle="--", linewidth=2.5, 
                   label=f"规范限值 1/{int(1/limit)}")
        ax1.fill_betweenx([0.5, len(drifts) + 0.5], limit, max(max_drift * 1.5, limit * 1.3), 
                         alpha=0.1, color='#e74c3c')
    
    x_max = max(max_drift * 2.0, limit * 1.5 if limit else max_drift * 2.0)
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0.5, len(drifts) + 0.5)
    ax1.set_yticks(stories)
    ax1.set_yticklabels([f"第{i}层" for i in stories])
    ax1.set_xlabel("层间位移角", fontsize=12)
    ax1.set_ylabel("楼层", fontsize=12)
    ax1.set_title("层间位移角分布", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10)
    
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    
    inter_story_disps = [abs(d) * cfg.story_heights[i] * 1000 for i, d in enumerate(drifts)]
    max_disp = max(inter_story_disps) if inter_story_disps else 1
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stories)))
    bars = ax2.barh(stories, inter_story_disps, color=colors, edgecolor='white', linewidth=2, height=0.6)
    
    for bar, disp in zip(bars, inter_story_disps):
        ax2.text(disp + max_disp * 0.03, bar.get_y() + bar.get_height()/2,
                f'{disp:.2f} mm', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(0, max_disp * 1.4)
    ax2.set_ylim(0.5, len(drifts) + 0.5)
    ax2.set_yticks(stories)
    ax2.set_yticklabels([f"第{i}层\n(h={cfg.story_heights[i-1]}m)" for i in stories])
    ax2.set_xlabel("层间位移 (mm)", fontsize=12)
    ax2.set_ylabel("楼层", fontsize=12)
    ax2.set_title("层间位移分布", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_ground_motion(time: np.ndarray, accel: np.ndarray, path: Path, cfg: RCFrameConfig) -> None:
    """绘制地震波"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    ax1.set_facecolor('#f8f9fa')
    ax1.plot(time, accel, color='#2c3e50', linewidth=0.8)
    ax1.fill_between(time, accel, alpha=0.3, color='#3498db')
    ax1.axhline(0, color='gray', linewidth=0.5)
    
    pga_max_idx = np.argmax(accel)
    pga_min_idx = np.argmin(accel)
    
    ax1.scatter([time[pga_max_idx]], [accel[pga_max_idx]], s=100, color='#e74c3c', zorder=5, marker='^')
    ax1.annotate(f'PGA+ = {accel[pga_max_idx]:.4f} m/s²\nt = {time[pga_max_idx]:.2f}s', 
                xy=(time[pga_max_idx], accel[pga_max_idx]),
                xytext=(time[pga_max_idx]+2, accel[pga_max_idx]*0.8),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.scatter([time[pga_min_idx]], [accel[pga_min_idx]], s=100, color='#2980b9', zorder=5, marker='v')
    ax1.annotate(f'PGA- = {accel[pga_min_idx]:.4f} m/s²\nt = {time[pga_min_idx]:.2f}s', 
                xy=(time[pga_min_idx], accel[pga_min_idx]),
                xytext=(time[pga_min_idx]+2, accel[pga_min_idx]*0.8),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2980b9'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel("时间 (s)", fontsize=11)
    ax1.set_ylabel("加速度 (m/s²)", fontsize=11)
    ax1.set_title(f"加速度时程 (PGA = {cfg.th_pga_g}g)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, time[-1])
    
    ax2 = axes[0, 1]
    ax2.set_facecolor('#f8f9fa')
    dt = time[1] - time[0]
    velocity = np.cumsum(accel) * dt
    ax2.plot(time, velocity * 100, color='#27ae60', linewidth=1)
    ax2.fill_between(time, velocity * 100, alpha=0.3, color='#27ae60')
    ax2.axhline(0, color='gray', linewidth=0.5)
    
    pgv_idx = np.argmax(np.abs(velocity))
    ax2.scatter([time[pgv_idx]], [velocity[pgv_idx]*100], s=100, color='#27ae60', zorder=5)
    ax2.annotate(f'PGV = {velocity[pgv_idx]*100:.2f} cm/s', 
                xy=(time[pgv_idx], velocity[pgv_idx]*100),
                xytext=(time[pgv_idx]+2, velocity[pgv_idx]*100*0.8),
                fontsize=10, fontweight='bold')
    
    ax2.set_xlabel("时间 (s)", fontsize=11)
    ax2.set_ylabel("速度 (cm/s)", fontsize=11)
    ax2.set_title("速度时程", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, time[-1])
    
    ax3 = axes[1, 0]
    ax3.set_facecolor('#f8f9fa')
    disp = np.cumsum(velocity) * dt
    ax3.plot(time, disp * 100, color='#9b59b6', linewidth=1)
    ax3.fill_between(time, disp * 100, alpha=0.3, color='#9b59b6')
    ax3.axhline(0, color='gray', linewidth=0.5)
    
    pgd_idx = np.argmax(np.abs(disp))
    ax3.scatter([time[pgd_idx]], [disp[pgd_idx]*100], s=100, color='#9b59b6', zorder=5)
    ax3.annotate(f'PGD = {disp[pgd_idx]*100:.2f} cm', 
                xy=(time[pgd_idx], disp[pgd_idx]*100),
                xytext=(time[pgd_idx]+2, disp[pgd_idx]*100*0.8),
                fontsize=10, fontweight='bold')
    
    ax3.set_xlabel("时间 (s)", fontsize=11)
    ax3.set_ylabel("位移 (cm)", fontsize=11)
    ax3.set_title("位移时程", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, time[-1])
    
    ax4 = axes[1, 1]
    ax4.set_facecolor('#f8f9fa')
    
    n = len(accel)
    freq = np.fft.rfftfreq(n, dt)
    fft_amp = np.abs(np.fft.rfft(accel)) * 2 / n
    
    ax4.plot(freq, fft_amp, color='#e74c3c', linewidth=1)
    ax4.fill_between(freq, fft_amp, alpha=0.3, color='#e74c3c')
    
    valid_idx = freq > 0.1
    if np.any(valid_idx):
        dominant_idx = np.argmax(fft_amp[valid_idx]) + np.argmax(valid_idx)
        dominant_freq = freq[dominant_idx]
        ax4.scatter([dominant_freq], [fft_amp[dominant_idx]], s=100, color='#e74c3c', zorder=5)
        ax4.annotate(f'主频 f = {dominant_freq:.2f} Hz\nT = {1/dominant_freq:.2f} s', 
                    xy=(dominant_freq, fft_amp[dominant_idx]),
                    xytext=(dominant_freq + 2, fft_amp[dominant_idx]*0.9),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    ax4.set_xlabel("频率 (Hz)", fontsize=11)
    ax4.set_ylabel("傅里叶幅值 (m/s²)", fontsize=11)
    ax4.set_title("傅里叶频谱", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0, 15)
    
    plt.suptitle('地震动特性分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_time_history(rows: List[Dict[str, float]], cfg: RCFrameConfig, path: Path) -> None:
    """绘制时程响应"""
    time = np.array([r["time_s"] for r in rows])
    roof = np.array([r["roof_disp_m"] * 1000 for r in rows])
    accel = np.array([r["ground_accel_m_s2"] for r in rows])
    shear = np.array([r["base_shear_kN"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1.plot(time, accel, color="#2c3e50", linewidth=0.8)
    ax1.fill_between(time, accel, alpha=0.3, color='#3498db')
    ax1.set_ylabel("加速度 (m/s²)", fontsize=11)
    ax1.set_title("地震加速度输入", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(0, color='gray', linewidth=0.5)

    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.plot(time, roof, color="#e74c3c", linewidth=1.0)
    ax2.fill_between(time, roof, alpha=0.2, color='#e74c3c')
    ax2.set_ylabel("位移 (mm)", fontsize=11)
    ax2.set_title("屋顶水平位移响应", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(0, color='gray', linewidth=0.5)
    
    if len(roof) > 0:
        max_idx = np.argmax(roof)
        min_idx = np.argmin(roof)
        ax2.scatter([time[max_idx]], [roof[max_idx]], s=100, color='#c0392b', zorder=5, marker='^')
        ax2.annotate(f'+{roof[max_idx]:.2f}mm @ t={time[max_idx]:.2f}s', 
                    xy=(time[max_idx], roof[max_idx]),
                    xytext=(time[max_idx]+1, roof[max_idx]*0.9),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax2.scatter([time[min_idx]], [roof[min_idx]], s=100, color='#2980b9', zorder=5, marker='v')
        ax2.annotate(f'{roof[min_idx]:.2f}mm @ t={time[min_idx]:.2f}s', 
                    xy=(time[min_idx], roof[min_idx]),
                    xytext=(time[min_idx]+1, roof[min_idx]*0.9),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax3 = axes[2]
    ax3.set_facecolor('#f8f9fa')
    ax3.plot(time, shear, color="#27ae60", linewidth=1.0)
    ax3.fill_between(time, shear, alpha=0.2, color='#27ae60')
    ax3.set_xlabel("时间 (s)", fontsize=11)
    ax3.set_ylabel("基底剪力 (kN)", fontsize=11)
    ax3.set_title("基底剪力响应", fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhline(0, color='gray', linewidth=0.5)
    
    if len(shear) > 0:
        max_shear_idx = np.argmax(np.abs(shear))
        ax3.scatter([time[max_shear_idx]], [shear[max_shear_idx]], s=100, color='#27ae60', zorder=5)
        ax3.annotate(f'Vmax = {shear[max_shear_idx]:.1f}kN', 
                    xy=(time[max_shear_idx], shear[max_shear_idx]),
                    xytext=(time[max_shear_idx]+1, shear[max_shear_idx]*0.9),
                    fontsize=10, fontweight='bold')
    
    plt.suptitle('时程分析响应', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_floor_responses(rows: List[Dict[str, float]], cfg: RCFrameConfig, out_dir: Path) -> None:
    """绘制各层响应"""
    if not rows:
        return
    
    time = np.array([r["time_s"] for r in rows])
    n_stories = cfg.n_stories
    
    disps = {i: np.array([r.get(f"story_{i}_disp_m", 0) * 1000 for r in rows]) for i in range(1, n_stories + 1)}
    vels = {i: np.array([r.get(f"story_{i}_vel_m_s", 0) * 1000 for r in rows]) for i in range(1, n_stories + 1)}
    accels = {i: np.array([r.get(f"story_{i}_accel_m_s2", 0) for r in rows]) for i in range(1, n_stories + 1)}
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_stories))
    
    # 位移响应
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor('#f8f9fa')
    for i in range(1, n_stories + 1):
        ax.plot(time, disps[i], linewidth=1.2, color=colors[i-1], label=f'第{i}层', alpha=0.9)
    
    max_idx = np.argmax(np.abs(disps[n_stories]))
    max_val = disps[n_stories][max_idx]
    ax.scatter([time[max_idx]], [max_val], s=100, color=colors[n_stories-1], zorder=5)
    ax.annotate(f'顶层最大: {max_val:.2f}mm', xy=(time[max_idx], max_val),
               xytext=(time[max_idx]+1, max_val*1.1), fontsize=10, fontweight='bold')
    
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("位移 (mm)", fontsize=12)
    ax.set_title("各层位移时程响应", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    ax.axhline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "08_floor_displacement_response.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # 峰值包络图
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    stories = np.arange(1, n_stories + 1)
    max_disps = [np.max(np.abs(disps[i])) for i in range(1, n_stories + 1)]
    max_vels = [np.max(np.abs(vels[i])) for i in range(1, n_stories + 1)]
    max_accels = [np.max(np.abs(accels[i])) for i in range(1, n_stories + 1)]
    
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1.plot(max_disps, stories, 'o-', color='#3498db', linewidth=3, markersize=12,
            markerfacecolor='white', markeredgewidth=2.5)
    ax1.fill_betweenx(stories, 0, max_disps, alpha=0.2, color='#3498db')
    for d, s in zip(max_disps, stories):
        ax1.text(d + max(max_disps)*0.05, s, f'{d:.2f}', fontsize=10, va='center', fontweight='bold')
    ax1.set_xlabel("最大位移 (mm)", fontsize=12)
    ax1.set_ylabel("楼层", fontsize=12)
    ax1.set_title("位移峰值包络", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yticks(stories)
    
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.plot(max_vels, stories, 's-', color='#27ae60', linewidth=3, markersize=12,
            markerfacecolor='white', markeredgewidth=2.5)
    ax2.fill_betweenx(stories, 0, max_vels, alpha=0.2, color='#27ae60')
    for v, s in zip(max_vels, stories):
        ax2.text(v + max(max_vels)*0.05, s, f'{v:.1f}', fontsize=10, va='center', fontweight='bold')
    ax2.set_xlabel("最大速度 (mm/s)", fontsize=12)
    ax2.set_ylabel("楼层", fontsize=12)
    ax2.set_title("速度峰值包络", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yticks(stories)
    
    ax3 = axes[2]
    ax3.set_facecolor('#f8f9fa')
    ax3.plot(max_accels, stories, '^-', color='#e74c3c', linewidth=3, markersize=12,
            markerfacecolor='white', markeredgewidth=2.5)
    ax3.fill_betweenx(stories, 0, max_accels, alpha=0.2, color='#e74c3c')
    for a, s in zip(max_accels, stories):
        ax3.text(a + max(max_accels)*0.05, s, f'{a:.3f}', fontsize=10, va='center', fontweight='bold')
    ax3.set_xlabel("最大加速度 (m/s²)", fontsize=12)
    ax3.set_ylabel("楼层", fontsize=12)
    ax3.set_title("加速度峰值包络", fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yticks(stories)
    
    plt.suptitle("各层响应峰值包络图", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "09_floor_response_envelope.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_pushover_curve_with_energy(rows: List[Dict[str, float]], cfg: RCFrameConfig, path: Path) -> Dict[str, float]:
    """
    绘制推覆曲线，标注两部分面积相等
    
    能量等效法原理：
    - 实际pushover曲线下的面积 = 双线性模型下的面积
    - 通过调整屈服点位置使两个面积相等
    - 图中用不同颜色填充两个区域，标注面积数值
    """
    if not rows:
        return {}
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    disp = np.array([r["roof_disp_m"] * 1000 for r in rows])
    shear = np.array([r["base_shear_kN"] for r in rows])
    
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    
    # 平滑曲线
    if len(disp) > 20:
        try:
            spline = UnivariateSpline(disp, shear, s=len(disp)*0.5)
            disp_smooth = np.linspace(disp.min(), disp.max(), 300)
            shear_smooth = spline(disp_smooth)
            shear_smooth = np.maximum(shear_smooth, 0)
        except:
            disp_smooth = disp
            shear_smooth = shear
    else:
        disp_smooth = disp
        shear_smooth = shear
    
    # 计算等效屈服点
    dy, fy, k0, k1, area_actual = find_equivalent_yield_point(disp, shear)
    du = disp.max()
    fu = fy + k1 * (du - dy)
    
    # 计算双线性面积
    area_bilinear = 0.5 * fy * dy + 0.5 * (fy + fu) * (du - dy)
    
    # ========== 绘制面积区域 ==========
    # 实际曲线下面积（蓝色填充）
    ax1.fill_between(disp_smooth, 0, shear_smooth, alpha=0.25, color='#3498db', 
                    label=f'实际曲线面积 A1 = {area_actual:.1f} kN·mm')
    
    # 双线性模型面积（红色边界）
    bilinear_x = [0, dy, du]
    bilinear_y = [0, fy, fu]
    ax1.fill(bilinear_x + [du, 0], bilinear_y + [0, 0], alpha=0.15, color='#e74c3c',
            edgecolor='#e74c3c', linewidth=2, linestyle='--',
            label=f'双线性面积 A2 = {area_bilinear:.1f} kN·mm')
    
    # 绘制实际曲线
    ax1.scatter(disp, shear, color='#bdc3c7', s=15, alpha=0.5, zorder=3)
    ax1.plot(disp_smooth, shear_smooth, color="#2c3e50", linewidth=3, label='Pushover曲线', zorder=4)
    
    # 绘制双线性模型
    ax1.plot([0, dy], [0, fy], 'r-', linewidth=2.5, zorder=5)
    ax1.plot([dy, du], [fy, fu], 'r-', linewidth=2.5, zorder=5)
    
    # 标注屈服点
    ax1.scatter([dy], [fy], s=250, color='#e74c3c', zorder=10, marker='o', 
               edgecolor='white', linewidth=2.5)
    ax1.annotate(f'等效屈服点\ndy = {dy:.1f} mm\nVy = {fy:.1f} kN', 
                xy=(dy, fy),
                xytext=(dy + du*0.12, fy * 0.6),
                fontsize=11, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5f5', edgecolor='#e74c3c', alpha=0.95))
    
    # 标注初始刚度
    x_k0 = dy * 0.5
    y_k0 = k0 * x_k0
    ax1.annotate(f'初始刚度 K0 = {k0:.2f} kN/mm\n(割线刚度至75%峰值)', 
                xy=(x_k0, y_k0),
                xytext=(x_k0 + du*0.15, y_k0 * 1.4),
                fontsize=10, color='#2980b9',
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f7ff', edgecolor='#2980b9', alpha=0.9))
    
    # 标注峰值
    max_shear = shear.max()
    max_idx = np.argmax(shear)
    ax1.scatter([disp[max_idx]], [max_shear], s=150, color='#27ae60', zorder=10, marker='*')
    ax1.annotate(f'Vmax = {max_shear:.1f} kN', 
                xy=(disp[max_idx], max_shear),
                xytext=(disp[max_idx]*1.05 + 10, max_shear*0.95),
                fontsize=10, fontweight='bold', color='#27ae60')
    
    # 面积相等说明
    area_diff_pct = abs(area_actual - area_bilinear) / area_actual * 100 if area_actual > 0 else 0
    info_text = "能量等效法\n"
    info_text += "-" * 16 + "\n"
    info_text += f"A1 (实际) = {area_actual:.1f} kN·mm\n"
    info_text += f"A2 (双线性) = {area_bilinear:.1f} kN·mm\n"
    info_text += f"面积差异 = {area_diff_pct:.2f}%\n"
    info_text += "-" * 16 + "\n"
    info_text += "原理: 调整 dy 使 A1 = A2\n"
    info_text += "等效能量耗散能力相同"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffef0', edgecolor='#f39c12', alpha=0.95))
    
    # 延性系数
    ductility = du / dy if dy > 0 else 0
    duct_text = f"延性系数 μ = {ductility:.2f}\n最大位移 = {du:.1f} mm"
    ax1.text(0.98, 0.02, duct_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel("屋顶位移 (mm)", fontsize=12)
    ax1.set_ylabel("基底剪力 (kN)", fontsize=12)
    ax1.set_title("推覆曲线与能量等效法", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # ========== 第二个子图：刚度退化 ==========
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    
    stiffness = np.zeros_like(disp)
    for i in range(len(disp)):
        if disp[i] > 0:
            stiffness[i] = shear[i] / disp[i]
    
    k0_actual = stiffness[stiffness > 0][0] if np.any(stiffness > 0) else 1
    stiffness_ratio = stiffness / k0_actual
    
    drift = disp / (cfg.roof_height * 1000) * 100
    
    ax2.plot(drift, stiffness_ratio, 'o-', color='#9b59b6', linewidth=2, markersize=3, alpha=0.8)
    ax2.fill_between(drift, stiffness_ratio, alpha=0.2, color='#9b59b6')
    ax2.axhline(1.0, color='#7f8c8d', linestyle='--', linewidth=1.5, label='初始刚度 K0')
    ax2.axhline(0.5, color='#e74c3c', linestyle=':', linewidth=1.5, label='50% K0')
    ax2.axhline(0.2, color='#c0392b', linestyle=':', linewidth=1.5, label='20% K0')
    
    # 标注刚度降到50%的位置
    idx_50 = np.argmin(np.abs(stiffness_ratio - 0.5))
    if stiffness_ratio[idx_50] < 0.6:
        ax2.scatter([drift[idx_50]], [stiffness_ratio[idx_50]], s=100, color='#e74c3c', zorder=5)
        ax2.annotate(f'K = 50%K0\nθ = {drift[idx_50]:.2f}%', 
                    xy=(drift[idx_50], stiffness_ratio[idx_50]),
                    xytext=(drift[idx_50]+0.5, 0.6),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    ax2.set_xlabel("屋顶位移角 (%)", fontsize=12)
    ax2.set_ylabel("刚度比 K/K0", fontsize=12)
    ax2.set_title("刚度退化曲线", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.2)
    
    # 刚度说明
    stiff_text = "关于初始刚度K0的说明:\n"
    stiff_text += "-" * 20 + "\n"
    stiff_text += "采用割线刚度而非切线刚度:\n"
    stiff_text += "1. 切线刚度是曲线某点的导数\n"
    stiff_text += "   对局部噪声敏感\n"
    stiff_text += "2. 割线刚度从原点到目标点\n"
    stiff_text += "   代表等效弹性行为\n"
    stiff_text += "3. 取75%峰值前的割线刚度\n"
    stiff_text += "   避免将塑性段纳入\n"
    stiff_text += "4. 符合ATC-40/FEMA规范"
    
    ax2.text(0.98, 0.98, stiff_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0fff0', edgecolor='#27ae60', alpha=0.95))
    
    plt.suptitle('推覆分析结果 - 能量等效法', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return {
        "yield_disp_mm": dy,
        "yield_force_kN": fy,
        "initial_stiffness_kN_mm": k0,
        "post_yield_stiffness_kN_mm": k1,
        "ultimate_disp_mm": du,
        "ductility_ratio": ductility,
        "max_base_shear_kN": max_shear,
        "area_actual_kN_mm": area_actual,
        "area_bilinear_kN_mm": area_bilinear,
    }


def plot_concrete_constitutive(cfg: RCFrameConfig, path: Path) -> None:
    """绘制混凝土本构曲线"""
    info = concrete_constitutive_info(cfg)
    cover = info["cover_unconfined_concrete"]
    core = info["core_confined_concrete"]
    strain = np.linspace(-0.025, 0.001, 800)
    
    def concrete01_envelope(strain_arr, fpc, epsc0, fpcu, epsu):
        stress = np.zeros_like(strain_arr)
        for i, eps in enumerate(strain_arr):
            if eps >= 0.0:
                stress[i] = 0.0
            elif eps >= epsc0:
                ratio = eps / epsc0
                stress[i] = fpc * (2.0 * ratio - ratio**2)
            elif eps >= epsu:
                stress[i] = fpc + (fpcu - fpc) * (eps - epsc0) / (epsu - epsc0)
            else:
                stress[i] = fpcu
        return stress
    
    cover_stress = concrete01_envelope(strain, float(cover["fpc_MPa"]), float(cover["epsc0"]),
                                        float(cover["fpcu_MPa"]), float(cover["epsU"]))
    core_stress = concrete01_envelope(strain, float(core["fpc_MPa"]), float(core["epsc0"]),
                                       float(core["fpcu_MPa"]), float(core["epsU"]))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#f8f9fa')
    
    ax.plot(strain * 1000, cover_stress, color="#7f8c8d", linewidth=3, label="非约束混凝土")
    ax.fill_between(strain * 1000, cover_stress, alpha=0.15, color="#7f8c8d")
    ax.plot(strain * 1000, core_stress, color="#3498db", linewidth=3, label="约束混凝土")
    ax.fill_between(strain * 1000, core_stress, alpha=0.15, color="#3498db")
    ax.axhline(0.0, color="#2c3e50", linewidth=1)
    ax.axvline(0.0, color="#2c3e50", linewidth=1)
    
    fc_cover = float(cover["fpc_MPa"])
    eps_c = float(cover["epsc0"]) * 1000
    ax.scatter([eps_c], [fc_cover], s=100, color='#7f8c8d', zorder=5)
    ax.annotate(f'fc={fc_cover:.1f}MPa', xy=(eps_c, fc_cover),
               xytext=(eps_c-4, fc_cover*0.7), fontsize=10, fontweight='bold')
    
    fcc = float(core["fpc_MPa"])
    eps_cc = float(core["epsc0"]) * 1000
    ax.scatter([eps_cc], [fcc], s=100, color='#3498db', zorder=5)
    ax.annotate(f'fcc={fcc:.1f}MPa', xy=(eps_cc, fcc),
               xytext=(eps_cc+2, fcc*0.9), fontsize=10, fontweight='bold')
    
    info_text = f"混凝土: {cfg.concrete_grade}\n"
    info_text += f"----------\n"
    info_text += f"fck = {cfg.fc/K_OVER_M2_PER_MPA:.1f} MPa\n"
    info_text += f"Ec = {cfg.ec/K_OVER_M2_PER_MPA/1000:.1f} GPa\n"
    info_text += f"约束强度提高: {(fcc/fc_cover-1)*100:.1f}%"
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bdc3c7', alpha=0.95))
    
    ax.set_title(f"混凝土本构曲线 (Concrete01) - {cfg.concrete_grade}", fontsize=14, fontweight='bold')
    ax.set_xlabel("应变 (‰)", fontsize=12)
    ax.set_ylabel("应力 (MPa)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_steel_constitutive(cfg: RCFrameConfig, path: Path) -> None:
    """绘制钢筋本构曲线"""
    info = steel_constitutive_info(cfg)
    fy = float(info["fy_MPa"])
    E0 = float(info["E0_MPa"])
    b = float(info["b"])
    
    strain = np.linspace(-0.05, 0.05, 500)
    ey = fy / E0
    
    stress = np.zeros_like(strain)
    for i, eps in enumerate(strain):
        if abs(eps) <= ey:
            stress[i] = E0 * eps
        else:
            sign = 1 if eps > 0 else -1
            stress[i] = sign * fy + b * E0 * (eps - sign * ey)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#f8f9fa')
    
    ax.plot(strain * 1000, stress, color="#e74c3c", linewidth=3, label=f"{cfg.steel_grade} (三级钢)")
    ax.fill_between(strain * 1000, stress, alpha=0.1, color='#e74c3c')
    ax.axhline(0.0, color="#2c3e50", linewidth=1)
    ax.axvline(0.0, color="#2c3e50", linewidth=1)
    
    ax.scatter([ey * 1000], [fy], s=120, color='#e74c3c', zorder=5)
    ax.scatter([-ey * 1000], [-fy], s=120, color='#e74c3c', zorder=5)
    ax.annotate(f'fy={fy:.0f}MPa\nεy={ey*1000:.2f}‰', xy=(ey*1000, fy),
               xytext=(ey*1000+10, fy*0.8), fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    info_text = f"钢筋: {cfg.steel_grade} (三级钢)\n"
    info_text += f"----------\n"
    info_text += f"fy = {fy:.0f} MPa\n"
    info_text += f"Es = {E0/1000:.0f} GPa\n"
    info_text += f"硬化比 b = {b:.3f}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bdc3c7', alpha=0.95))
    
    ax.set_title(f"钢筋本构曲线 (Steel02) - {cfg.steel_grade}", fontsize=14, fontweight='bold')
    ax.set_xlabel("应变 (‰)", fontsize=12)
    ax.set_ylabel("应力 (MPa)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-55, 55)
    
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_grade3_rebar_symbol(ax, x, y, radius, color='#e74c3c'):
    """绘制三级钢符号"""
    circle = Circle((x, y), radius, facecolor=color, edgecolor='#c0392b', linewidth=2)
    ax.add_patch(circle)
    line_len = radius * 0.6
    line_offset = radius * 0.25
    ax.plot([x - line_len, x + line_len], [y + line_offset, y + line_offset], 
            color='white', linewidth=2, solid_capstyle='round')
    ax.plot([x - line_len, x + line_len], [y - line_offset, y - line_offset], 
            color='white', linewidth=2, solid_capstyle='round')


def plot_section_reinforcement(cfg: RCFrameConfig, path: Path) -> None:
    """绘制配筋图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    
    def draw_section(ax, b, h, cover, bar_dia, bars_top, bars_bottom, side_bars, 
                     stirrup_dia, title, stirrup_spacing=None):
        ax.set_facecolor('#f8f9fa')
        
        scale = 1000
        b_mm, h_mm = b * scale, h * scale
        cover_mm = cover * scale
        bar_r = bar_dia * scale / 2
        stirrup_r = stirrup_dia * scale / 2
        
        outer = Rectangle((-b_mm/2, -h_mm/2), b_mm, h_mm, 
                         fill=True, facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=3)
        ax.add_patch(outer)
        
        core_b = b_mm - 2*cover_mm
        core_h = h_mm - 2*cover_mm
        core = Rectangle((-core_b/2, -core_h/2), core_b, core_h,
                        fill=True, facecolor='#d5e8d4', edgecolor='#82b366', 
                        linewidth=2, linestyle='--')
        ax.add_patch(core)
        
        stirrup_offset = cover_mm - stirrup_r
        stirrup_rect = Rectangle((-b_mm/2 + stirrup_offset, -h_mm/2 + stirrup_offset),
                                  b_mm - 2*stirrup_offset, h_mm - 2*stirrup_offset,
                                  fill=False, edgecolor='#27ae60', linewidth=2.5)
        ax.add_patch(stirrup_rect)
        
        y_top = h_mm/2 - cover_mm
        if bars_top > 1:
            x_positions = np.linspace(-b_mm/2 + cover_mm + bar_r*1.5, b_mm/2 - cover_mm - bar_r*1.5, bars_top)
        else:
            x_positions = [0]
        for x in x_positions:
            draw_grade3_rebar_symbol(ax, x, y_top, bar_r)
        
        y_bot = -h_mm/2 + cover_mm
        if bars_bottom > 1:
            x_positions = np.linspace(-b_mm/2 + cover_mm + bar_r*1.5, b_mm/2 - cover_mm - bar_r*1.5, bars_bottom)
        else:
            x_positions = [0]
        for x in x_positions:
            draw_grade3_rebar_symbol(ax, x, y_bot, bar_r)
        
        if side_bars > 0:
            y_side = np.linspace(-h_mm/2 + cover_mm + bar_r*3, h_mm/2 - cover_mm - bar_r*3, side_bars + 2)[1:-1]
            for y in y_side:
                draw_grade3_rebar_symbol(ax, -b_mm/2 + cover_mm, y, bar_r * 0.8)
                draw_grade3_rebar_symbol(ax, b_mm/2 - cover_mm, y, bar_r * 0.8)
        
        ax.annotate('', xy=(-b_mm/2, -h_mm/2 - 35), xytext=(b_mm/2, -h_mm/2 - 35),
                   arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=2))
        ax.text(0, -h_mm/2 - 50, f'{b_mm:.0f}', fontsize=12, ha='center', fontweight='bold')
        
        ax.annotate('', xy=(b_mm/2 + 35, -h_mm/2), xytext=(b_mm/2 + 35, h_mm/2),
                   arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=2))
        ax.text(b_mm/2 + 50, 0, f'{h_mm:.0f}', fontsize=12, va='center', fontweight='bold', rotation=90)
        
        ax.set_xlim(-b_mm/2 - 90, b_mm/2 + 90)
        ax.set_ylim(-h_mm/2 - 90, h_mm/2 + 90)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        total_bars = bars_top + bars_bottom + (side_bars * 2 if side_bars > 0 else 0)
        bar_area_single = math.pi * (bar_dia * 1000 / 2) ** 2
        total_area = total_bars * bar_area_single
        rho = total_area / (b_mm * h_mm) * 100
        
        info_lines = [
            f"截面: {b_mm:.0f}x{h_mm:.0f} mm",
            f"保护层: c = {cover_mm:.0f} mm",
            f"纵筋: Φ{bar_dia*1000:.0f} (三级钢)",
            f"  上部: {bars_top}根",
            f"  下部: {bars_bottom}根",
        ]
        if side_bars > 0:
            info_lines.append(f"  腰筋: {side_bars*2}根")
        info_lines.extend([
            f"  合计: {total_bars}根",
            f"  As = {total_area:.0f} mm²",
            f"  ρ = {rho:.2f}%",
            f"箍筋: Φ{stirrup_dia*1000:.0f}@{stirrup_spacing*1000:.0f}",
        ])
        
        info_text = "\n".join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bdc3c7', alpha=0.95))
    
    draw_section(axes[0], cfg.beam_b, cfg.beam_h, cfg.cover, cfg.beam_bar_dia,
                cfg.beam_bars_top, cfg.beam_bars_bottom, 0, cfg.beam_stirrup_dia,
                f"梁截面配筋图 ({cfg.beam_b*1000:.0f}x{cfg.beam_h*1000:.0f})",
                cfg.beam_stirrup_spacing)
    
    col_side_bars = (cfg.col_bars_total - 8) // 4 if cfg.col_bars_total > 8 else 0
    draw_section(axes[1], cfg.col_b, cfg.col_h, cfg.cover, cfg.col_bar_dia,
                4, 4, col_side_bars, cfg.col_stirrup_dia,
                f"柱截面配筋图 ({cfg.col_b*1000:.0f}x{cfg.col_h*1000:.0f})",
                cfg.col_stirrup_spacing)
    
    fig.text(0.5, 0.02, "注: 三级钢(HRB400)符号为圆圈内两条水平线", 
            ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    plt.suptitle(f"截面配筋详图 - 混凝土{cfg.concrete_grade} / 钢筋{cfg.steel_grade}", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==================== 分析主函数 ====================

def run_modal_and_gravity(cfg: RCFrameConfig, out_dir: Path) -> Dict[str, object]:
    """运行模态和重力分析"""
    print("  [1/5] 建立模型...")
    coords, elements = build_frame_model(cfg)
    
    print("  [2/5] 施加重力荷载...")
    apply_gravity_loads(cfg, elements)
    
    print("  [3/5] 运行重力分析...")
    run_gravity_analysis(cfg)
    
    print("  [4/5] 特征值分析...")
    periods = estimate_periods(cfg)
    mode_shapes = get_mode_shapes(cfg, min(3, len(periods)))
    drifts = story_drifts(cfg)
    vertical_reaction = base_reaction_y(cfg)
    
    print("  [5/5] 获取内力并绘图...")
    element_forces = get_element_forces(elements, coords)
    write_internal_force_outputs(element_forces, out_dir, "04a_gravity")

    plot_model(coords, elements, cfg, out_dir / "01_model_geometry.png", "钢筋混凝土框架结构模型")
    plot_mode_shapes(cfg, periods, mode_shapes, out_dir / "02_mode_shapes.png")
    plot_modal_analysis(periods, cfg, out_dir / "03_modal_analysis.png")
    plot_deformed_shape(coords, elements, collect_node_displacements(coords),
                       out_dir / "04_gravity_deformed_shape.png", "重力作用下变形图")
    plot_story_drifts_line(drifts, cfg, out_dir / "05_gravity_story_drifts.png", "重力作用下层间位移角")
    
    # 绘制重力荷载下的内力图
    plot_all_internal_forces(coords, elements, element_forces, cfg, out_dir, "04a_gravity")

    modal_rows = [{"mode": i + 1, "period_s": period} for i, period in enumerate(periods)]
    write_csv(out_dir / "modal_periods.csv", modal_rows)
    
    return {
        "periods_s": periods,
        "gravity_base_vertical_reaction_kN": vertical_reaction,
        "gravity_max_abs_lateral_drift": max(abs(v) for v in drifts),
    }


def run_time_history(cfg: RCFrameConfig, out_dir: Path) -> Dict[str, object]:
    """运行时程分析"""
    print("  [1/5] 重建模型...")
    coords, elements = build_frame_model(cfg)
    apply_gravity_loads(cfg, elements)
    run_gravity_analysis(cfg)
    
    print("  [2/5] 设置阻尼...")
    periods = estimate_periods(cfg)
    alpha_m, beta_k = set_rayleigh_damping(cfg, periods)

    print("  [3/5] 生成地震波...")
    time, accel = synthetic_ground_motion(cfg)
    plot_ground_motion(time, accel, out_dir / "06_ground_motion.png", cfg)
    
    print("  [4/5] 施加地震作用...")
    ops.timeSeries("Path", 10, "-dt", cfg.th_dt, "-values", *accel.tolist())
    ops.pattern("UniformExcitation", 10, 1, "-accel", 10)
    setup_transient_analysis(cfg.th_dt, cfg)

    print("  [5/5] 运行时程分析...")
    rows: List[Dict[str, float]] = []
    max_drift = 0.0
    total_steps = len(time) - 1
    max_shear_step = 0
    max_shear_val = 0
    
    for i in range(1, len(time)):
        ok = analyze_one_step_improved(cfg.th_dt, cfg)
        if ok != 0:
            print(f"    警告: 时程分析在 t = {time[i]:.3f} s 时收敛困难")
            break
            
        drifts = story_drifts(cfg)
        floor_disps = floor_average_displacements(cfg)
        floor_vels = floor_average_velocities(cfg)
        floor_accels = floor_average_accelerations(cfg)
        max_drift = max(max_drift, max(abs(v) for v in drifts))
        roof_node = node_tag(cfg.n_stories, cfg.n_bays, cfg)
        
        current_shear = abs(base_shear_x(cfg))
        if current_shear > max_shear_val:
            max_shear_val = current_shear
            max_shear_step = i
        
        row_data = {
            "time_s": ops.getTime(),
            "ground_accel_m_s2": float(accel[i]),
            "roof_disp_m": ops.nodeDisp(roof_node, 1),
            "base_shear_kN": base_shear_x(cfg),
        }
        
        for j in range(1, cfg.n_stories + 1):
            row_data[f"story_{j}_disp_m"] = floor_disps[j]
            row_data[f"story_{j}_vel_m_s"] = floor_vels[j]
            row_data[f"story_{j}_accel_m_s2"] = floor_accels[j]
            row_data[f"story_{j}_drift"] = drifts[j-1]
        
        rows.append(row_data)
        
        if i % (total_steps // 10) == 0:
            print(f"    进度: {i}/{total_steps} ({100*i/total_steps:.0f}%)")
    
    # 在最大基底剪力时刻绘制内力图
    seismic_hinge_summary: Dict[str, object] = {}
    if max_shear_step > 0 and max_shear_step < len(rows):
        print(f"  绘制t={time[max_shear_step]:.2f}s时刻内力图...")
        element_forces = get_element_forces(elements, coords)
        write_internal_force_outputs(element_forces, out_dir, "07a_seismic_max")
        plot_all_internal_forces(coords, elements, element_forces, cfg, out_dir, "07a_seismic_max")
        seismic_hinges = collect_plastic_hinges(element_forces, cfg)
        seismic_hinge_summary = write_plastic_hinge_outputs(
            seismic_hinges, cfg, out_dir, "07b_seismic_max"
        )
        plot_plastic_hinge_distribution(
            coords, elements, seismic_hinges, cfg,
            out_dir / "07b_seismic_max_plastic_hinges.png",
            "时程最大基底剪力时刻塑性铰分布"
        )

    write_csv(out_dir / "time_history_response.csv", rows)
    plot_time_history(rows, cfg, out_dir / "07_time_history_response.png")
    plot_floor_responses(rows, cfg, out_dir)
    
    if rows:
        max_drifts = []
        for j in range(1, cfg.n_stories + 1):
            story_max = max(abs(r[f"story_{j}_drift"]) for r in rows)
            max_drifts.append(story_max)
        plot_story_drifts_line(max_drifts, cfg, out_dir / "10_time_history_max_drifts.png", 
                          "时程分析最大层间位移角", limit=1.0 / 550.0)

    return {
        "periods_s": periods,
        "rayleigh_alpha_m": alpha_m,
        "rayleigh_beta_k": beta_k,
        "pga_g": cfg.th_pga_g,
        "max_abs_roof_disp_m": max(abs(r["roof_disp_m"]) for r in rows) if rows else 0,
        "max_abs_roof_disp_mm": max(abs(r["roof_disp_m"]) for r in rows) * 1000 if rows else 0,
        "max_abs_interstory_drift": max_drift,
        "max_abs_interstory_drift_ratio": f"1/{int(1/max_drift)}" if max_drift > 0 else "N/A",
        "elastic_drift_limit": "1/550",
        "elastic_drift_check": "PASS" if max_drift <= 1.0 / 550.0 else "CHECK",
        "plastic_hinge_summary": seismic_hinge_summary,
    }


def run_frequent_seismic_static(cfg: RCFrameConfig, out_dir: Path) -> Dict[str, object]:
    """采用底部剪力法进行频遇地震作用静力分析。"""
    print("  [1/4] 重建模型并完成重力分析...")
    coords, elements = build_frame_model(cfg)
    apply_gravity_loads(cfg, elements)
    run_gravity_analysis(cfg)

    print("  [2/4] 计算等效侧向地震力...")
    weights_by_floor = []
    for story in range(1, cfg.n_stories + 1):
        floor_weight = 0.0
        for bay in range(cfg.n_bays + 1):
            mass = ops.nodeMass(node_tag(story, bay, cfg), 1)
            floor_weight += mass * cfg.g
        weights_by_floor.append(floor_weight)

    heights = np.array(story_elevations(cfg)[1:])
    weights = np.array(weights_by_floor)
    lateral_factors = weights * heights
    lateral_factors = lateral_factors / lateral_factors.sum()
    seismic_weight = float(weights.sum())
    base_shear_target = cfg.th_pga_g * seismic_weight

    ops.timeSeries("Linear", 30)
    ops.pattern("Plain", 30, 30)
    for story, floor_factor in enumerate(lateral_factors, start=1):
        floor_force = base_shear_target * float(floor_factor)
        for bay in range(cfg.n_bays + 1):
            trib = 0.5 if bay in (0, cfg.n_bays) else 1.0
            floor_sum = cfg.n_bays
            ops.load(node_tag(story, bay, cfg), float(floor_force * trib / floor_sum), 0.0, 0.0)

    print("  [3/4] 运行等效静力分析...")
    setup_static_analysis(("LoadControl", 1.0 / 20), cfg)
    for step in range(20):
        ok = analyze_one_step_improved(cfg=cfg)
        if ok != 0:
            raise RuntimeError(f"频遇地震底部剪力法分析在第 {step + 1}/20 步失败")

    print("  [4/4] 输出频遇地震结果...")
    drifts = story_drifts(cfg)
    element_forces = get_element_forces(elements, coords)
    write_internal_force_outputs(element_forces, out_dir, "06a_frequent_seismic")
    plot_all_internal_forces(coords, elements, element_forces, cfg, out_dir, "06a_frequent_seismic")
    plot_deformed_shape(
        coords, elements, collect_node_displacements(coords),
        out_dir / "06a_frequent_seismic_deformed_shape.png", "频遇地震底部剪力法变形图"
    )
    plot_story_drifts_line(
        drifts, cfg, out_dir / "06a_frequent_seismic_drifts.png",
        "频遇地震底部剪力法层间位移角", limit=1.0 / 550.0
    )

    return {
        "method": "equivalent base shear method",
        "seismic_weight_kN": seismic_weight,
        "base_shear_target_kN": base_shear_target,
        "base_shear_actual_kN": base_shear_x(cfg),
        "max_abs_interstory_drift": max(abs(v) for v in drifts),
        "max_abs_interstory_drift_ratio": f"1/{int(1/max(abs(v) for v in drifts))}" if max(abs(v) for v in drifts) > 0 else "N/A",
        "elastic_drift_limit": "1/550",
        "elastic_drift_check": "PASS" if max(abs(v) for v in drifts) <= 1.0 / 550.0 else "CHECK",
    }


def run_pushover(cfg: RCFrameConfig, out_dir: Path) -> Dict[str, object]:
    """运行推覆分析"""
    print("  [1/4] 重建模型...")
    coords, elements = build_frame_model(cfg)
    apply_gravity_loads(cfg, elements)
    run_gravity_analysis(cfg)

    print("  [2/4] 计算侧向力分布...")
    ops.timeSeries("Linear", 20)
    ops.pattern("Plain", 20, 20)
    
    weights_by_floor = []
    for story in range(1, cfg.n_stories + 1):
        floor_weight = 0.0
        for bay in range(cfg.n_bays + 1):
            mass = ops.nodeMass(node_tag(story, bay, cfg), 1)
            floor_weight += mass * cfg.g
        weights_by_floor.append(floor_weight)

    heights = story_elevations(cfg)[1:]
    lateral_factors = np.array(weights_by_floor) * np.array(heights)
    lateral_factors = lateral_factors / lateral_factors.sum()
    
    for story, floor_factor in enumerate(lateral_factors, start=1):
        for bay in range(cfg.n_bays + 1):
            trib = 0.5 if bay in (0, cfg.n_bays) else 1.0
            floor_sum = cfg.n_bays
            ops.load(node_tag(story, bay, cfg), float(floor_factor * trib / floor_sum), 0.0, 0.0)

    print("  [3/4] 运行推覆分析...")
    roof_node = node_tag(cfg.n_stories, cfg.n_bays, cfg)
    target_disp = cfg.pushover_target_drift * cfg.roof_height
    
    rows: List[Dict[str, float]] = []
    current_step = cfg.pushover_step
    min_step = cfg.min_step
    current_disp = 0.0
    step_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    print(f"    目标位移: {target_disp*1000:.1f} mm")
    
    while current_disp < target_disp:
        remaining = target_disp - current_disp
        step = min(current_step, remaining)
        
        setup_static_analysis(("DisplacementControl", roof_node, 1, step), cfg)
        ok = analyze_one_step_improved(cfg=cfg)
        
        if ok == 0:
            step_count += 1
            consecutive_failures = 0
            
            drifts = story_drifts(cfg)
            roof_disp = ops.nodeDisp(roof_node, 1)
            current_disp = roof_disp
            
            rows.append({
                "step": step_count,
                "roof_disp_m": roof_disp,
                "roof_drift_ratio": roof_disp / cfg.roof_height,
                "base_shear_kN": base_shear_x(cfg),
                "max_abs_interstory_drift": max(abs(v) for v in drifts),
                **{f"story_{j + 1}_drift": drifts[j] for j in range(cfg.n_stories)},
            })
            
            current_step = min(current_step * 1.2, cfg.pushover_step * 2)
            
            if step_count % 50 == 0:
                print(f"    步骤 {step_count}: 位移 {roof_disp*1000:.2f} mm, 基底剪力 {rows[-1]['base_shear_kN']:.1f} kN")
        else:
            consecutive_failures += 1
            current_step = current_step / 2.0
            
            if current_step < min_step:
                if consecutive_failures >= max_consecutive_failures:
                    print(f"    连续{max_consecutive_failures}次失败，停止分析")
                    break
                current_step = min_step
    
    print(f"  [4/4] 输出结果...")
    print(f"    完成步数: {step_count}")
    print(f"    最终位移: {current_disp*1000:.2f} mm")
    
    # 绘制最终状态的内力图
    element_forces = get_element_forces(elements, coords)
    write_internal_force_outputs(element_forces, out_dir, "11a_pushover_final")
    plot_all_internal_forces(coords, elements, element_forces, cfg, out_dir, "11a_pushover_final")
    pushover_hinges = collect_plastic_hinges(element_forces, cfg)
    plastic_hinge_summary = write_plastic_hinge_outputs(
        pushover_hinges, cfg, out_dir, "11b_pushover_final"
    )
    plot_plastic_hinge_distribution(
        coords, elements, pushover_hinges, cfg,
        out_dir / "11b_pushover_final_plastic_hinges.png",
        "推覆最终状态塑性铰分布"
    )
    
    write_csv(out_dir / "pushover_curve.csv", rows)
    
    # 使用新的绘图函数（带面积标注）
    yield_info = plot_pushover_curve_with_energy(rows, cfg, out_dir / "11_pushover_curve.png")
    
    plot_deformed_shape(coords, elements, collect_node_displacements(coords),
                       out_dir / "12_pushover_final_deformed_shape.png", "推覆分析最终变形图")
    
    if rows:
        final_drifts = [rows[-1][f"story_{j + 1}_drift"] for j in range(cfg.n_stories)]
        plot_story_drifts_line(final_drifts, cfg, out_dir / "13_pushover_final_drifts.png", 
                          "推覆分析最终层间位移角", limit=1.0 / 50.0)

    max_drift = max((r["max_abs_interstory_drift"] for r in rows), default=0.0)
    reached = current_disp >= target_disp * 0.95
    
    result = {
        "target_roof_disp_m": target_disp,
        "target_roof_disp_mm": target_disp * 1000,
        "actual_roof_disp_mm": current_disp * 1000,
        "steps_completed": len(rows),
        "target_reached": reached,
        "max_base_shear_kN": max((abs(r["base_shear_kN"]) for r in rows), default=0.0),
        "max_abs_interstory_drift": max_drift,
        "max_abs_interstory_drift_ratio": f"1/{int(1/max_drift)}" if max_drift > 0 else "N/A",
        "elastoplastic_drift_limit": "1/50",
        "elastoplastic_drift_check": "PASS" if max_drift <= 1.0 / 50.0 else "CHECK",
        "plastic_hinge_definition": plastic_hinge_definition(cfg),
        "plastic_hinge_summary": plastic_hinge_summary,
    }
    
    result.update(yield_info)
    return result


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("    钢筋混凝土框架结构分析程序 (修正版 v2)")
    print("    OpenSeesPy Nonlinear Analysis")
    print("    - 框架内力图直接绘制")
    print("    - Pushover面积等效可视化")
    print("=" * 70)
    
    configure_chinese_font()
    
    root = Path(__file__).resolve().parent
    out_dir = root / "analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {out_dir}")

    cfg = RCFrameConfig(
        n_bays=3,
        bay_width=6.0,
        story_heights=(3.9, 3.6, 3.6, 3.6, 3.6),
        tributary_width=6.0,
        beam_b=0.30,
        beam_h=0.65,
        col_b=0.55,
        col_h=0.55,
        th_pga_g=0.10,
        th_duration=30.0,
        th_dt=0.02,
        pushover_target_drift=0.04,
        pushover_step=0.0005,
        max_iterations=100,
        tolerance=1.0e-5,
        min_step=0.0001,
    )
    
    write_json(out_dir / "model_config.json", asdict(cfg))
    
    print(f"\n模型信息:")
    print(f"  楼层数: {cfg.n_stories}层")
    print(f"  跨数: {cfg.n_bays}跨")
    print(f"  总高度: {cfg.roof_height:.1f}m")
    print(f"  总宽度: {cfg.total_width:.1f}m")
    print(f"  梁截面: {cfg.beam_b*1000:.0f}x{cfg.beam_h*1000:.0f}mm")
    print(f"  柱截面: {cfg.col_b*1000:.0f}x{cfg.col_h*1000:.0f}mm")

    # Step 0: 结构信息汇总
    print("\n[Step 0] 绘制结构信息汇总...")
    plot_structure_info(cfg, out_dir / "00_structure_info.png")

    # Step 1: 材料本构
    print("\n[Step 1] 绘制材料本构曲线...")
    constitutive = concrete_constitutive_info(cfg)
    steel_info = steel_constitutive_info(cfg)
    write_json(out_dir / "concrete_constitutive.json", constitutive)
    write_json(out_dir / "steel_constitutive.json", steel_info)
    plot_concrete_constitutive(cfg, out_dir / "00a_concrete_constitutive.png")
    plot_steel_constitutive(cfg, out_dir / "00b_steel_constitutive.png")
    
    # Step 2: 配筋图
    print("\n[Step 2] 绘制截面配筋图...")
    plot_section_reinforcement(cfg, out_dir / "00c_section_reinforcement.png")

    # Step 3: 模态和重力分析
    print("\n[Step 3] 模态和重力分析...")
    modal_results = run_modal_and_gravity(cfg, out_dir)
    print(f"  基本周期 T1 = {modal_results['periods_s'][0]:.4f}s")
    print(f"  重力反力 = {modal_results['gravity_base_vertical_reaction_kN']:.1f}kN")

    # Step 4: 频遇地震底部剪力法
    print("\n[Step 4] 频遇地震底部剪力法分析...")
    frequent_seismic_results = run_frequent_seismic_static(cfg, out_dir)
    print(f"  目标基底剪力 = {frequent_seismic_results['base_shear_target_kN']:.1f}kN")
    print(f"  最大层间位移角 = {frequent_seismic_results['max_abs_interstory_drift_ratio']}")
    print(f"  弹性验算: {frequent_seismic_results['elastic_drift_check']}")

    # Step 5: 时程分析
    print("\n[Step 5] 时程分析...")
    th_results = run_time_history(cfg, out_dir)
    print(f"  最大屋顶位移 = {th_results['max_abs_roof_disp_mm']:.2f}mm")
    print(f"  最大层间位移角 = {th_results['max_abs_interstory_drift_ratio']}")
    print(f"  弹性验算: {th_results['elastic_drift_check']}")

    # Step 6: 推覆分析
    print("\n[Step 6] 推覆分析...")
    pushover_results = run_pushover(cfg, out_dir)
    print(f"  完成步数: {pushover_results['steps_completed']}")
    print(f"  实际位移: {pushover_results['actual_roof_disp_mm']:.1f}mm")
    print(f"  最大基底剪力 = {pushover_results['max_base_shear_kN']:.1f}kN")
    print(f"  弹塑性验算: {pushover_results['elastoplastic_drift_check']}")
    if 'yield_disp_mm' in pushover_results:
        print(f"  等效屈服位移 = {pushover_results['yield_disp_mm']:.1f}mm")
        print(f"  等效屈服力 = {pushover_results['yield_force_kN']:.1f}kN")
        print(f"  延性系数 = {pushover_results['ductility_ratio']:.2f}")
        print(f"  实际曲线面积 = {pushover_results['area_actual_kN_mm']:.1f} kN·mm")
        print(f"  双线性面积 = {pushover_results['area_bilinear_kN_mm']:.1f} kN·mm")

    # 汇总结果
    summary = {
        "model_info": {
            "n_stories": cfg.n_stories,
            "n_bays": cfg.n_bays,
            "total_height_m": cfg.roof_height,
            "total_width_m": cfg.total_width,
            "beam_section": f"{cfg.beam_b*1000:.0f}x{cfg.beam_h*1000:.0f}mm",
            "column_section": f"{cfg.col_b*1000:.0f}x{cfg.col_h*1000:.0f}mm",
        },
        "units": "kN, m, s",
        "modal_and_gravity": modal_results,
        "frequent_seismic": frequent_seismic_results,
        "time_history": th_results,
        "pushover": pushover_results,
    }
    write_json(out_dir / "analysis_summary.json", summary)

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"\n结果文件夹: {out_dir}")
    print("\n生成的图表文件:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  [图] {f.name}")
    print("\n数据文件:")
    for f in sorted(out_dir.glob("*.json")):
        print(f"  [JSON] {f.name}")
    for f in sorted(out_dir.glob("*.csv")):
        print(f"  [CSV] {f.name}")
    
    print(f"\n共生成 {len(list(out_dir.glob('*.png')))} 张图表")


if __name__ == "__main__":
    main()
