"""ERTM 项目主入口。

常用命令
--------
    python main.py                  # 使用 config/default.json 跑完整 RTM 流水线
    python main.py --check          # 只做结构/接口自检
    python main.py --config my.json # 使用自定义配置
    python main.py --output out     # 覆盖输出目录
    python main.py --version        # 查看版本

PyCharm 中直接右键 Run ``main.py`` 即可；工作目录会自动定位到项目根目录。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# 允许从任意工作目录直接运行（PyCharm 默认工作目录通常是项目根）。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ertm  # noqa: E402
from ertm.config import load_config  # noqa: E402
from ertm.diagnostics import format_checks, run_checks  # noqa: E402
from ertm.exceptions import ERTMError  # noqa: E402
from ertm.logging_utils import get_logger, setup_logging  # noqa: E402
from ertm.pipeline import ERTMPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ertm",
        description="ERTM 弹性波逆时偏移（双程波 RTM + 井中多分量）",
    )
    parser.add_argument(
        "--config",
        default="config/default.json",
        help="JSON 配置文件路径（默认 config/default.json）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="覆盖配置中的输出根目录",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="覆盖配置中的日志级别",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只运行项目结构与模块接口自检，不执行流水线",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="打印版本并退出",
    )
    return parser


def _resolve_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _print_summary(result) -> None:
    summary = result.summary()
    print("\n" + "=" * 72)
    print("ERTM 运行完成")
    print("=" * 72)
    print(f"项目        : {summary['project']}")
    print(f"模型形状    : {summary['model_shape']} (nz, nx)")
    print(f"正演引擎    : {summary['engine']}")
    print(f"成像条件    : {summary['condition']}")
    print(f"炮数/道数   : {summary['n_shots']} / {summary['n_receivers']}")
    print(f"时间采样    : {summary['n_time_samples']}")
    print(f"成像模式    : {summary['image_mode']}")
    print(f"占位实现    : {summary['placeholder']}")
    print(f"耗时        : {summary['elapsed_s']} s")
    print(f"运行报告    : {summary['report']}")
    print("-" * 72)
    peak_x = summary["metrics"].get("image_peak_x")
    peak_z = summary["metrics"].get("image_peak_z")
    if peak_x is not None and peak_z is not None:
        print(f"最强成像点  : x={peak_x:.0f} m, z={peak_z:.0f} m")
    print("产物:")
    for path in summary["artifacts"]:
        print(f"  - {path}")
    print("=" * 72 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"ERTM {ertm.__version__}")
        return 0

    # 先建立一个基础日志（配置文件可能出错，仍需有日志可看）。
    log_dir = PROJECT_ROOT / "output" / "logs"
    setup_logging(level=args.log_level or "INFO", log_dir=log_dir, force=True)
    logger = get_logger("main")

    if args.check:
        results = run_checks(PROJECT_ROOT)
        print(format_checks(results))
        failed = [item for item in results if not item.ok]
        if failed:
            logger.error("结构自检失败: %d 项", len(failed))
            return 1
        logger.info("结构自检通过: %d 项", len(results))
        print(f"\n结构自检通过：{len(results)} 项全部 OK。")
        return 0

    try:
        config_path = _resolve_config_path(args.config)
        config = load_config(config_path)
        if args.output:
            config = replace(config, output=replace(config.output, root=args.output))
        if args.log_level:
            config = replace(
                config,
                logging=replace(config.logging, level=args.log_level),
            )

        # 根据配置重新设置日志级别和输出目录。
        output_root = Path(config.output.root).expanduser()
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        setup_logging(
            level=config.logging.level,
            log_dir=output_root / "logs",
            log_file=config.logging.file_name,
            force=True,
        )
        logger = get_logger("main")
        logger.info("配置加载完成: %s", config_path)

        pipeline = ERTMPipeline(config)
        result = pipeline.run()
        _print_summary(result)
        return 0
    except ERTMError as exc:
        logger.error("运行失败: %s", exc)
        print(f"\n[ERTM 错误] {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - 入口层需要兜底并保留堆栈
        logger.exception("未预期异常: %s", exc)
        print(f"\n[未预期异常] {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
